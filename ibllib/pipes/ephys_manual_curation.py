""" Ephys manual creation
Module used to register and upload new cluster datasets following manual curation in Phy

Examples
--------
Upload manual curation results for a given session and probe

>>> mc = ManualCuration(eid, 'probe', 'mf', spike_sorter='pykilosort', one=one)
>>> mc.process('/path_to_manually_curated_spike_cluster_file/spikes.clusters.npy')

Notes
-----
Current implentation DOES NOT
- Support upload from local server
- Support sessions with multiple spike sorting revisions
- Extract waveforms, do not have access to the raw data file
"""

import logging
import numpy as np
from pathlib import Path
import shutil
import tarfile

from ibllib.ephys import ephysqc, spikes
from ibllib.pipes.ephys_tasks import SpikeSorting
import one.alf.io as alfio
import phylib.io.alf

_logger = logging.getLogger(__name__)


CLUSTER_FILES = [
    'clusters.channels.npy',
    'clusters.depths.npy',
    'clusters.amps.npy',
    'clusters.peakToTrough.npy',
    'clusters.metrics.pqt',
    'clusters.waveforms.npy',
    'clusters.waveformChannels.npy',
    'spikes.clusters.npy'
]


class ManualCuration:

    def __init__(self, eid, pname, namespace, spike_sorter='pykilosort', one=None):

        """
        Class to extract and upload new cluster datasets following manual curation of spikesorting using Phy. During
        the manual curation process the assignment of clusters in the spikes.clusters.npy is changed. The spikes from a
        single cluster can either be split into two new clusters, or the spikes from two or more clusters can be merged
        to form a new cluster. The user must provide a spikes.clusters.npy file output from phy with the new cluster
        assignments.

        The new cluster datasets are computed using the provided spikes.clusters.npy file and the original spikesorting
        output. There are two spikesorting outputs that can be used to compute the new datasets,
        - tar spikesorting - spikesorting output directly from pykilosort prior to alf conversion
        - alf spikesorting - spikesorting output following alf conversion (used by SpikeSortingLoader)

        This class will first attempt to use the tar spikesorting output to compute the new datasets, if
        this is not available, however, it will compute the new datasets from the alf spikesorting output.

        There is not a one-to-one correspondence when computing the new datasets from the two different spikesorting outputs.
        This is because the template waveforms in the alf spikesorting output are sparse (only store top 32 channels) and
        scaled (whitened an scaled according to max template amplitude) compared to the waveforms in the tar spikesorting output.

        The following datasets will therefore differ depending on which spikesorting output is used,
        - clusters.peakToTrough.npy - can vary for all units, due to the peak to trough being computed prior to scaling
        (tar output) vs post scaling (alf output)
        - clusters.waveforms.npy - will be different for clusters that have been merged during the manual curation process, due to
        only having access to 32 channels and the waveforms already being scaled
        - clusters.waveformsChannels.npy - will be different for clusters that have been merged during the manual curation process

        The outputs for the following datasets should be the same regardless of which initial spikesorting output is used
        - clusters.channels.npy
        - clusters.amps.npy
        - clusters.depths.npy
        - clusters.metrics.pqt

        A namespace is inserted into the file name to differentiate the new datasets from the origial alf datasets, e,g
        _av_spikes.clusters.npy. The convention is that the namespace should be the initials of the user who performed the
        manual creation

        Parameters
        ----------
        eid : str
            A session experiment id
        pname : str
            A probe label, e.g 'probe00'
        namespace : str
            The namespace to add to the new datasets, initials or the user e.g mf
        spike_sorter : str
            The spikesorting to use, either '' for ks_matlab or 'pykilosort' for pykilosort
        one: ONE
            An one instance
        """

        self.one = one
        self.eid = eid
        self.spike_sorter = spike_sorter
        self.pname = pname
        self.namespace = namespace
        self.session_path = self.one.eid2path(self.eid)
        self.alf_collection = f'alf/{self.pname}/{self.spike_sorter}'
        self.bin_path = self.session_path.joinpath('raw_ephys_data', self.pname)

        # Location that acts as a temporary directory for extraction
        self.conversion_path = self.session_path.joinpath('manual_curation', self.pname)
        self.conversion_path.mkdir(exist_ok=True, parents=True)
        self.out_path = self.conversion_path.joinpath('alf_curated')
        self.out_path.mkdir(exist_ok=True, parents=True)

        self.cluster_file = None

    def process(self, cluster_file):
        """
        Main method to call in order to extract the new clusters datasets. This method will download the necessary data,
        extract the new files, add the namespace to the file names, copy the relevant files to the correct location and
        finally register and upload the files

        Parameters
        ----------
        cluster_file : Path, str
            Path to the manually curated spikes.clusters file output from Phy
        """

        self.cluster_file = Path(cluster_file)

        # Check to see if there are multiple revisions for this eid, pname, spikesorter combination. If there are
        # eventually we need the user to specify which revision was used, currently not yet implemented
        revision_check = self.check_revisions()
        if not revision_check:
            # If the revision check fails, return, we don't proceed
            return

        # Double check that the manually curated clusters is not the same as the original clusters file, if it is
        # no manual creation has taken place so there is no need to proceed
        cluster_check = self.check_clusters()
        if not cluster_check:
            # If the cluster check fails, return, we don't proceed
            return

        # Download the dat necessary for extraction of the new datasets, we are returned which spikesorting output
        # should be used for extraction, either tar_ss or alf_ss
        ss_type = self.download_data()

        # Depending on the spikesorting output used, extract the new datasets
        if ss_type == 'tar_ss':
            self.extract_from_tar_data()
        else:
            self.extract_from_alf_data()

        # Compute the new clusters metrics for the manually curated clusters
        self.compute_cluster_metrics()

        # Rename the files, add the user namespace to the beginning of each file
        files = self.rename_files()

        # Move the files from the temp directory to the final alf location
        files = self.move_files(files)

        # Remove the temp folders that were used for extraction
        self.cleanup()

        # Register and upload the files
        self.upload_files(files)

    def check_cluster_file(self):
        """
        Check that the cluster assignment in the user provided manually curated spikes.clusters file differs from the
        original cluster assignment in the spikes.templates file. If it does not differ then it indicates that no
        manual creation has been performed.

        Returns
        -------
        bool:
            Whether or not the cluster check passes
        """

        clusters = np.load(self.cluster_file)
        templates = self.one.load_dataset(self.eid, 'spikes.templates.npy', collection=self.alf_collection)
        if np.array_equal(clusters, templates):
            _logger.warning('The cluster assignment in the provided clusters file does not differ from the original'
                            'cluster assignment. Are you sure this is the correct manually curated file')
            return False
        else:
            return True

    def check_revisions(self):
        """
        Check that there are not mulitple revisions for the eid, pname, spikesorter combination used. If there are
        mulitple revisions we can not be sure which revision was used for manual curation.

        Returns
        -------
        bool:
            Whether or not the revision check passes
        """

        revisions = self.one.list_datasets(self.eid, 'spikes.templates.npy', collection=self.alf_collection)
        if len(revisions) > 1:
            # TODO eventually change to handle revisions
            _logger.warning('Multiple spikesorting revisions found for this session, uploading curated data for these'
                            'sessions is currently not supported')
            return False
        else:
            return True

    def _get_tar_spikesorting(self):
        """
        Downloads the tar spikesorting data files required for extraction of new datasets and untars them. If the
        tar spikesorting data is not available, none is returned

        Returns
        -------
        string, none:
            If the tar spikesorting is available returns string `tar_ss`, if not available returns None
        """

        ss_tar_collection = f'spikesorters/{self.spike_sorter}/{self.pname}'
        ss_tar_fname = '_kilosort_output.raw.tar'
        ss_tar_dset = self.one.list_datasets(self.eid, ss_tar_fname, collection=ss_tar_collection)
        if len(ss_tar_dset) >= 1:
            ss_tar_output = self.conversion_path.joinpath('tar_ss')
            ss_tar_output.mkdir(exist_ok=True, parents=True)
            # If the tar dataset is available download and extract the tar output
            ss_tar_file = self.one.load_dataset(self.eid, ss_tar_fname, collection=ss_tar_collection,
                                                download_only=True)
            with tarfile.open(ss_tar_file, 'r') as tar_dir:
                tar_dir.extractall(path=ss_tar_output)

            return 'tar_ss'

    def _get_alf_spikesorting(self):

        """
        Downloads the alf spikesorting data files required for extraction of new datasets

        Returns
        -------
        string:
            Returns 'alf_ss'
        """

        # Want to avoid re-downloading the spikes.clusters just in case
        ss_alf_dsets = ['spikes.times.npy',
                        'spikes.amplitudes.npy',
                        'spikes.templates.npy',
                        'spikes.depths.npy',
                        'channels.localCoordinates.npy',
                        'channels.rawInd.npy',
                        'templates.waveforms.npy',
                        'templates.waveformChannels.npy']

        ss_alf_output = self.conversion_path.joinpath('alf_ss')
        ss_alf_output.mkdir(exist_ok=True, parents=True)

        ss_alf_files = self.one.load_datasets(self.eid, ss_alf_dsets, collection=self.alf_collection, download_only=True)
        for file in ss_alf_files:
            shutil.copy(file, ss_alf_output.joinpath(file.name))

        return 'alf_ss'

    def _get_metadata(self):
        """
        Downloads spikeglx metadata files required for extraction of new datasets
        """

        _ = self.one.load_dataset(self.eid, dataset='*.ap.meta', collection=f'raw_ephys_data/{self.pname}',
                                  download_only=True)

    def download_data(self):
        """
        Downloads all data required to extract the new manually curated datasets. It will first attempt to download the
        tar spikesorting output. If this is not available or if this spikesorting differs from the alf spikesorting output
        then the alf spikesorting will be downloaded. Also downloads necessary spikeglx ap metadata files for the probe.

        Returns
        -------
        string:
            'tar_ss' or 'alf_ss' depending on which spikesorting output was downloaded
        """

        ss_type = self._get_tar_spikesorting()

        if ss_type is None:
            ss_type = self._get_alf_spikesorting()
        else:
            # Check to make sure the tar dataset is the latest spikesorting, if not default back to
            tar_status = self._check_tar_spikesorting()
            if not tar_status:
                ss_type = self._get_alf_spikesorting()

        self._get_metadata()

        return ss_type

    def _check_tar_spikesorting(self):
        """
        Checks that the spikesorting result stored in the tar file is the same as that stored in the alf
        collection. These may differ for the BWM and Repro Ephys reruns where the tar file for the new spikesorting
        was not uploaded. If this is the case we extract the new datasets from the alf spikesorting

        Returns
        -------
        bool:
            Whether or not the tar spikesorting is equivalent to the alf spikesorting
        """

        alf_templates = self.one.load_dataset(self.eid, 'spikes.templates.npy', collection=self.alf_collection)
        tar_templates = np.load(self.conversion_path.joinpath('tar_ss', 'spike_templates.npy'))

        return np.array_equal(alf_templates, tar_templates)

    def extract_from_tar_data(self):
        """
        Extracts the new manually curated datasets from the tar spikesorting output
        """

        # Move the new spike cluster file to correct location
        shutil.copy(self.cluster_file, self.conversion_path.joinpath('tar_ss', 'spike_clusters.npy'))

        spikes.ks2_to_alf(
            self.conversion_path.joinpath('tar_ss'),
            bin_path=self.bin_path,
            out_path=self.out_path,
            bin_file=None,
            ampfactor=SpikeSorting._sample2v(next(self.bin_path.glob('*.meta'))),
        )

    def extract_from_alf_data(self):
        """
        Extracts the new manually curated datasets from the alf spikesorting output
        """

        # Move the new spike cluster file to correct location
        shutil.copy(self.cluster_file, self.conversion_path.joinpath('alf_ss', 'spikes.clusters.npy'))

        m = ephysqc.phy_model_from_ks2_path(ks2_path=self.conversion_path.joinpath('alf_ss'), bin_path=self.bin_path)
        ac = phylib.io.alf.EphysAlfCreator(m)
        ac.out_path = self.out_path
        ac.ampfactor = 1
        ac.label = ''
        ac.dir_path = ac.out_path
        ac.make_cluster_objects()
        np.save(ac.out_path.joinpath('clusters.waveforms'), m.sparse_clusters.data)
        np.save(ac.out_path.joinpath('clusters.waveformsChannels'), m.sparse_clusters.cols)
        _ = ac.make_cluster_depths()

    def compute_cluster_metrics(self):
        """
        Computes the clusters metrics from the new clusters datasets
        """

        spikes = alfio.load_object(self.out_path, 'spikes')
        clusters = alfio.load_object(self.out_path, 'clusters')
        df_units, drift = ephysqc.spike_sorting_metrics(
            spikes.times, spikes.clusters, spikes.amps, spikes.depths,
            cluster_ids=np.arange(clusters.channels.size))
        # save as parquet file
        df_units.to_parquet(self.out_path.joinpath("clusters.metrics.pqt"))

    def rename_files(self):
        """
        Adds the user initials as the namespace to the new datasets. The relevant new datasets are defined in
        CLUSTER_FILES

        Returns
        -------
        list:
            list of files with user namespace inserted
        """
        # add the namespace to the files, only to the relevant ones
        namespace_files = []
        for clu_file in CLUSTER_FILES:
            file = self.out_path.joinpath(clu_file)
            new_file = file.parent.joinpath(f'_{self.namespace}_{file.name}')
            namespace_files.append(new_file)
            shutil.move(file, new_file)

        return namespace_files

    def move_files(self, files):
        """
        Copies the files from the temporary directory where extraction was performed to there final location in the
        alf/pname/spikesorter collection

        Parameters
        ----------
        files : list
            The list of files to move

        Returns
        -------
        list
            List of new location of files following copy
        """
        copied_files = []
        for file in files:
            # TODO eventually need to account for revisions
            new_file = self.session_path.joinpath(self.alf_collection, file.name)
            copied_files.append(new_file)
            shutil.copy(file, new_file)

        return copied_files

    def upload_data(self, files):
        """
        Registers and uploads the new datasets
        Parameters
        ----------
        files : list
            The list of files to register and upload

        Returns
        -------
        dict
            one.register_dataset response for file registration
        """

        # TODO

        return

    def cleanup(self):
        """
        Removes the temporary directories that were used during the extraction process
        """

        # Remove all the subfolders in the temporary directory
        shutil.rmtree(self.conversion_path)
        # Remove the empty temporary directory
        self.conversion_path.unlink()
