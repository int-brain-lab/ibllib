import logging
from ibllib.atlas import AllenAtlas, regions_from_allen_csv
from ibllib.pipes import histology
from ibllib.ephys.neuropixel import SITES_COORDINATES
import numpy as np
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.qc import base
from oneibl.patcher import FTPPatcher
from ibllib.qc.base import CRITERIA as CRITERIA_BASE

_log = logging.getLogger('ibllib')
CRITERIA = {"PASS": 0.8}


class AlignmentQC(base.QC):
    """
    Class that is used to update the extended_qc of the probe insertion fields with the results
    from the ephys alignment procedure
    """
    def __init__(self, probe_id, one=None, brain_atlas=None, channels=True):
        super().__init__(probe_id, one=one, log=_log, endpoint='insertions')

        # Data
        self.alignments = None
        self.xyz_picks = None
        self.depths = None
        self.cluster_chns = None
        self.align_keys_sorted = None

        # Metrics and passed trials
        self.sim_matrix = None
        self.criteria = CRITERIA

        # Get the brain atlas
        self.brain_atlas = brain_atlas or AllenAtlas(25)
        # Flag for uploading channels to alyx. For testing purposes
        self.channels = channels

        self.insertion = self.one.alyx.rest('insertions', 'read', id=self.eid)
        self.resolved = (self.insertion.get('json', {'temp': 0}).get('extended_qc').
                         get('alignment_resolved', False))

    def load_data(self, prev_alignments=None, xyz_picks=None, depths=None, cluster_chns=None):
        """"
        Load data required to assess alignment qc and compute similarity matrix. If no arguments
        are given load_data will fetch all the relevant data required
        """
        if not np.any(prev_alignments):
            aligned_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.eid,
                                              provenance='Ephys aligned histology track')
            if len(aligned_traj) > 0:
                self.alignments = aligned_traj[0].get('json', {})
            else:
                self.alignments = {}
                return
        else:
            self.alignments = prev_alignments

        align_keys = [*self.alignments.keys()]
        self.align_keys_sorted = sorted(align_keys, reverse=True)

        if len(self.alignments) < 2:
            return

        if not np.any(xyz_picks):
            self.xyz_picks = np.array(self.insertion['json']['xyz_picks']) / 1e6
        else:
            self.xyz_picks = xyz_picks

        if not np.any(depths):
            self.depths = SITES_COORDINATES[:, 1]
        else:
            self.depths = depths

        if not np.any(cluster_chns):
            _ = self.one.load(self.insertion['session'], dataset_types='clusters.channels',
                              download_only=True)
            self.cluster_chns = np.load(self.one.path_from_eid(self.insertion['session']).
                                        joinpath('alf', self.insertion['name'],
                                                 'clusters.channels.npy'))
        else:
            self.cluster_chns = cluster_chns

    def compute(self):
        """
        Computes the similarity matrix if > 2 alignments. If no data loaded, wraps around load_data
        to get all relevant data needed
        """

        if self.alignments is None:
            self.load_data()

        if len(self.alignments) < 2:
            self.log.info(f"Insertion {self.eid}: One or less alignment found...")
            self.sim_matrix = np.array([len(self.alignments)])
        else:
            self.log.info(f"Insertion {self.eid}: Running QC on alignment data...")
            self.sim_matrix = self.compute_similarity_matrix()

        return self.sim_matrix

    def run(self, update=True, upload_alyx=True, upload_flatiron=True):
        """
        Compute alignment_qc for a specified probe insertion and updates extended qc field in alyx.
        If alignment is resolved and upload flags set to True channels from resolved
        alignment will be updated to alyx and datasets sent to ibl-ftp-patcher to be uploaded to
        flatiron
        """
        if self.sim_matrix is None:
            self.compute()

        # Case where the alignment has already been resolved
        if self.resolved:
            self.log.info(f"Alignment for insertion {self.eid} already resolved, channels won't be"
                          f" updated. To force update of channels use "
                          f"resolve_manual method with force=True")
            results = {'alignment_count': len(self.alignments)}
            if update:
                self.update_extended_qc(results)
            results.update({'alignment_resolved': True})

        # Case where no alignments have been made
        elif np.all(self.sim_matrix == 0) and self.sim_matrix.shape[0] == 1:
            # We don't update database
            results = {'alignment_resolved': False}

        # Case where only one alignment
        elif np.all(self.sim_matrix == 1) and self.sim_matrix.shape[0] == 1:
            results = {'alignment_count': len(self.alignments),
                       'alignment_stored': self.align_keys_sorted[0],
                       'alignment_resolved': False}
            if update:
                self.update_extended_qc(results)

        # Case where 2 or more alignments and alignments haven't been resolved
        else:
            results = self.compute_alignment_status()

            if update:
                self.update_extended_qc(results)

            if results['alignment_resolved'] and (upload_alyx or upload_flatiron):
                self.upload_channels(results['alignment_stored'], upload_alyx, upload_flatiron)

        return results

    def resolve_manual(self, align_key, update=True, upload_alyx=True, upload_flatiron=True,
                       force=False):
        """
        Method to manually resolve the alignment of a probe insertion with a given alignment
        regardless of the number of alignments or the alignment qc value. Channels from specified
        alignment will be uploaded to alyx and datasets sent to ibl-ftp-patcher to be uploaded to
        flatiron. If alignment already resolved will only upload if force flag set to True
        """

        if self.sim_matrix is None:
            self.compute()
        assert align_key in self.align_keys_sorted, 'align key not recognised'

        if self.resolved == 1 and not force:
            self.log.info(f"Alignment for insertion {self.eid} already resolved, channels won't be"
                          f"updated. To overwrite stored channels with alignment {align_key} "
                          f"set 'force=True'")
            file_paths = []
        else:
            results = self.compute_alignment_status()
            results['alignment_resolved'] = True
            results['alignment_stored'] = align_key
            results['alignment_resolved_by'] = 'experimenter'

            if update:
                self.update_extended_qc(results)
                file_paths = []

            if upload_alyx or upload_flatiron:
                file_paths = self.upload_channels(align_key, upload_alyx, upload_flatiron)

        return file_paths

    def compute_similarity_matrix(self):
        """
        Computes the similarity matrix between each alignment stored in the ephys aligned
        trajectory. Similarity matrix based on number of clusters that share brain region and
        parent brain region
        """

        r = regions_from_allen_csv()

        clusters = dict()
        for iK, key in enumerate(self.align_keys_sorted):
            # Location of reference lines used for alignment
            feature = np.array(self.alignments[key][0])
            track = np.array(self.alignments[key][1])

            # Instantiate EphysAlignment object
            ephysalign = EphysAlignment(self.xyz_picks, self.depths, track_prev=track,
                                        feature_prev=feature,
                                        brain_atlas=self.brain_atlas)

            # Find xyz location of all channels
            xyz_channels = ephysalign.get_channel_locations(feature, track)
            brain_regions = ephysalign.get_brain_locations(xyz_channels)

            # Find the location of clusters along the alignment
            cluster_info = dict()
            cluster_info['brain_id'] = brain_regions['id'][self.cluster_chns]
            cluster_info['parent_id'] = r.get(ids=cluster_info['brain_id']).parent.astype(int)
            clusters.update({key: cluster_info})

        sim_matrix = np.zeros((len(self.align_keys_sorted), len(self.align_keys_sorted)))

        for ik, key in enumerate(self.align_keys_sorted):
            for ikk, key2 in enumerate(self.align_keys_sorted):
                same_id = np.where(clusters[key]['brain_id'] == clusters[key2]['brain_id'])[0]
                not_same_id = \
                    np.where(clusters[key]['brain_id'] != clusters[key2]['brain_id'])[0]
                same_parent = np.where(clusters[key]['parent_id'][not_same_id] ==
                                       clusters[key2]['parent_id'][not_same_id])[0]
                sim_matrix[ik, ikk] = len(same_id) + (len(same_parent) * 0.5)
        # Normalise
        sim_matrix_norm = sim_matrix / np.max(sim_matrix)

        return sim_matrix_norm

    def compute_alignment_status(self):
        """
        Determine whether alignments agree based on value in similarity matrix. If any alignments
        have similarity of 0.8 set the alignment to be resolved
        """
        # Set diagonals to zero so we don't use those to find max
        np.fill_diagonal(self.sim_matrix, 0)
        # self.sim_matrix[self.sim_matrix == 1] = 0
        max_sim = np.max(self.sim_matrix)

        results = {'alignment_qc': max_sim,
                   'alignment_count': self.sim_matrix.shape[0]}

        if max_sim > CRITERIA['PASS']:
            location = np.where(self.sim_matrix == max_sim)
            results.update({'alignment_stored': self.align_keys_sorted[np.min(location)]})
            results.update({'alignment_resolved': True})
            results.update({'alignment_resolved_by': 'qc'})

            # outcome = 'PASS'

        else:
            results.update({'alignment_stored': self.align_keys_sorted[0]})
            results.update({'alignment_resolved': False})

            # outcome = 'WARNING'

        return results

    def upload_channels(self, alignment_key, upload_alyx, upload_flatiron):
        """
        Upload channels to alyx and flatiron based on the alignment specified by the alignment key
        """

        feature = np.array(self.alignments[alignment_key][0])
        track = np.array(self.alignments[alignment_key][1])
        ephysalign = EphysAlignment(self.xyz_picks, self.depths,
                                    track_prev=track,
                                    feature_prev=feature,
                                    brain_atlas=self.brain_atlas)

        # Find the channels
        channels_mlapdv = np.int32(ephysalign.get_channel_locations(feature, track) * 1e6)
        channels_brainID = ephysalign.get_brain_locations(channels_mlapdv / 1e6)['id']

        # Find the clusters
        r = regions_from_allen_csv()
        clusters_mlapdv = channels_mlapdv[self.cluster_chns]
        clusters_brainID = channels_brainID[self.cluster_chns]
        clusters_brainAcro = r.get(ids=clusters_brainID).acronym

        # upload datasets to flatiron
        files_to_register = []
        if upload_flatiron:
            ftp_patcher = FTPPatcher(one=self.one)
            insertion = self.one.alyx.rest('insertions', 'read', id=self.eid)
            alf_path = self.one.path_from_eid(insertion['session']).joinpath('alf',
                                                                             insertion['name'])
            alf_path.mkdir(exist_ok=True, parents=True)

            # Make the channels.mlapdv dataset
            f_name = alf_path.joinpath('channels.mlapdv.npy')
            np.save(f_name, channels_mlapdv)
            files_to_register.append(f_name)

            # Make the channels.brainLocationIds dataset
            f_name = alf_path.joinpath('channels.brainLocationIds_ccf_2017.npy')
            np.save(f_name, channels_brainID)
            files_to_register.append(f_name)

            # Make the clusters.mlapdv dataset
            f_name = alf_path.joinpath('clusters.mlapdv.npy')
            np.save(f_name, clusters_mlapdv)
            files_to_register.append(f_name)

            # Make the clusters.brainLocationIds dataset
            f_name = alf_path.joinpath('clusters.brainLocationIds_ccf_2017.npy')
            np.save(f_name, clusters_brainID)
            files_to_register.append(f_name)

            # Make the clusters.brainLocationAcronym dataset
            f_name = alf_path.joinpath('clusters.brainLocationAcronyms_ccf_2017.npy')
            np.save(f_name, clusters_brainAcro)
            files_to_register.append(f_name)

            self.log.info("Writing datasets to FlatIron")
            ftp_patcher.create_dataset(path=files_to_register, created_by=self.one._par.ALYX_LOGIN)

        # Need to change channels stored on alyx as well as the stored key is not the same as the
        # latest key
        if upload_alyx:
            if alignment_key != self.align_keys_sorted[0]:
                histology.register_aligned_track(self.eid, channels_mlapdv / 1e6,
                                                 chn_coords=SITES_COORDINATES, one=self.one,
                                                 overwrite=True, channels=self.channels)

                ephys_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.eid,
                                                provenance='Ephys aligned histology track')
                patch_dict = {'json': self.alignments}
                self.one.alyx.rest('trajectories', 'partial_update', id=ephys_traj[0]['id'],
                                   data=patch_dict)

        return files_to_register

    def update_experimenter_evaluation(self, prev_alignments=None, override=False):

        if not np.any(prev_alignments) and not np.any(self.alignments):
            aligned_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.eid,
                                              provenance='Ephys aligned histology track')
            if len(aligned_traj) > 0:
                self.alignments = aligned_traj[0].get('json', {})
            else:
                self.alignments = {}
                return
        else:
            self.alignments = prev_alignments

        outcomes = [align[2].split(':')[0] for key, align in self.alignments.items()
                    if len(align) == 3]
        if len(outcomes) > 0:
            vals = [CRITERIA_BASE[out] for out in outcomes]
            max_qc = np.argmax(vals)
            outcome = outcomes[max_qc]
            self.update(outcome, namespace='experimenter', override=override)
        else:
            self.log.warning(f'No experimenter qc found, qc field of probe insertion {self.eid} '
                             f'will not be updated')
