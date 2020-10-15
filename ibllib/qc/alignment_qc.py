import logging
from ibllib.atlas import AllenAtlas, regions_from_allen_csv
from ibllib.pipes import histology
from ibllib.ephys.neuropixel import SITES_COORDINATES
import numpy as np
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.qc import base
from oneibl.patcher import FTPPatcher

_log = logging.getLogger('ibllib')
CRITERIA = {"PASS": 0.8}


class AlignmentQC(base.QC):
    def __init__(self, probe_id, one=None, brain_atlas=None, override=True, channels=True):
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
        self.override = override

        # Get the brain atlas
        self.brain_atlas = brain_atlas or AllenAtlas(25)
        # Flag for uploading to alyx. For testing purposes
        self.channels = channels

    def load_data(self, prev_alignments=None, xyz_picks=None, depths=None, cluster_chns=None):
        if not np.any(prev_alignments):
            self.alignments = self.one.alyx.rest('trajectories', 'list',
                                                 probe_insertion=self.eid, provenance=
                                                 'Ephys aligned histology track')[0]['json']
        else:
            self.alignments = prev_alignments

        align_keys = [*self.alignments.keys()]
        self.align_keys_sorted = sorted(align_keys, reverse=True)

        if not np.any(xyz_picks):
            self.xyz_picks = np.array(self.one.alyx.rest('insertions', 'read', id=self.eid)
                                      ['json']['xyz_picks'])/1e6
        else:
            self.xyz_picks = xyz_picks

        if not np.any(depths):
            self.depths = SITES_COORDINATES[:, 1]
        else:
            self.depths = depths

        if not np.any(cluster_chns):
            ins = self.one.alyx.rest('insertions', 'read', id=self.eid)
            session_id = ins['session']
            probe_name = ins['name']
            _ = self.one.load(session_id, dataset_types='clusters.channels', download_only=True)
            self.cluster_chns = np.load(self.one.path_from_eid(session_id).
                                        joinpath('alf', probe_name, 'clusters.channels.npy'))
        else:
            self.cluster_chns = cluster_chns

    def compute(self):
        """Compute and store the QC metrics
        Runs the QC on the session and stores a map of the metrics for each datapoint for each
        test, and a map of which datapoints passed for each test
        :return:
        """
        if self.alignments is None:
            self.load_data()
        self.log.info(f"Insertion {self.eid}: Running QC on alignment data...")
        self.sim_matrix = self.compute_similarity_matrix()
        return

    def run(self, update=False, upload_alyx=False, upload_flatiron=False):
        if self.sim_matrix is None:
            self.compute()
        self.outcome, results = self.compute_alignment_status()

        if update:
            self.update_extended_qc(results)
            self.update(self.outcome, 'alignment', override=self.override)

        if results['_alignment_resolved'] == 1 and (upload_alyx or upload_flatiron):
            self.upload_channels(results['_alignment_stored'], upload_alyx, upload_flatiron)

        return self.outcome, results

    def resolve_manual(self, align_key, update=False, upload_alyx=False, upload_flatiron=False):
        if self.sim_matrix is None:
            self.compute()
        assert align_key in self.align_keys_sorted, 'align key not recognised'
        self.outcome, results = self.compute_alignment_status()
        results['_alignment_resolved'] = 1
        results['_alignment_stored'] = align_key
        self.outcome = 'PASS'

        if update:
            self.update_extended_qc(results)
            self.update(self.outcome, 'alignment_user', override=self.override)

        if upload_alyx or upload_flatiron:
            file_paths = self.upload_channels(align_key, upload_alyx, upload_flatiron)

        return file_paths

    def compute_similarity_matrix(self):

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

        # Set diagonals to zero so we don't use those to find max
        self.sim_matrix[self.sim_matrix == 1] = 0
        max_sim = np.max(self.sim_matrix)

        results = {'_alignment_qc': max_sim,
                   '_alignment_number': self.sim_matrix.shape[0]}

        if max_sim > CRITERIA['PASS']:
            location = np.where(self.sim_matrix == max_sim)
            results.update({'_alignment_stored': self.align_keys_sorted[np.min(location)]})
            results.update({'_alignment_resolved': 1})
            outcome = 'PASS'

        else:
            results.update({'_alignment_stored': self.align_keys_sorted[0]})
            results.update({'_alignment_resolved': 0})

            outcome = 'WARNING'

        return outcome, results

    def upload_channels(self, alignment_key, upload_alyx, upload_flatiron, dry=True):

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

            ftp_patcher.create_dataset(path=files_to_register, created_by=self.one._par.ALYX_LOGIN)

        # Need to change channels stored on alyx as well as the stored key is not the same as the
        # latest key
        if upload_alyx:
            if alignment_key != self.align_keys_sorted[0]:
                histology.register_aligned_track(self.eid, channels_mlapdv,
                                                 chn_coords=SITES_COORDINATES, one=self.one,
                                                 overwrite=True, channels=self.channels)

                ephys_traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.eid,
                                                provenance='Ephys aligned histology track')
                patch_dict = {'json': self.alignments}
                self.one.alyx.rest('trajectories', 'partial_update', id=ephys_traj[0]['id'],
                                   data=patch_dict)

        return files_to_register





