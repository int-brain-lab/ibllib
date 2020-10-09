import logging
from ibllib.atlas import AllenAtlas, regions_from_allen_csv
from ibllib.ephys.neuropixel import SITES_COORDINATES
import numpy as np
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.qc import base

_log = logging.getLogger('ibllib')
CRITERIA = {"PASS": 0.8}


class AlignmentQC(base.QC):
    def __init__(self, probe_id, one=None, brain_atlas=None):
        super().__init__(probe_id, one=one, log=_log, endpoint='insertions')

        # Data
        self.alignments = None
        self.xyz_picks = None
        self.depths = None
        self.cluster_chns = None

        # Metrics and passed trials
        self.sim_matrix = None
        self.criteria = CRITERIA

        # Get the brain atlas
        self.brain_atlas = brain_atlas or AllenAtlas(25)

    def load_data(self, prev_alignments=None, xyz_picks=None, depths=None, cluster_chns=None):
        if not self.alignments:
            self.alignments = self.one.alyx.rest('trajectories', 'list',
                                                 probe_insertion=self.eid, provenance=
                                                 'Ephys aligned histology track')[0]['json']
        else:
            self.alignments = prev_alignments

        if not np.any(xyz_picks):
            self.xyz_picks = np.array(self.one.alyx.rest('insertions', 'read', id=self.eid)
                                      ['json']['xyz_picks'])/1e6
        else:
            self.xyz_picks=xyz_picks

        if not np.any(depths):
            self.depths = SITES_COORDINATES[:, 1]
        else:
            self.depths = depths

        if not np.any(cluster_chns):
            ins = self.one.alyx.rest('insertions', 'read', id= self.eid)
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

    def run(self, update=False):
        if self.sim_matrix is None:
            self.compute()
        self.outcome, results = self.compute_alignment_status()
        if update:
            self.update_extended_qc(results)
            self.update(self.outcome, 'alignment')
        return self.outcome, results

    def compute_similarity_matrix(self):

        r = regions_from_allen_csv()
        # Get the keys ordered by date
        align_keys = [*self.alignments.keys()]
        self.align_keys_sorted = sorted(align_keys, reverse=True)

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
            if self.sim_matrix.shape[0] > 2:
                location = np.where(self.sim_matrix == max_sim)

                if not np.any(location == (self.sim_matrix.shape[0] - 1)):
                    # in this case the one if the ones that align are not the latest uploaded
                    # and so we need to reassign the channels that are stored on alyx
                    results.update({'_alignment_stored': self.align_keys_sorted[np.max(location)]})
                    results.update({'_alignment_resolved': 1})
                else:
                    results.update({'_alignment_stored': self.align_keys_sorted[0]})
                    results.update({'_alignment_resolved': 1})

            outcome = 'PASS'

        else:
            results.update({'_alignment_stored': self.align_keys_sorted[0]})
            results.update({'_alignment_resolved': 0})

            outcome = 'WARNING'

        return outcome, results
