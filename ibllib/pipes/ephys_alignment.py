import scipy
import numpy as np
import ibllib.pipes.histology as histology
import ibllib.atlas as atlas

brain_atlas = atlas.AllenAtlas(25)
TIP_SIZE_UM = 200


def _cumulative_distance(xyz):
    return np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))])


class EphysAlignment:

    def __init__(self, xyz_picks, chn_depths=None, track_prev=None,
                 feature_prev=None):

        self.xyz_track, self.track_extent = self.get_insertion_track(xyz_picks)
        self.chn_depths = chn_depths
        if np.any(track_prev):
            self.track_init = track_prev
            self.feature_init = feature_prev
        else:
            self.track_init = np.copy(self.track_extent)
            self.feature_init = np.copy(self.track_extent)

        self.region, self.region_label, self.region_colour, self.region_id\
            = self.get_histology_regions()

    def get_insertion_track(self, xyz_picks):
        """
        Extends probe trajectory from bottom of brain to upper bound of allen atlas
        :param xyz_picks: points defining probe trajectory in 3D space (xyz)
        :type xyz_picks: np.array((n, 3)) - n: no. of unique points
        :return xyz_track: points defining extended trajectory in 3D space (xyz)
        :type xyz_track: np.array((n+2, 3))
        :return track_extent: cumulative distance between two extremes of xyz_track (bottom of
        brain and top of atlas) offset by distance to probe tip
        :type track_extent: np.array((2))
        """
        # Use the first and last quarter of xyz_picks to estimate the trajectory beyond xyz_picks
        n_picks = np.max([4, round(xyz_picks.shape[0] / 4)])
        traj_entry = atlas.Trajectory.fit(xyz_picks[:n_picks, :])
        traj_exit = atlas.Trajectory.fit(xyz_picks[-1 * n_picks:, :])
        # Force the entry to be on the upper z lim of the atlas to account for cases where channels
        # may be located above the surface of the brain
        entry = (traj_entry.eval_z(brain_atlas.bc.zlim))[0, :]
        # The exit is just below the bottom surfacce of the brain
        exit = atlas.Insertion.get_brain_exit(traj_exit, brain_atlas)
        exit[2] = exit[2] - 200 / 1e6

        xyz_track = np.r_[exit[np.newaxis, :], xyz_picks, entry[np.newaxis, :]]
        # Sort so that most ventral coordinate is first
        xyz_track = xyz_track[np.argsort(xyz_track[:, 2]), :]

        # Compute distance to first electrode from bottom coordinate
        tip_distance = _cumulative_distance(xyz_track)[1] + TIP_SIZE_UM / 1e6
        track_length = _cumulative_distance(xyz_track)[-1]
        track_extent = np.array([0, track_length]) - tip_distance

        return xyz_track, track_extent

    def get_track_and_feature(self):
        """
        Return track, feature and xyz_track variables
        """
        return self.feature_init, self.track_init, self.xyz_track

    @staticmethod
    def feature2track(trk, feature, track):
        """
        Estimate new values of trk according to interpolated fit between feature and track space
        :param trk: points in track space to convert feature space
        :type trk: np.array
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :return fcn(trk): interpolated values of trk
        :type fcn(trk): np.array
        """

        fcn = scipy.interpolate.interp1d(feature, track, fill_value="extrapolate")
        return fcn(trk)

    @staticmethod
    def track2feature(ft, feature, track):
        """
        Estimate new values of ft according to interpolated fit between track and feature space
        :param ft: points in feature space to convert track space
        :type ft: np.array
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :return fcn(ft): interpolated values of ft
        :type fcn(ft): np.array
        """
        fcn = scipy.interpolate.interp1d(track, feature, fill_value="extrapolate")
        return fcn(ft)

    @staticmethod
    def feature2track_lin(trk, feature, track):
        """
        Estimate new values of trk according to linear fit between feature and track space, only
        implemented if no. of reference points >= 3
        :param trk: points in track space to convert feature space
        :type trk: np.array
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :return fcn(trk): linear fit values of trk
        :type fcn(trk): np.array
        """
        if feature.size >= 5:
            fcn_lin = np.poly1d(np.polyfit(feature[1:-1], track[1:-1], 1))
            lin_fit = fcn_lin(trk)
        else:
            lin_fit = 0
        return lin_fit

    @staticmethod
    def adjust_extremes_uniform(feature, track):
        """
        Change the value of the first and last reference points (non user chosen points) such
        that coordinates outside user picked regions are left unchanged
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :return track: reference coordinates in track space with first and last value adjusted
        :type track: np.array((n_lines + 2))
        """
        diff = np.diff(feature - track)
        track[0] -= diff[0]
        track[-1] += diff[-1]
        return track

    def adjust_extremes_linear(self, feature, track, extend_feature=1):
        """
        Change the value of the first and last reference points (non user chosen points) such
        that coordinates outside user picked regions have a linear fit applied
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :param extend_feature: amount to extend extreme coordinates before applying linear fit
        :type extend_feature: float
        :return feature: reference coordinates in feature space with first and last value adjusted
        :type feature: np.array((n_lines + 2))
        :return track: reference coordinates in track space with first and last value adjusted
        :type track: np.array((n_lines + 2))
        """
        feature[0] = self.track_extent[0] - extend_feature
        feature[-1] = self.track_extent[-1] + extend_feature
        extend_track = self.feature2track_lin(feature[[0, -1]], feature, track)
        track[0] = extend_track[0]
        track[-1] = extend_track[-1]
        return feature, track

    def scale_histology_regions(self, feature, track):
        """
        Recompute locations of brain region boundaries using interpolated fit based on reference
        lines
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_lines + 2))
        :return region: new coordinates of histology boundaries after applying interpolation
        :type region: np.array((n_bound, 2)) n_bound: no. of histology boundaries
        :return region_label: new coordinates of histology labels positions after applying
                              interpolation
        :type region_label: np.array((n_bound)) of tuples (coordinate - float, label - str)
        """
        region_label = np.copy(self.region_label)
        region = self.track2feature(self.region, feature, track) * 1e6
        region_label[:, 0] = (self.track2feature(np.float64(region_label[:, 0]), feature,
                              track) * 1e6)
        return region, region_label

    def get_histology_regions(self):
        """
        Find all brain regions and their boundaries along the depth of extrapolated xyz_track
        :return region: coordinates bounding each brain region
        :type region: np.array((n_bound, 2)) n_bound: no. of histology boundaries
        :return region_label: label for each brain region and coordinate of where to place label
        :type region_label: np.array((n_bound)) of tuples (coordinate - float, label - str)
        :return region_colour: allen atlas rgb colour for each brain region along track
        :type region_colour: np.array((n_bound, 3))
        :return region_id: allen atlas id for each brain region along track
        :type region_id: np.array((n_bound))
        """
        sampling_trk = np.arange(self.track_extent[0],
                                 self.track_extent[-1] - 10 * 1e-6, 10 * 1e-6)
        xyz_samples = histology.interpolate_along_track(self.xyz_track,
                                                        sampling_trk - sampling_trk[0])
        region_ids = brain_atlas.get_labels(xyz_samples)
        region_info = brain_atlas.regions.get(region_ids)
        boundaries = np.where(np.diff(region_info.id))[0]
        region = np.empty((boundaries.size + 1, 2))
        region_label = np.empty((boundaries.size + 1, 2), dtype=object)
        region_id = np.empty((boundaries.size + 1, 1), dtype=int)
        region_colour = np.empty((boundaries.size + 1, 3), dtype=int)
        for bound in np.arange(boundaries.size + 1):
            if bound == 0:
                _region = np.array([0, boundaries[bound]])
            elif bound == boundaries.size:
                _region = np.array([boundaries[bound - 1], region_info.id.size - 1])
            else:
                _region = np.array([boundaries[bound - 1], boundaries[bound]])
            _region_colour = region_info.rgb[_region[1]]
            _region_label = region_info.acronym[_region[1]]
            _region_id = region_info.id[_region[1]]
            _region = sampling_trk[_region]
            _region_mean = np.mean(_region)
            region[bound, :] = _region
            region_colour[bound, :] = _region_colour
            region_id[bound, :] = _region_id
            region_label[bound, :] = (_region_mean, _region_label)

        return region, region_label, region_colour, region_id

    def get_scale_factor(self, region):
        """
        Find how much each brain region has been scaled following interpolation
        :param region: scaled histology boundaries
        :type region: np.array((n_bound, 2)) n_bound: no. of histology boundaries
        :return scaled_region: regions that have unique scaling applied
        :type scaled_region: np.array((n_scale, 2)) n_scale: no. of uniquely scaled regions
        :return scale_factor: scale factor applied to each scaled region
        :type scale_factor: np.array((n_scale))
        """
        scale = []
        for iR, (reg, reg_orig) in enumerate(zip(region, self.region * 1e6)):
            scale = np.r_[scale, (reg[1] - reg[0]) / (reg_orig[1] - reg_orig[0])]
        boundaries = np.where(np.diff(np.around(scale, 3)))[0]
        if boundaries.size == 0:
            scaled_region = np.array([[region[0][0], region[-1][1]]])
            scale_factor = np.unique(scale)
        else:
            scaled_region = np.empty((boundaries.size + 1, 2))
            scale_factor = []
            for bound in np.arange(boundaries.size + 1):
                if bound == 0:
                    _scaled_region = np.array([region[0][0],
                                              region[boundaries[bound]][1]])
                    _scale_factor = scale[0]
                elif bound == boundaries.size:
                    _scaled_region = np.array([region[boundaries[bound - 1]][1],
                                              region[-1][1]])
                    _scale_factor = scale[-1]
                else:
                    _scaled_region = np.array([region[boundaries[bound - 1]][1],
                                              region[boundaries[bound]][1]])
                    _scale_factor = scale[boundaries[bound]]
                scaled_region[bound, :] = _scaled_region
                scale_factor = np.r_[scale_factor, _scale_factor]
        return scaled_region, scale_factor

    def get_channel_locations(self, feature, track, depths=None):
        """
        Gets 3d coordinates from a depth along the electrophysiology feature. 2 steps
        1) interpolate from the electrophys features depths space to the probe depth space
        2) interpolate from the probe depth space to the true 3D coordinates
        if depths is not provided, defaults to channels local coordinates depths
        """
        if depths is None:
            depths = self.chn_depths / 1e6
        # nb using scipy here so we can change to cubic spline if needed
        channel_depths_track = self.feature2track(depths, feature, track) - self.track_extent[0]
        xyz_channels = histology.interpolate_along_track(self.xyz_track, channel_depths_track)
        return xyz_channels

    def get_brain_locations(self, xyz_channels):
        """
        Finds the brain regions from 3D coordinates of electrode locations
        :param xyz_channels: 3D coordinates of electrodes on probe
        :type xyz_channels: np.array((n_elec, 3)) n_elec: no. of electrodes (384)
        :return brain_regions: brain region object for each electrode
        :type dict
        """
        brain_regions = brain_atlas.regions.get(brain_atlas.get_labels(xyz_channels))
        return brain_regions

    def get_perp_vector(self, feature, track):
        """
        Finds the perpendicular vector along the trajectory at the depth of reference lines
        :param feature: reference coordinates in feature space (ephys plots)
        :type feature: np.array((n_lines + 2)) n_lines: no. of user reference lines
        :param track: reference coordinates in track space (histology track)
        :type track: np.array((n_line+2))
        :return slice_lines: coordinates of perpendicular lines
        :type slice_lines: np.array((n_lines, 2))
        """

        slice_lines = []
        for line in feature[1:-1]:
            depths = np.array([line, line + 10 / 1e6])
            xyz = self.get_channel_locations(feature, track, depths)

            extent = 500e-6
            vector = np.diff(xyz, axis=0)[0]
            point = xyz[0, :]
            vector_perp = np.array([1, 0, -1 * vector[0] / vector[2]])
            xyz_per = np.r_[[point + (-1 * extent * vector_perp)],
                            [point + (extent * vector_perp)]]
            slice_lines.append(xyz_per)

        return slice_lines
