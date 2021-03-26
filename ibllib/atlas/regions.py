from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from brainbox.core import Bunch
from brainbox.numerical import ismember

_logger = logging.getLogger('ibllib')
BERYL = np.array([184, 985, 993, 353, 329, 480149202, 337, 345, 369, 361, 182305689, 378, 1057, 677, 1011, 480149230, 1002, 1027, 1018, 402, 394, 409, 385, 425, 533, 312782574, 312782628, 39, 48, 972, 44, 723, 731, 738, 746, 104, 111, 119, 894, 879, 886, 312782546, 417, 541, 922, 895, 507, 151, 159, 597, 605, 814, 961, 619, 639, 647, 788, 566, 382, 423, 463, 726, 982, 19, 918, 926, 843, 1037, 1084, 502, 484682470, 589508447, 484682508, 583, 952, 966, 131, 295, 319, 780, 672, 56, 998, 754, 250, 258, 266, 310, 333, 23, 292, 536, 1105, 403, 1022, 1031, 342, 298, 564, 596, 581, 351, 629, 685, 718, 725, 733, 741, 563807435, 406, 609, 1044, 475, 170, 218, 1020, 1029, 325, 560581551, 255, 127, 64, 1120, 1113, 155, 59, 362, 366, 1077, 149, 15, 181, 560581559, 189, 599, 907, 575, 930, 560581563, 262, 1014, 27, 563807439, 178, 321, 483, 186, 1097, 390, 38, 30, 118, 223, 72, 263, 272, 830, 452, 523, 1109, 126, 133, 347, 286, 338, 576073699, 689, 88, 210, 491, 525, 557, 515, 980, 1004, 63, 693, 946, 194, 226, 364, 576073704, 173, 470, 614, 797, 302, 4, 580, 271, 874, 381, 749, 607344830, 246, 128, 294, 795, 50, 67, 587, 215, 531, 628, 634, 706, 1061, 549009203, 616, 214, 35, 549009211, 975, 115, 606826663, 757, 231, 66, 75, 58, 374, 1052, 12, 100, 197, 591, 872, 612, 7, 867, 398, 280, 880, 599626927, 898, 931, 1093, 318, 534, 574, 621, 549009215, 549009219, 549009223, 549009227, 679, 147, 162, 604, 146, 238, 350, 358, 207, 96, 101, 711, 1039, 903, 642, 651, 429, 437, 445, 589508451, 653, 661, 135, 839, 1048, 372, 83, 136, 106, 203, 235, 307, 395, 852, 859, 938, 177, 169, 995, 1069, 209, 202, 225, 217, 765, 773, 781, 206, 230, 222, 912, 976, 984, 1091, 936, 944, 951, 957, 968, 1007, 1056, 1064, 1025, 1033, 1041, 1049, 989, 91, 846, 589508455,])  # noqa
FILE_REGIONS = str(Path(__file__).parent.joinpath('allen_structure_tree.csv'))


@dataclass
class _BrainRegions:
    id: np.ndarray
    name: object
    acronym: object
    rgb: np.uint8
    level: np.ndarray
    parent: np.ndarray


class BrainRegions(_BrainRegions):
    """
    ibllib.atlas.regions.BrainRegions(brainmap='Allen')
    The Allen atlas ids are kept intact but lateralized as follows: labels are duplicated
     and ids multiplied by -1, with the understanding that left hemisphere regions have negative
     ids.
    """
    def __init__(self):
        df_regions = pd.read_csv(FILE_REGIONS)
        # lateralize
        df_regions_left = df_regions.iloc[np.array(df_regions.id > 0), :].copy()
        df_regions_left['id'] = - df_regions_left['id']
        df_regions_left['parent_structure_id'] = - df_regions_left['parent_structure_id']
        df_regions_left['name'] = df_regions_left['name'].apply(lambda x: x + ' (left)')
        df_regions = pd.concat((df_regions, df_regions_left), axis=0)
        # converts colors to RGB uint8 array
        c = np.uint32(df_regions.color_hex_triplet.map(
            lambda x: int(x, 16) if isinstance(x, str) else 256 ** 3 - 1))
        c = np.flip(np.reshape(c.view(np.uint8), (df_regions.id.size, 4))[:, :3], 1)
        c[0, :] = 0  # set the void region to black
        # creates the BrainRegion instance
        super().__init__(id=df_regions.id.to_numpy(),
                         name=df_regions.name.to_numpy(),
                         acronym=df_regions.acronym.to_numpy(),
                         rgb=c,
                         level=df_regions.depth.to_numpy(),
                         parent=df_regions.parent_structure_id.to_numpy())
        # mappings are indices not ids: they range from 0 to n regions -1
        self.mappings = {
            'Allen': self._mapping_from_regions_list(np.unique(np.abs(self.id)), lateralize=False),
            'Allen-lr': np.arange(self.id.size),
            'Beryl': self._mapping_from_regions_list(BERYL, lateralize=False),
            'Beryl-lr': self._mapping_from_regions_list(BERYL, lateralize=True),
        }

    def get(self, ids) -> Bunch:
        """
        Get a bunch of the name/id
        """
        uid, uind = np.unique(ids, return_inverse=True)
        a, iself, _ = np.intersect1d(self.id, uid, assume_unique=False, return_indices=True)
        b = Bunch()
        for k in self.__dataclass_fields__.keys():
            b[k] = self.__getattribute__(k)[iself[uind]]
        return b

    def _navigate_tree(self, ids, direction='down'):
        """
        Private method to navigate the tree and get all related objects either up or down
        :param ids:
        :param direction:
        :return: Bunch
        """
        indices = ismember(self.id, ids)[0]
        count = np.sum(indices)
        while True:
            if direction == 'down':
                indices |= ismember(self.parent, self.id[indices])[0]
            elif direction == 'up':
                indices |= ismember(self.id, self.parent[indices])[0]
            else:
                raise ValueError("direction should be either 'up' or 'down'")
            if count == np.sum(indices):  # last iteration didn't find any match
                break
            else:
                count = np.sum(indices)
        return self.get(self.id[indices])

    def descendants(self, ids):
        """
        Get descendants from one or an array of ids
        :param ids: np.array or scalar representing the region primary key
        :return: Bunch
        """
        return self._navigate_tree(ids, direction='down')

    def ancestors(self, ids):
        """
        Get ancestors from one or an array of ids
        :param ids: np.array or scalar representing the region primary key
        :return: Bunch
        """
        return self._navigate_tree(ids, direction='up')

    def leaves(self):
        """
        Get all regions that do not have children
        :return:
        """
        leaves = np.setxor1d(self.id, self.parent)
        return self.get(np.int64(leaves[~np.isnan(leaves)]))

    def _mapping_from_regions_list(self, new_map, lateralize=False):
        """
        From a vector of regions id, creates a mapping such as
        newids = self.mapping
        :param new_map: np.array: vector of regions id
        """
        I_ROOT = 1
        I_VOID = 0
        # to lateralize we make sure all regions are represented in + and -
        new_map = np.unique(np.r_[-new_map, new_map])
        assert np.all(np.isin(new_map, self.id)), \
            "All mapping ids should be represented in the Allen ids"
        # with the lateralization, self.id may have duplicate values so ismember is necessary
        iid, inm = ismember(self.id, new_map)
        iid = np.where(iid)[0]
        mapind = np.zeros_like(self.id) + I_ROOT  # non assigned regions are root
        mapind[iid] = iid  # regions present in the list have the same index
        # Starting by the higher up levels in the hierarchy, assign all descendants to the mapping
        for i in np.argsort(self.level[iid]):
            descendants = self.descendants(self.id[iid[i]]).id
            _, idesc, _ = np.intersect1d(self.id, descendants, return_indices=True)
            mapind[idesc] = iid[i]
        mapind[0] = I_VOID  # void stays void
        # to delateralize the regions, assign the positive index to all mapind elements
        if lateralize is False:
            _, iregion = ismember(np.abs(self.id), self.id)
            mapind = mapind[iregion]
        return mapind


def regions_from_allen_csv():
    """
    Reads csv file containing the ALlen Ontology and instantiates a BrainRegions object
    :return: BrainRegions object
    """
    _logger.warning("ibllib.atlas.regions.regions_from_allen_csv() is deprecated. "
                    "Use BrainRegions() instead")
    return BrainRegions()
