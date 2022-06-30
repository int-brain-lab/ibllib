from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from iblutil.util import Bunch
from iblutil.numerical import ismember

_logger = logging.getLogger('ibllib')
# 'Beryl' is the name given to an atlas containing a subset of the most relevant allen annotations
FILE_MAPPINGS = str(Path(__file__).parent.joinpath('mappings.pqt'))
FILE_REGIONS = str(Path(__file__).parent.joinpath('allen_structure_tree.csv'))


@dataclass
class _BrainRegions:
    id: np.ndarray
    name: object
    acronym: object
    rgb: np.uint8
    level: np.ndarray
    parent: np.ndarray
    order: np.uint16


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
                         level=df_regions.depth.to_numpy().astype(np.uint16),
                         parent=df_regions.parent_structure_id.to_numpy(),
                         order=df_regions.graph_order.to_numpy().astype(np.uint16))
        # mappings are indices not ids: they range from 0 to n regions -1
        mappings = pd.read_parquet(FILE_MAPPINGS)
        self.mappings = {k: mappings[k].to_numpy() for k in mappings}
        self.n_lr = int((len(self.id) - 1) / 2)

    @property
    def rgba(self):
        rgba = np.c_[self.rgb, self.rgb[:, 0] * 0 + 255]
        rgba[0, :] = 0  # set the void to transparent
        return rgba

    def _compute_order(self):
        """
        Compute the order of regions, per region order by left hemisphere and then right hemisphere
        :return:
        """
        orders = np.zeros_like(self.id)
        # Left hemisphere first
        orders[1::2] = np.arange(self.n_lr) + self.n_lr + 1
        # Then right hemisphere
        orders[2::2] = np.arange(self.n_lr) + 1

    def _compute_mappings(self):
        """
        Recomputes the mapping indices for all mappings
        This is left mainly as a reference for adding future mappings as this take a few seconds
        to execute. In production,we use the MAPPING_FILES pqt to avoid recompuing at each \
        instantiation
        """
        beryl = np.load(Path(__file__).parent.joinpath('beryl.npy'))
        cosmos = np.load(Path(__file__).parent.joinpath('cosmos.npy'))
        swanson = np.load(Path(__file__).parent.joinpath('swanson_regions.npy'))
        self.mappings = {
            'Allen': self._mapping_from_regions_list(np.unique(np.abs(self.id)), lateralize=False),
            'Allen-lr': np.arange(self.id.size),
            'Beryl': self._mapping_from_regions_list(beryl, lateralize=False),
            'Beryl-lr': self._mapping_from_regions_list(beryl, lateralize=True),
            'Cosmos': self._mapping_from_regions_list(cosmos, lateralize=False),
            'Cosmos-lr': self._mapping_from_regions_list(cosmos, lateralize=True),
            'Swanson': self._mapping_from_regions_list(swanson, lateralize=False),
            'Swanson-lr': self._mapping_from_regions_list(swanson, lateralize=True),
        }
        pd.DataFrame(self.mappings).to_parquet(FILE_MAPPINGS)

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

    def _navigate_tree(self, ids, direction='down', return_indices=False):
        """
        Private method to navigate the tree and get all related objects either up, down or along the branch.
        By convention the provided id is returned in the list of regions
        :param ids: array or single allen id (int32)
        :param direction: 'up' returns ancestors, 'down' descendants
        :param return indices: Bool (False), if true returns a second argument with indices mapping
        to the current br object
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
        if return_indices:
            return self.get(self.id[indices]), np.where(indices)[0]
        else:
            return self.get(self.id[indices])

    def subtree(self, scalar_id, return_indices=False):
        """
        Given a node, returns the subtree containing the node along with ancestors
        :param return indices: Bool (False), if true returns a second argument with indices mapping
        to the current br object
        :return: Bunch
        """
        if not np.isscalar(scalar_id):
            assert scalar_id.size == 1
        _, idown = self._navigate_tree(scalar_id, direction='down', return_indices=True)
        _, iup = self._navigate_tree(scalar_id, direction='up', return_indices=True)
        indices = np.unique(np.r_[idown, iup])
        if return_indices:
            return self.get(self.id[indices]), np.where(indices)[0]
        else:
            return self.get(self.id[indices])

    def descendants(self, ids, **kwargs):
        """
        Get descendants from one or an array of ids
        :param ids: np.array or scalar representing the region primary key
        :param return_indices: Bool (False) returns the indices in the current br obj
        :return: Bunch
        """
        return self._navigate_tree(ids, direction='down', **kwargs)

    def ancestors(self, ids, **kwargs):
        """
        Get ancestors from one or an array of ids
        :param ids: np.array or scalar representing the region primary key
        :param return_indices: Bool (False) returns the indices in the current br obj
        :return: Bunch
        """
        return self._navigate_tree(ids, direction='up', **kwargs)

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
        # TO DO should root be lateralised?
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

    def remap(self, region_ids, source_map='Allen', target_map='Beryl'):
        """
        Remap atlas regions ids from source map to target map
        :param region_ids: atlas ids to map
        :param source_map: map name which original region_ids are in
        :param target_map: map name onto which to map
        :return:
        """
        _, inds = ismember(region_ids, self.id[self.mappings[source_map]])
        return self.id[self.mappings[target_map][inds]]

    def acronym2acronym(self, acronym, mapping='Allen'):
        """
        Remap acronyms onto mapping

        :param acronym: list or array of acronyms
        :param mapping: target map to remap acronyms
        :return: array of remapped acronyms
        """
        inds = self._find_inds(acronym, self.acronym)
        return self.acronym[self.mappings[mapping]][inds]

    def acronym2id(self, acronym, mapping='Allen', hemisphere=None):
        """
        Convert acronyms to atlas ids and remap

        :param acronym: list or array of acronyms
        :param mapping: target map to remap atlas_ids
        :param hemisphere: which hemisphere to return atlas ids for, options left or right
        :return: array of remapped atlas ids
        """
        inds = self._find_inds(acronym, self.acronym)
        return self.id[self.mappings[mapping]][self._filter_lr(inds, mapping, hemisphere)]

    def acronym2index(self, acronym, mapping='Allen', hemisphere=None):
        """
        Convert acronym to index and remap
        :param acronym:
        :param mapping:
        :param hemisphere:
        :return: array of remapped acronyms and list of indexes for each acronnym
        """
        acronym = self.acronym2acronym(acronym, mapping=mapping)
        index = list()
        for id in acronym:
            inds = np.where(self.acronym[self.mappings[mapping]] == id)[0]
            index.append(self._filter_lr_index(inds, hemisphere))

        return acronym, index

    def id2acronym(self, atlas_id, mapping='Allen'):
        """
        Convert atlas id to acronym and remap

        :param acronym: list or array of atlas ids
        :param mapping: target map to remap acronyms
        :return: array of remapped acronyms
        """
        inds = self._find_inds(atlas_id, self.id)
        return self.acronym[self.mappings[mapping]][inds]

    def id2id(self, atlas_id, mapping='Allen'):
        """
        Remap atlas id onto mapping

        :param acronym: list or array of atlas ids
        :param mapping: target map to remap acronyms
        :return: array of remapped atlas ids
        """

        inds = self._find_inds(atlas_id, self.id)
        return self.id[self.mappings[mapping]][inds]

    def id2index(self, atlas_id, mapping='Allen'):
        """
        Convert atlas id to index and remap

        :param atlas_id: list or array of atlas ids
        :param mapping: mapping to use
        :return: dict of indices for each atlas_id
        """

        atlas_id = self.id2id(atlas_id, mapping=mapping)
        index = list()
        for id in atlas_id:
            inds = np.where(self.id[self.mappings[mapping]] == id)[0]
            index.append(inds)

        return atlas_id, index

    def index2acronym(self, index, mapping='Allen'):
        """
        Convert index to acronym and remap

        :param index:
        :param mapping:
        :return:
        """
        inds = self.acronym[self.mappings[mapping]][index]
        return inds

    def index2id(self, index, mapping='Allen'):
        """
        Convert index to atlas id and remap

        :param index:
        :param mapping:
        :return:
        """
        inds = self.id[self.mappings[mapping]][index]
        return inds

    def _filter_lr(self, values, mapping, hemisphere):
        """
        Filter values by those on left or right hemisphere
        :param values: array of index values
        :param mapping: mapping to use
        :param hemisphere: hemisphere
        :return:
        """
        if 'lr' in mapping:
            if hemisphere == 'left':
                return values + self.n_lr
            elif hemisphere == 'right':
                return values
            else:
                return np.c_[values + self.n_lr, values]
        else:
            return values

    def _filter_lr_index(self, values, hemisphere):
        """
        Filter index values  by those on left or right hemisphere

        :param values: array of index values
        :param mapping: mapping to use
        :param hemisphere: hemisphere
        :return:
        """
        if hemisphere == 'left':
            return values[values > self.n_lr]
        elif hemisphere == 'right':
            return values[values <= self.n_lr]
        else:
            return values

    def _find_inds(self, values, all_values):
        if not isinstance(values, list) and not isinstance(values, np.ndarray):
            values = np.array([values])
        _, inds = ismember(np.array(values), all_values)

        return inds

    def parse_acronyms_argument(self, acronyms, mode='raise'):
        """
        Parse an input acronym arguments: returns a numpy array of allen regions ids
        regardless of the input: list of acronyms, np.array of acronyms strings or np aray of allen ids
        To be used into functions to provide flexible input type
        :param acronyms: List, np.array of acronym strings or np.array of allen ids
        :return: np.array of int ids
        """
        # first get the allen region ids regardless of the input type
        acronyms = np.array(acronyms)
        # if the user provides acronyms they're not signed by definition
        if not np.issubdtype(acronyms.dtype, np.number):
            user_aids = self.acronym2id(acronyms)
            if mode == 'raise':
                assert user_aids.size == acronyms.size, "All acronyms should exist in Allen ontology"
        else:
            user_aids = acronyms
        return user_aids


def regions_from_allen_csv():
    """
    Reads csv file containing the ALlen Ontology and instantiates a BrainRegions object
    :return: BrainRegions object
    """
    _logger.warning("ibllib.atlas.regions.regions_from_allen_csv() is deprecated. "
                    "Use BrainRegions() instead")
    return BrainRegions()
