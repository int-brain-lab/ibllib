"""Brain region mappings.

Four mappings are currently available within the IBL, these are:

* Allen Atlas - total of 1328 annotation regions provided by Allen Atlas.
* Beryl Atlas - total of 308 annotation regions determined by Nick Steinmetz for the brain wide map, mainly at the level of
  major cortical areas, nuclei/ganglia. Thus annotations relating to layers and nuclear subregions are absent.
* Cosmos Atlas - total of 10 annotation regions determined by Nick Steinmetz for coarse analysis.  Annotations include the major
  divisions of the brain only.
* Swanson Atlas - total of 319 annotation regions provided by the Swanson atlas (FIXME which one?).

Terminology
-----------
* **Name** - The full anatomical name of a brain region.
* **Acronymn** - A shortened version of a brain region name.
* **Index** - The index of the of the brain region within the ordered list of brain regions.
* **ID** - A unique numerical identifier of a brain region.  These are typically integers that
  therefore take up less space than storing the region names or acronyms.
* **Mapping** - A function that maps one ordered list of brain region IDs to another, allowing one
  to control annotation granularity and brain region hierarchy, or to translate brain region names
  from one atlas to another. The default mapping is identity. See
  [atlas package documentation](./ibllib.atlas.html#mappings) for other mappings.
* **Order** - Each structure is assigned a consistent position within the flattened graph. This
  value is known as the annotation index, i.e. the annotation volume contains the brain region
  order at each point in the image.

FIXME Document the two structure trees. Which Website did they come from, and which publication/edition?
"""
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from iblutil.util import Bunch
from iblutil.numerical import ismember

_logger = logging.getLogger(__name__)
FILE_MAPPINGS = str(Path(__file__).parent.joinpath('mappings.pqt'))
ALLEN_FILE_REGIONS = str(Path(__file__).parent.joinpath('allen_structure_tree.csv'))
FRANKLIN_FILE_REGIONS = str(Path(__file__).parent.joinpath('franklin_paxinos_structure_tree.csv'))


@dataclass
class _BrainRegions:
    """A struct of brain regions, their names, IDs, relationships and associated plot colours."""

    """numpy.array: An integer array of unique brain region IDs."""
    id: np.ndarray
    """numpy.array: A str array of verbose brain region names."""
    name: object
    """numpy.array: A str array of brain region acronyms."""
    acronym: object
    """numpy.array: A, (n, 3) uint8 array of brain region RGB colour values."""
    rgb: np.uint8
    """numpy.array: An unsigned integer array indicating the number of degrees removed from root."""
    level: np.ndarray
    """numpy.array: An integer array of parent brain region IDs."""
    parent: np.ndarray
    """numpy.array: The position within the flattened graph."""
    order: np.uint16

    def __post_init__(self):
        self._compute_mappings()

    def _compute_mappings(self):
        """Compute default mapping for the structure tree.

        Default mapping is identity. This method is intended to be overloaded by subclasses.
        """
        self.default_mapping = None
        self.mappings = dict(default_mapping=self.order)
        # the number of lateralized regions (typically half the number of regions in a lateralized structure tree)
        self.n_lr = 0

    def to_df(self):
        """
        Return dataclass as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            The object as a pandas DataFrame with attributes as columns.
        """
        attrs = ['id', 'name', 'acronym', 'hexcolor', 'level', 'parent', 'order']
        d = dict(zip(attrs, list(map(self.__getattribute__, attrs))))
        return pd.DataFrame(d)

    @property
    def rgba(self):
        """numpy.array: An (n, 4) uint8 array of RGBA values for all n brain regions."""
        rgba = np.c_[self.rgb, self.rgb[:, 0] * 0 + 255]
        rgba[0, :] = 0  # set the void to transparent
        return rgba

    @property
    def hexcolor(self):
        """numpy.array of str: The RGB colour values as hexadecimal triplet strings."""
        return np.apply_along_axis(lambda x: "#{0:02x}{1:02x}{2:02x}".format(*x.astype(int)), 1, self.rgb)

    def get(self, ids) -> Bunch:
        """
        Return a map of id, name, acronym, etc. for the provided IDs.

        Parameters
        ----------
        ids : int, tuple of ints, numpy.array
            One or more brain region IDs to get information for.

        Returns
        -------
        iblutil.util.Bunch[str, numpy.array]
            A dict-like object containing the keys {'id', 'name', 'acronym', 'rgb', 'level',
            'parent', 'order'} with arrays the length of `ids`.
        """
        uid, uind = np.unique(ids, return_inverse=True)
        a, iself, _ = np.intersect1d(self.id, uid, assume_unique=False, return_indices=True)
        b = Bunch()
        for k in self.__dataclass_fields__.keys():
            b[k] = self.__getattribute__(k)[iself[uind]]
        return b

    def _navigate_tree(self, ids, direction='down', return_indices=False):
        """
        Navigate the tree and get all related objects either up, down or along the branch.

        By convention the provided id is returned in the list of regions.

        Parameters
        ----------
        ids : int, array_like
            One or more brain region IDs (int32).
        direction : {'up', 'down'}
            Whether to return ancestors ('up') or descendants ('down').
        return_indices : bool, default=False
            If true returns a second argument with indices mapping to the current brain region
            object.

        Returns
        -------
        iblutil.util.Bunch[str, numpy.array]
            A dict-like object containing the keys {'id', 'name', 'acronym', 'rgb', 'level',
            'parent', 'order'} with arrays the length of `ids`.
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
        Given a node, returns the subtree containing the node along with ancestors.

        Parameters
        ----------
        scalar_id : int
            A brain region ID.
        return_indices : bool, default=False
            If true returns a second argument with indices mapping to the current brain region
            object.

        Returns
        -------
        iblutil.util.Bunch[str, numpy.array]
            A dict-like object containing the keys {'id', 'name', 'acronym', 'rgb', 'level',
            'parent', 'order'} with arrays the length of one.
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
        Get descendants from one or more IDs.

        Parameters
        ----------
        ids : int, array_like
            One or more brain region IDs.
        return_indices : bool, default=False
            If true returns a second argument with indices mapping to the current brain region
            object.

        Returns
        -------
        iblutil.util.Bunch[str, numpy.array]
            A dict-like object containing the keys {'id', 'name', 'acronym', 'rgb', 'level',
            'parent', 'order'} with arrays the length of `ids`.
        """
        return self._navigate_tree(ids, direction='down', **kwargs)

    def ancestors(self, ids, **kwargs):
        """
        Get ancestors from one or more IDs.

        Parameters
        ----------
        ids : int, array_like
            One or more brain region IDs.
        return_indices : bool, default=False
            If true returns a second argument with indices mapping to the current brain region
            object.

        Returns
        -------
        iblutil.util.Bunch[str, numpy.array]
            A dict-like object containing the keys {'id', 'name', 'acronym', 'rgb', 'level',
            'parent', 'order'} with arrays the length of `ids`.
        """
        return self._navigate_tree(ids, direction='up', **kwargs)

    def leaves(self):
        """
        Get all regions that do not have children.

        Returns
        -------
        iblutil.util.Bunch[str, numpy.array]
            A dict-like object containing the keys {'id', 'name', 'acronym', 'rgb', 'level',
            'parent', 'order'} with arrays of matching length.
        """
        leaves = np.setxor1d(self.id, self.parent)
        return self.get(np.int64(leaves[~np.isnan(leaves)]))

    def _mapping_from_regions_list(self, new_map, lateralize=False):
        """
        From a vector of region IDs, creates a structure tree index mapping.

        For example, given a subset of brain region IDs, this returns an array the length of the
        total number of brain regions, where each element is the structure tree index for that
        region.  The IDs in `new_map` and their descendants are given that ID's index and any
        missing IDs are given the root index.


        Parameters
        ----------
        new_map : array_like of int
            An array of atlas brain region IDs.
        lateralize : bool
            If true, lateralized indices are assigned to all IDs. If false, IDs are assigned to q
            non-lateralized index regardless of their sign.

        Returns
        -------
        numpy.array
            A vector of brain region indices representing the structure tree order corresponding to
            each input ID and its descendants.
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
        # TODO should root be lateralised?
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

    def acronym2acronym(self, acronym, mapping=None):
        """
        Remap acronyms onto mapping

        :param acronym: list or array of acronyms
        :param mapping: target map to remap acronyms
        :return: array of remapped acronyms
        """
        mapping = mapping or self.default_mapping
        inds = self._find_inds(acronym, self.acronym)
        return self.acronym[self.mappings[mapping]][inds]

    def acronym2id(self, acronym, mapping=None, hemisphere=None):
        """
        Convert acronyms to atlas ids and remap

        :param acronym: list or array of acronyms
        :param mapping: target map to remap atlas_ids
        :param hemisphere: which hemisphere to return atlas ids for, options left or right
        :return: array of remapped atlas ids
        """
        mapping = mapping or self.default_mapping
        inds = self._find_inds(acronym, self.acronym)
        return self.id[self.mappings[mapping]][self._filter_lr(inds, mapping, hemisphere)]

    def acronym2index(self, acronym, mapping=None, hemisphere=None):
        """
        Convert acronym to index and remap
        :param acronym:
        :param mapping:
        :param hemisphere:
        :return: array of remapped acronyms and list of indexes for each acronnym
        """
        mapping = mapping or self.default_mapping
        acronym = self.acronym2acronym(acronym, mapping=mapping)
        index = list()
        for id in acronym:
            inds = np.where(self.acronym[self.mappings[mapping]] == id)[0]
            index.append(self._filter_lr_index(inds, hemisphere))

        return acronym, index

    def id2acronym(self, atlas_id, mapping=None):
        """
        Convert atlas id to acronym and remap

        :param atlas_id: list or array of atlas ids
        :param mapping: target map to remap acronyms
        :return: array of remapped acronyms
        """
        mapping = mapping or self.default_mapping
        inds = self._find_inds(atlas_id, self.id)
        return self.acronym[self.mappings[mapping]][inds]

    def id2id(self, atlas_id, mapping='Allen'):
        """
        Remap atlas id onto mapping

        :param atlas_id: list or array of atlas ids
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

    def index2acronym(self, index, mapping=None):
        """
        Convert index to acronym and remap

        :param index:
        :param mapping:
        :return:
        """
        mapping = mapping or self.default_mapping
        inds = self.acronym[self.mappings[mapping]][index]
        return inds

    def index2id(self, index, mapping=None):
        """
        Convert index to atlas id and remap

        :param index:
        :param mapping:
        :return:
        """
        mapping = mapping or self.default_mapping
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
        """Parse input acronyms.

        Returns a numpy array of region IDs regardless of the input: list of acronyms, array of
        acronym strings or region IDs. To be used by functions to provide flexible input type.

        Parameters
        ----------
        acronyms : array_like
            An array of region acronyms to convert to IDs. An array of region IDs may also be
            provided, in which case they are simply returned.
        mode : str, optional
            If 'raise', asserts that all acronyms exist in the structure tree.

        Returns
        -------
        numpy.array of int
            An array of brain regions corresponding to `acronyms`.
        """
        # first get the allen region ids regardless of the input type
        acronyms = np.array(acronyms)
        # if the user provides acronyms they're not signed by definition
        if not np.issubdtype(acronyms.dtype, np.number):
            user_aids = self.acronym2id(acronyms)
            if mode == 'raise':
                assert user_aids.size == acronyms.size, 'all acronyms must exist in the ontology'
        else:
            user_aids = acronyms
        return user_aids


class FranklinPaxinosRegions(_BrainRegions):
    """Mouse Brain in Stereotaxic Coordinates (MBSC).

    Paxinos G, and Franklin KBJ (2012). The Mouse Brain in Stereotaxic Coordinates, 4th edition (Elsevier Academic Press).
    """
    def __init__(self):
        df_regions = pd.read_csv(FRANKLIN_FILE_REGIONS)
        # get rid of nan values, there are rows that are in Allen but are not in the Franklin Paxinos atlas
        df_regions = df_regions[~df_regions['Structural ID'].isna()]
        # add in root
        root = [{'Structural ID': int(997), 'Franklin-Paxinos Full name': 'root', 'Franklin-Paxinos abbreviation': 'root',
                 'structure Order': 50, 'red': 255, 'green': 255, 'blue': 255, 'Allen Full name': 'root',
                 'Allen abbreviation': 'root'}]
        df_regions = pd.concat([pd.DataFrame(root), df_regions], ignore_index=True)

        allen_regions = pd.read_csv(ALLEN_FILE_REGIONS)

        # Find the level of acronyms that are the same as Allen
        a, b = ismember(df_regions['Allen abbreviation'].values, allen_regions['acronym'].values)
        level = allen_regions['depth'].values[b]
        df_regions['level'] = np.full(len(df_regions), np.nan)
        df_regions['allen level'] = np.full(len(df_regions), np.nan)
        df_regions.loc[a, 'level'] = level
        df_regions.loc[a, 'allen level'] = level

        nan_idx = np.where(df_regions['Allen abbreviation'].isna())[0]
        df_regions.loc[nan_idx, 'Allen abbreviation'] = df_regions['Franklin-Paxinos abbreviation'].values[nan_idx]
        df_regions.loc[nan_idx, 'Allen Full name'] = df_regions['Franklin-Paxinos Full name'].values[nan_idx]

        # Now fill in the nan values with one level up from their parents we need to this multiple times
        while np.sum(np.isnan(df_regions['level'].values)) > 0:
            nan_loc = np.isnan(df_regions['level'].values)
            parent_level = df_regions['Parent ID'][nan_loc].values
            a, b = ismember(parent_level, df_regions['Structural ID'].values)
            assert len(a) == len(b) == np.sum(nan_loc)
            level = df_regions['level'].values[b] + 1
            df_regions.loc[nan_loc, 'level'] = level

        # lateralize
        df_regions_left = df_regions.iloc[np.array(df_regions['Structural ID'] > 0), :].copy()
        df_regions_left['Structural ID'] = - df_regions_left['Structural ID']
        df_regions_left['Parent ID'] = - df_regions_left['Parent ID']
        df_regions_left['Allen Full name'] = \
            df_regions_left['Allen Full name'].apply(lambda x: x + ' (left)')
        df_regions = pd.concat((df_regions, df_regions_left), axis=0)

        # insert void
        void = [{'Structural ID': int(0), 'Franklin-Paxinos Full Name': 'void', 'Franklin-Paxinos abbreviation': 'void',
                'Parent ID': int(0), 'structure Order': 0, 'red': 0, 'green': 0, 'blue': 0, 'Allen Full name': 'void',
                 'Allen abbreviation': 'void'}]
        df_regions = pd.concat([pd.DataFrame(void), df_regions], ignore_index=True)

        # converts colors to RGB uint8 array
        c = np.c_[df_regions['red'], df_regions['green'], df_regions['blue']].astype(np.uint32)

        super().__init__(id=df_regions['Structural ID'].to_numpy().astype(np.int64),
                         name=df_regions['Allen Full name'].to_numpy(),
                         acronym=df_regions['Allen abbreviation'].to_numpy(),
                         rgb=c,
                         level=df_regions['level'].to_numpy().astype(np.uint16),
                         parent=df_regions['Parent ID'].to_numpy(),
                         order=df_regions['structure Order'].to_numpy().astype(np.uint16))

    def _compute_mappings(self):
        """
        Compute lateralized and non-lateralized mappings.

        This method is called by __post_init__.
        """
        self.mappings = {
            'FranklinPaxinos': self._mapping_from_regions_list(np.unique(np.abs(self.id)), lateralize=False),
            'FranklinPaxinos-lr': np.arange(self.id.size),
        }
        self.default_mapping = 'FranklinPaxinos'
        self.n_lr = int((len(self.id) - 1) / 2)  # the number of lateralized regions


class BrainRegions(_BrainRegions):
    """
    A struct of Allen brain regions, their names, IDs, relationships and associated plot colours.

    ibllib.atlas.regions.BrainRegions(brainmap='Allen')

    Notes
    -----
    The Allen atlas IDs are kept intact but lateralized as follows: labels are duplicated
    and IDs multiplied by -1, with the understanding that left hemisphere regions have negative
    IDs.
    """
    def __init__(self):
        df_regions = pd.read_csv(ALLEN_FILE_REGIONS)
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
        # For void assign the depth and level to avoid warnings of nan being converted to int
        df_regions.loc[0, 'depth'] = 0
        df_regions.loc[0, 'graph_order'] = 0
        # creates the BrainRegion instance
        super().__init__(id=df_regions.id.to_numpy(),
                         name=df_regions.name.to_numpy(),
                         acronym=df_regions.acronym.to_numpy(),
                         rgb=c,
                         level=df_regions.depth.to_numpy().astype(np.uint16),
                         parent=df_regions.parent_structure_id.to_numpy(),
                         order=df_regions.graph_order.to_numpy().astype(np.uint16))

    def _compute_mappings(self):
        """
        Recomputes the mapping indices for all mappings.

        Attempts to load mappings from the FILE_MAPPINGS file, otherwise generates from arrays of
        brain IDs. In production, we use the MAPPING_FILES pqt to avoid recomputing at each
        instantiation as this take a few seconds to execute.

        Currently there are 8 available mappings (Allen, Beryl, Cosmos, and Swanson), lateralized
        (with suffix -lr) and non-lateralized. Each row contains the correspondence to the Allen
        CCF structure tree order (i.e. index) for each mapping.

        This method is called by __post_init__.
        """
        # mappings are indices not ids: they range from 0 to n regions -1
        if Path(FILE_MAPPINGS).exists():
            mappings = pd.read_parquet(FILE_MAPPINGS)
            self.mappings = {k: mappings[k].to_numpy() for k in mappings}
        else:
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
        self.default_mapping = 'Allen'
        self.n_lr = int((len(self.id) - 1) / 2)  # the number of lateralized regions

    def compute_hierarchy(self):
        """
        Creates a self.hierarchy attribute that is an n_levels by n_region array
        of indices. This is useful to perform fast vectorized computations of
        ancestors and descendants.
        :return:
        """
        if hasattr(self, 'hierarchy'):
            return
        n_levels = np.max(self.level)
        n_regions = self.id.size
        # creates the parent index. Void and root are omitted from intersection
        # as they figure as NaN
        pmask, i_p = ismember(self.parent, self.id)
        self.iparent = np.arange(n_regions)
        self.iparent[pmask] = i_p
        # the last level of the hierarchy is the actual mapping, then going up level per level
        # we assign the parend index
        self.hierarchy = np.tile(np.arange(n_regions), (n_levels, 1))
        _mask = np.zeros(n_regions, bool)
        for lev in np.flipud(np.arange(n_levels)):
            if lev < (n_levels - 1):
                self.hierarchy[lev, _mask] = self.iparent[self.hierarchy[lev + 1, _mask]]
            sel = self.level == (lev + 1)
            self.hierarchy[lev, sel] = np.where(sel)[0]
            _mask[sel] = True

    def propagate_down(self, acronyms, values):
        """
        This function remaps a set of user specified acronyms and values to the
        swanson map, by filling down the child nodes when higher up values are
        provided.
        :param acronyms: list or array of allen ids or acronyms
        :param values: list or array of associated values
        :return:
        # FIXME Why only the swanson map? Also, how is this actually related to the Swanson map?
        """
        user_aids = self.parse_acronyms_argument(acronyms)
        _, user_indices = ismember(user_aids, self.id)
        self.compute_hierarchy()
        ia, ib = ismember(self.hierarchy, user_indices)
        v = np.zeros_like(ia, dtype=np.float64) * np.NaN
        v[ia] = values[ib]
        all_values = np.nanmedian(v, axis=0)
        indices = np.where(np.any(ia, axis=0))[0]
        all_values = all_values[indices]
        return indices, all_values

    def remap(self, region_ids, source_map='Allen', target_map='Beryl'):
        """
        Remap atlas regions IDs from source map to target map.

        Any NaNs in `region_ids` remain as NaN in the output array.

        Parameters
        ----------
        region_ids : array_like of int
            The region IDs to remap.
        source_map : str
            The source map name, in `self.mappings`.
        target_map : str
            The target map name, in `self.mappings`.

        Returns
        -------
        numpy.array of int
            The input IDs mapped to `target_map`.
        """
        isnan = np.isnan(region_ids)
        if np.sum(isnan) > 0:
            # In case the user provides nans
            nan_loc = np.where(isnan)[0]
            _, inds = ismember(region_ids[~isnan], self.id[self.mappings[source_map]])
            mapped_ids = self.id[self.mappings[target_map][inds]].astype(float)
            mapped_ids = np.insert(mapped_ids, nan_loc, np.full(nan_loc.shape, np.nan))
        else:
            _, inds = ismember(region_ids, self.id[self.mappings[source_map]])
            mapped_ids = self.id[self.mappings[target_map][inds]]

        return mapped_ids


def regions_from_allen_csv():
    """
    (DEPRECATED) Reads csv file containing the ALlen Ontology and instantiates a BrainRegions object.

    NB: Instantiate BrainRegions directly instead.

    :return: BrainRegions object
    """
    _logger.warning("ibllib.atlas.regions.regions_from_allen_csv() is deprecated. "
                    "Use BrainRegions() instead")
    return BrainRegions()
