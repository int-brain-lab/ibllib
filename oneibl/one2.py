"""
TODO Dataset based queries
TODO Collection level query with wildcards
TODO Name level query with wildcards
TODO Load objects easily (trials, wheel, spikes, clusters)
TODO Dictionary of bunches/dictionary by default - Load arrays
TODO Cache the results - need to agree on the internal representation
TODO Default loading from disk for load, add a flag
TODO Support both eid/string representation for session
TODO Singleton Alyx instance
TODO Module-level documentation
TODO Docstrings for proposal
"""
import abc
import fnmatch
import re
from collections import defaultdict
from typing import Any, Sequence, Union, Optional, Mapping, List
from uuid import UUID
import concurrent.futures
import logging
import os
from pathlib import Path
from functools import wraps

import requests
import tqdm
import pandas as pd
import numpy as np

import oneibl.params
import oneibl.webclient as wc
from oneibl.one import OneAlyx
# from alf.io import (AlfBunch, get_session_path, is_uuid_string, is_session_path,
#                     load_file_content, remove_uuid_file, load_object)
import alf.io as alfio
from alf.files import is_valid, alf_parts
from ibllib.misc.exp_ref import ref2eid, is_exp_ref

from ibllib.io import hashfile
from oneibl.dataclass import SessionDataInfo
from ibllib.exceptions import ALFMultipleObjectsFound, ALFObjectNotFound
from brainbox.io import parquet
from brainbox.core import ismember, ismember2d

_logger = logging.getLogger('ibllib')

NTHREADS = 4  # number of download threads
Listable = lambda t: Union[t, Sequence[t]]


def parse_id(method):
    """
    Ensures the input experiment identifier is an experiment UUID string
    :param method: An ONE method whose second arg is an experiment id
    :return: A wrapper function that parses the id to the expected string
    """
    @wraps(method)
    def wrapper(self, id, *args, **kwargs):
        id = self.to_eid(id)
        return method(self, id, *args, **kwargs)
    return wrapper


class ONE2(OneAlyx):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_id(self, id: Union[str, Path, UUID], to: str = 'eid') -> Union[str, UUID, Path]:
        if isinstance(id, UUID):
            # FIXME Edit is_uuid_string
            id = str(id)
            return id

    def to_eid(self,
               id: Union[str, Path, UUID, dict],
               cache_dir: Optional[Union[str, Path]] = None) -> str:
        if isinstance(id, UUID):
            return str(id)  # TODO fix in is_uuid
        elif is_exp_ref(id):
            return ref2eid(id, one=self)
        elif isinstance(id, dict):
            assert {'subject', 'number', 'start_time', 'lab'}.issubset(id)
            root = Path(self._get_cache_dir(cache_dir))
            id = root.joinpath(
                id['lab'],
                'Subjects', id['subject'],
                id['start_time'][:10],
                ('%03d' % id['number']))

        if alfio.is_session_path(id):
            return self.eid_from_path(id)
        elif isinstance(id, str):
            if len(id) > 36:
                id = id[-36:]
            if not alfio.is_uuid_string(id):
                raise ValueError('Invalid experiment ID')
            else:
                return id
        else:
            raise ValueError('Unrecognized experiment ID')

    def list(self, eid, keyword='dataset', by_collection=True):
        """
        TODO Function annotations
        :param eid:
        :param keyword:
        :param by_collection:
        :return:

        Examples:
            one.list(eid)['alf']
            one.list(eid, 'dataset-type', by_collection=False)
        """
        results = self.alyx.rest('datasets', 'list', session=eid)
        collection = []
        name = []
        if keyword not in ('dataset', 'dataset-type'):
            raise ValueError('keyword should be either "dataset" or "dataset-type"')
        for r in results:
            collection.append(r['collection'])
            name.append(r['name'] if keyword == 'dataset' else r['dataset_type'])

        if by_collection:
            out = defaultdict(list)
            [out[k].append(v) for k, v in zip(collection, name)]
            return out
        else:
            return name

    def load(self,
             eid: Union[str, Path, UUID],
             items: Listable(str),
             kind: str = 'name',
             cache_dir: Union[str, Path] = None,
             **kwargs) -> Listable(Mapping[str, Any]):
        """
        Load ALF data from cache or remote server.  NB: Uses the 'dataset' endpoint

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        or Path
        :param items: dataset(s), file(s) or object(s) to load
        :param kind: loading type; one of 'object', 'dataset', 'dataset_type', 'collection'
        :param cache_dir: temporarly overrides the cache_dir from the parameter file
        :param kwargs:
        :return: An ALFBunch of data arrays

        Examples:
            wheel = one.load(eid, 'wheel', type='object')
            spikes, clusters = one.load(eid, ('spikes', 'clusters'), type='object')
            spikeglx = one.load(eid, 'spikeglx')
            camera = one.load(session_path, 'camera.times')
            camera_left = one.load(exp_ref, 'leftCamera.times')
            camera_raw = one.load(eid, '*camera', type='collection', namespace='iblrig')
            spikes_clusters, clusters_channels = one.load(eid,
                                                          ['spikes.clusters', 'clusters.channels'])
            spikes = one.load(uuid, 'probe00/spikes')  # Only one probe's spikes
        """
    pass

    @parse_id
    def load_object(self,
                    eid: Union[str, Path, UUID],
                    obj: str,
                    collection: Optional[str] = 'alf',
                    **kwargs) -> alfio.AlfBunch:
        """

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param obj: The ALF object to load
        :param collection:
        :param kwargs:
        :return:

        >>> eid = 'f3ce3197-d534-4618-bf81-b687555d1883'
        >>> load_object(eid, 'spi*')
        >>> load_object(eid, 'spikes')

        TODO Process ID
        TODO Collection check
        TODO Collection 'all'
        TODO Add namespace, etc.
        """
        results = self.alyx.rest('datasets', 'list',
                                 session=eid, django='name__contains,' + obj.replace('*', ''))
        pattern = re.compile(fnmatch.translate(obj))

        def match(r):
            match_name = (is_valid(r['name']) and
                          pattern.match(alf_parts(r['name'])[1]) is not None)
            match_collection = (collection == 'all') or (r['collection'] == collection)
            return match_name and match_collection

        # Get filenames of returned ALF files
        returned_obj = {alf_parts(x['name'])[1] for x in results if match(x)}
        if len(returned_obj) > 1:
            raise ALFMultipleObjectsFound('The following objects were found: ' +
                                          ', '.join(returned_obj))
        elif len(returned_obj) == 0:
            raise ALFObjectNotFound(f'ALF object "{obj}" not found on Alyx')

        assert len({x['collection'] for x in results if match(x)}) == 1

        out_files = self.download_datasets(x for x in results if match(x))
        assert not any(x is None for x in out_files), 'failed to download dataset'
        if kwargs.pop('download_only', False):
            return out_files
        else:
            return alfio.load_object(out_files[0].parent, obj)

    @parse_id
    def load_dataset(self,
                     eid: Union[str, Path, UUID],
                     dataset: str,
                     collection: Optional[str] = 'alf',
                     **kwargs) -> Any:
        """
        TODO Change exceptions
        :param eid:
        :param obj:
        :param collection:
        :param kwargs:
        :return:
        """
        results = self.alyx.rest('datasets', 'list',
                                 session=eid, django='name__contains,' + dataset.replace('*', ''))
        pattern = re.compile(fnmatch.translate(dataset))

        def match(r):
            match_name = pattern.match(r['name']) is not None
            match_collection = (collection == 'all') or (r['collection'] == collection)
            return match_name and match_collection

        # Get filenames of returned ALF files
        returned = [x for x in results if match(x)]
        assert len({x['collection'] for x in returned}) <= 1
        if len(returned) > 1:
            raise ALFMultipleObjectsFound('The following matching datasets were found: ' +
                                          ', '.join(x['name'] for x in returned))
        elif len(returned) == 0:
            raise ALFObjectNotFound(f'Dataset "{dataset}" not found on Alyx')

        filename = self.download_dataset(returned[0])
        assert filename is not None, 'failed to download dataset'

        return filename if kwargs.pop('download_only') else alfio.load_file_content(filename)
