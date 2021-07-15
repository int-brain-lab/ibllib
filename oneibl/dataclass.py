from functools import singledispatch
from dataclasses import dataclass, field, fields
from pathlib import Path

from brainbox.io import parquet
from iblutil.util import flatten
import warnings

warnings.warn('`oneibl.dataclass` will be removed in future version', DeprecationWarning)


@dataclass
class SessionDataInfo:
    """
    Dataclass that provides dataset list, dataset_id, local_path, dataset_type, url and eid fields
    """
    dataset_type: list = field(default_factory=list)
    dataset_id: list = field(default_factory=list)
    local_path: list = field(default_factory=list)
    eid: list = field(default_factory=list)
    url: list = field(default_factory=list)
    data: list = field(default_factory=list)
    hash: list = field(default_factory=list)
    file_size: list = field(default_factory=list)

    def __str__(self):
        str_out = ''
        d = self.__dict__
        for k in d.keys():
            str_out += (k + '    : ' + str(type(d[k])) + ' , ' + str(len(d[k])) + ' items = ' +
                        str(d[k][0])) + '\n'
        return str_out

    def __getitem__(self, ind):
        return SessionDataInfo(
            dataset_type=self.dataset_type[ind],
            dataset_id=self.dataset_id[ind],
            local_path=self.local_path[ind],
            eid=self.eid[ind],
            url=self.url[ind],
            data=self.data[ind],
            hash=self.hash[ind],
            file_size=self.file_size[ind]
        )

    def __len__(self):
        return len(self.dataset_type)

    def append(self, d):
        def getattr_list(d, name):
            out = getattr(d, name, None)
            if isinstance(out, list) and len(out) == 0:
                out = None
            return out if isinstance(out, list) else [out]
        for f in fields(self):
            setattr(self, f.name, getattr_list(self, f.name) + getattr_list(d, f.name))

    @staticmethod
    def from_datasets(dsets, dataset_types=None, eid=None):
        # if no dataset is specified download only the root alf folder
        if not dataset_types:
            dsets = [d for d in dsets if d['data_url'] and
                     'alf' in Path(d['data_url']).parts and
                     'raw_ephys_data' not in Path(d['data_url']).parts]
        elif dataset_types == ['__all__']:
            dsets = [d for d in dsets if d['data_url']]
        else:
            dsets = [d for d in dsets if d['dataset_type'] in dataset_types]
        return SessionDataInfo(
            dataset_type=[d['dataset_type'] for d in dsets],
            dataset_id=[d['id'] for d in dsets],
            local_path=[None for d in dsets],
            eid=[eid for _ in dsets],  # [ses_info['url'][-36:] for d in dsets],
            url=[d['data_url'] for d in dsets],
            data=[None for _ in dsets],
            hash=[d['hash'] for d in dsets],
            file_size=[d['file_size'] for d in dsets]
        )

    @staticmethod
    def from_session_details(ses_info, **kwargs):
        return _session_details_to_dataclasses(ses_info, **kwargs)

    @staticmethod
    def from_pandas(df, cache_dir):
        if df.size == 0:
            return SessionDataInfo()
        fcn_local_path = lambda rec: Path(cache_dir).joinpath(  # noqa
            rec['lab'], 'Subjects', rec['subject'], rec['start_time'].isoformat()[:10],
            str(rec['number']).zfill(3), rec['collection'], rec['name'])
        nrecs = df.shape[0]
        return SessionDataInfo(
            dataset_type=df.dataset_type.to_list(),
            dataset_id=list(parquet.np2str(df[['id_0', 'id_1']])),
            local_path=df.apply(fcn_local_path, axis=1).to_list(),
            eid=list(parquet.np2str(df[['eid_0', 'eid_1']])),
            url=[None for _ in range(nrecs)],
            data=[None for _ in range(nrecs)],
            hash=df.hash.to_list(),
            file_size=df.file_size.to_list()
        )


@singledispatch
def _session_details_to_dataclasses(ses_info, **kwargs):
    dsets = [d for d in ses_info['data_dataset_session_related']]
    return SessionDataInfo.from_datasets(dsets, **kwargs)


@_session_details_to_dataclasses.register(list)
def _(ses_info: list, **kwargs):
    dsets = flatten([ses['data_dataset_session_related'] for ses in ses_info])
    return SessionDataInfo.from_datasets(dsets, **kwargs)
