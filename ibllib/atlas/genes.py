import logging
from pathlib import Path

import numpy as np
import pandas as pd

from iblutil.io.hashfile import md5
import one.remote.aws as aws

from ibllib.atlas.atlas import AllenAtlas

_logger = logging.getLogger(__name__)


def allen_gene_expression(filename='gene-expression.pqt', folder_cache=None):
    """
    Reads in the Allen gene expression experiments binary data.
    :param filename:
    :param folder_cache:
    :return: a dataframe of experiments, where each record correspond to a single gene expression
    and a memmap of all experiments volumes, size (4345, 58, 41, 67) corresponding to
    (nexperiments, ml, dv, ap). The spacing between slices is 200 um
    """
    OLD_MD5 = []
    DIM_EXP = (4345, 58, 41, 67)
    folder_cache = folder_cache or AllenAtlas._get_cache_dir().joinpath(filename)
    file_parquet = Path(folder_cache).joinpath('gene-expression.pqt')
    file_bin = file_parquet.with_suffix(".bin")

    if not file_parquet.exists() or md5(file_parquet) in OLD_MD5:
        file_parquet.parent.mkdir(exist_ok=True, parents=True)
        _logger.info(f'downloading gene expression data from {aws.S3_BUCKET_IBL} s3 bucket...')
        aws.s3_download_file(f'atlas/{file_parquet.name}', file_parquet)
        aws.s3_download_file(f'atlas/{file_bin.name}', file_bin)
    df_genes = pd.read_parquet(file_parquet)
    gexp_all = np.memmap(file_bin, dtype=np.float16, mode='r', offset=0, shape=DIM_EXP)
    return df_genes, gexp_all
