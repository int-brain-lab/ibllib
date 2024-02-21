"""Gene expression maps."""

from iblatlas.genomics import agea
from ibllib.atlas import deprecated_decorator


@deprecated_decorator
def allen_gene_expression(filename='gene-expression.pqt', folder_cache=None):
    """
    Reads in the Allen gene expression experiments binary data.
    :param filename:
    :param folder_cache:
    :return: a dataframe of experiments, where each record corresponds to a single gene expression
    and a memmap of all experiments volumes, size (4345, 58, 41, 67) corresponding to
    (nexperiments, ml, dv, ap). The spacing between slices is 200 um
    """

    return agea.load(filename=filename, folder_cache=folder_cache)
