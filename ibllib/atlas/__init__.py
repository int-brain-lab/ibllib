"""A package for working with brain atlases.

TODO A longer description of this package and how we work with atlases at IBL, namely the bregma
 coordinate system and lateralization.

For examples and tutorials on using the IBL atlas package, see
https://docs.internationalbrainlab.org/atlas_examples.html

Supported brain atlases:

* TODO list atlases and their citations, along with the function/class

Fixtures:

* TODO List the data files in this package, their purpose, data types, shape, etc.

http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/
"""
from .atlas import *  # noqa
from .regions import regions_from_allen_csv
from .flatmaps import FlatMap
