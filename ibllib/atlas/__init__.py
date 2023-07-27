"""A package for working with brain atlases.

For examples and tutorials on using the IBL atlas package, see
https://docs.internationalbrainlab.org/atlas_examples.html


Terminology
-----------
There are many terms used somewhat incoherently within this API and the community at large. Below
are some definitions of the most common terms.

* **Atlas** - A set of serial sections along different anatomical planes of a brain where each
  relevant brain structure is assigned a number of coordinates to define its outline or volume.
  An atlas essentially comprises a set of images, annotations, and a coordinate system.
* **Annotation** - A set of identifiers assigned to different atlas regions.
* **Mapping** - A function that maps one ordered list of brain region IDs to another, allowing one
  to control annotation granularity and brain region hierarchy, or to translate brain region names
  from one atlas to another. The default mapping is identity.
* **Coordinate framework** - The way in which an atlas translates image coordinates (e.g. Cartesian
  or sperical coordinates) to real-world anatomical coordinates in (typically physical distance
  from a given landmark such as bregma, along three axes, ML-AP-DV).
* **Reference space** - The coordinate system and annotations used by a given atlas. It is
  sometimes useful to compare anatomy between atlases, which requires expressing one atlas in
  another's reference space.
* **Scaling** - Atlases typically comprise images averaged over a number of brains. Scaling allows
  one to account for any consistent and measurable imgaging or tissue distortion, or to better
  align to an individual brain of a specific size. The default scaling is identity.
* **Flat map** - An annotated projection of the 3D brain to 2D.
* **Slice** - A 2D section of a brain atlas volume. Typically these are coronal (cut along the
  medio-lateral axis), sagittal (along the dorso-ventral axis) or transverse a.k.a. axial,
  horizontal (along the rostro-caudal a.k.a. anterio-posterior axis).


Atlases
-------
There are two principal mouse brain atlases in this module:

1. The Allen Common Coordinate Framework (CCF)[1]_.
2. The Mouse Brain in Stereotaxic Coordinates (MBSC) 4th Edition, by Paxinos G, and Franklin KBJ[2]_.

The latter is referred to here as the 'Franklin-Paxinos atlas'.  These atlases comprise a 3D array
of voxels and their associated brain region identifiers (labels) at a given resolution.


Scalings
--------

Additionally there are two further atlases that apply some form of scaling to the Allen CCF atlas
to account for distortion that occurs during the imaging and fixation process:

1. The Needles atlas -  14 C57BL/6 mice underwent MRI and conventional Nissl histology, then the
   images were transformed onto the Allen CCF atlas to determine the scaling[3]_.
2. The MRI Toronto - 12 p65 mice MRI images were averaged and transformed on the Allen CCF atlas to determine the scaling[4]_.

Scaling of this kind can be applied arbitrarily to better represent mouse age and sex[4]_.

TODO Mention FlatMap class.


Mappings
--------
In addition to the atlases there are also multiple brain region mappings that serve one of two
purposes: 1. control the granularity particular brain regions; 2. support differing anatomical
sub-devisions or nomenclature.  TODO Give examples


Notes
-----
The Allen atlas and the CCF annotations have different release dates and versions[5]_. The
annotations used by IBL are the 2017 version.

The IBL uses the following conventions:

- All atlas images have dimensions (AP, ML, DV). With C-ordering this makes coronal slicing most
  efficient.
- Coordinates are provided in the order (ML AP DV) and are in meters relative to bregma.
- Left hemisphere ML coordinates are -ve; right, +ve.
- AP coordinates anterior to bregma are +ve; posterior, -ve.
- DV coordinates ventral to bregma are -ve; ventral +ve.
- Bregma was determined by asking five experimentalists to pick the voxel containing bregma on the
  Allen atlas and taking the average.  NB: The midline appears slightly off-center in the Allen
  atlas image volume.


References
----------
.. [1] © 2015 Allen Institute for Brain Science. Allen Mouse Brain Atlas (2015) with region annotations (2017).
   Available from: http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/
.. [2] Paxinos G, and Franklin KBJ (2012). The Mouse Brain in Stereotaxic Coordinates, 4th edition (Elsevier Academic Press)
.. [3] Johnson GA, Badea A, Brandenburg J, Cofer G, Fubara B, Liu S, Nissanov J (2010). Waxholm space: an image-based reference
   for coordinating mouse brain research. Neuroimage 53(2):365-72. [doi 10.1016/j.neuroimage.2010.06.067]
.. [4] Qiu, L.R., Fernandes, D.J., Szulc-Lerch, K.U. et al. (2018). Mouse MRI shows brain areas relatively larger
   in males emerge before those larger in females. Nat Commun 9, 2615. [doi 10.1038/s41467-018-04921-2]
.. [5] Allen Mouse Common Coordinate Framework Technical White Paper (October 2017 v3)
       http://help.brain-map.org/download/attachments/8323525/Mouse_Common_Coordinate_Framework.pdf


Examples
--------
Below are some breif API examples. For in depth tutorials on using the IBL atlas package, see
https://docs.internationalbrainlab.org/atlas_examples.html.

Find bregma position in indices * resolution in um

>>> ba = AllenAtlas()
>>> bregma_index = ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / ba.res_um

Find bregma position in xyz in m (expect this to be 0 0 0)

>>> bregma_xyz = ba.bc.i2xyz(bregma_index)


Fixtures
--------

* TODO List the data files in this package, their purpose, data types, shape, etc.
* TODO List the remote files used by this package, e.g. annotations files, swansonpaths.json, etc.

### Local files

* **allen_structure_tree.csv** - TODO Document
* **beryl.npy** - TODO Document
* **cosmos.npy** - TODO Document
* **franklin_paxinos_structure_tree.csv** - TODO Document
* **mappings.pqt** - TODO Document
* **swanson_regions.npy** - TODO Document

### Remote files

* **annotation_<res_um>.nrrd** - TODO Document
* **average_template_<res_um>.nrrd** - TODO Document
* **annotation_<res_um>_lut_<LUT_VERSION>.npz** - TODO Document
* **swansonpaths.json** - TODO Document
* **swanson2allen.npz** - TODO Document
* **<name>_<res_um>.nrrd** - TODO Document
"""
from .atlas import *  # noqa
from .regions import regions_from_allen_csv
from .flatmaps import FlatMap
