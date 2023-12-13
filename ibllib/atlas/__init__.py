"""(DEPRECATED) A package for working with brain atlases.

For the correct atlas documentation, see
https://docs.internationalbrainlab.org/_autosummary/iblatlas.html

For examples and tutorials on using the IBL atlas package, see
https://docs.internationalbrainlab.org/atlas_examples.html

.. TODO Explain differences between this package and the Allen SDK.
Much of this was adapted from the `cortexlab allenCCF repository <https://github.com/cortex-lab/allenCCF>`_.

Terminology
-----------
There are many terms used somewhat incoherently within this API and the community at large. Below
are some definitions of the most common terms.

* **Atlas** - A set of serial sections along different anatomical planes of a brain where each relevant brain structure is
  assigned a number of coordinates to define its outline or volume. An atlas essentially comprises a set of images, annotations,
  and a coordinate system.
* **Annotation** - A set of identifiers assigned to different atlas regions.
* **Mapping** - A function that maps one ordered list of brain region IDs to another, allowing one to control annotation
  granularity and brain region hierarchy, or to translate brain region names from one atlas to another. The default mapping is
  identity.
* **Coordinate framework** - The way in which an atlas translates image coordinates (e.g. Cartesian or sperical coordinates) to
  real-world anatomical coordinates in (typically physical distance from a given landmark such as bregma, along three axes,
  ML-AP-DV).
* **Reference space** - The coordinate system and annotations used by a given atlas. It is sometimes useful to compare anatomy
  between atlases, which requires expressing one atlas in another's reference space.
* **Structure tree** - The hirarchy of brain regions, handled by the BrainRegions class.
* **Scaling** - Atlases typically comprise images averaged over a number of brains. Scaling allows one to account for any
  consistent and measurable imgaging or tissue distortion, or to better align to an individual brain of a specific size. The
  default scaling is identity.
* **Flat map** - An annotated projection of the 3D brain to 2D.
* **Slice** - A 2D section of a brain atlas volume. Typically these are coronal (cut along the medio-lateral axis), sagittal
  (along the dorso-ventral axis) or transverse a.k.a. axial, horizontal (along the rostro-caudal a.k.a. anterio-posterior axis).


Atlases
-------
There are two principal mouse brain atlases in this module:

1. The Allen Common Coordinate Framework (CCF) [1]_.
2. The Mouse Brain in Stereotaxic Coordinates (MBSC) 4th Edition, by Paxinos G, and Franklin KBJ [2]_, matched to
   to the Allen Common Coordiante Framework by Chon et al. [3]_.

The latter is referred to here as the 'Franklin-Paxinos atlas'.  These atlases comprise a 3D array of voxels and their associated
brain region identifiers (labels) at a given resolution. The Allen Atlas can be instantiated in 10um, 25um or 50um resolution.
The Franklin-Paxinos atlas has a resolution of 10um in the ML and DV axis, and 100um in the AP axis. **TODO Mention flat maps.**


Scalings
--------
Additionally there are two further atlases that apply some form of scaling to the Allen CCF atlas
to account for distortion that occurs during the imaging and tissue fixation process:

1. The Needles atlas - 40 C57BL/6J (p84) mice underwnt MRI imaging post-mortem while the brain was still in the skull, followed by
   conventional Nissl histology [4]_. These mouse brain atlas images combined with segmentation (known as DSURQE) were manually
   transformed onto the Allen CCF atlas to determine the scaling.
2. The MRI Toronto - 12 p65 mice MRI images were taken *in vivo* then averaged and transformed on the Allen CCF atlas to determine
   the scaling [5]_.

All scaling is currently linear. Scaling of this kind can be applied arbitrarily to better represent a specific mouse age and
sex [5]_. NB: In addition to distortions, the Allen CFF atlas is pitched down by about 5 degrees relative to a flat skull (where
bregma and lambda are at the same DV height) [6]_, however this is not currently accounted for.


Mappings
--------
In addition to the atlases there are also multiple brain region mappings that serve one of two purposes: 1. control the
granularity particular brain regions; 2. support differing anatomical sub-devisions or nomenclature.  The two Allen atlas mappings
below were created somewhat arbirarily by Nick Steinmetz to aid in analysis:

1. Beryl - brain atlas annotations without layer sub-divisions or certain ganglial/nucleus sub-devisisions (e.g. the core/shell
   sub-division of the lateral geniculate nucleus). Fibre tracts, pia, etc. are also absent.  The choice of which areas to combine
   was guided partially by the computed volume of each area.  This mapping is used in the brainwide map and prior papers [7]_, [8]_
   .
2. Cosmos - coarse brain atlas annotations, dividing the atlas into 10 broad areas: isocortex, olfactory areas, cortical subplate,
   cerebral nuclei, thalamus, hypothalamus, midbrain, hindbrain, cerebellum and hippocampal formation.

The names of these two mappings appear to be without meaning.

Non-Allen mappings:

3. Swanson - the brain atlas annotations from the Swansan rat brain flat map [9]_, mapped to the Allen atlas manually by Olivier
   Winter. See `Fixtures`_ for details.

Each mapping includes both a lateralized (suffix '-lr') and non-laterized version. The lateralized mappings assign a different ID
to structures in the right side of the brain. The Allen atlas IDs are kept intact but lateralized as follows: labels are
duplicated and IDs multiplied by -1, with the understanding that left hemisphere regions have negative IDs. There is currently no
mapping between Franklin & Paxinos and the Allen atlases.


Notes
-----
The Allen atlas and the CCF annotations have different release dates and versions [10]_. The annotations used by IBL are the 2017
version.

The IBL uses the following conventions:

- All atlas images have dimensions (AP, ML, DV). With C-ordering this makes coronal slicing most efficient. The origin is the top
  left corner of the image.
- Coordinates are provided in the order (ML AP DV) and are in meters relative to bregma.
- Left hemisphere ML coordinates are -ve; right, +ve.
- AP coordinates anterior to bregma are +ve; posterior, -ve.
- DV coordinates ventral to bregma are -ve; ventral +ve.
- Bregma was determined by asking five experimentalists to pick the voxel containing bregma on the Allen atlas and taking the
  average.  NB: The midline appears slightly off-center in the Allen atlas image volume.
- All left hemisphere regions have negative region IDs in all lateralized mappings.


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

.. TODO List the data files in this package, their purpose, data types, shape, etc.
.. TODO List the remote files used by this package, e.g. annotations files, swansonpaths.json, etc.

Local files
^^^^^^^^^^^

* **allen_structure_tree.csv** - TODO Document. Where does this come from? Is it modified from either structure_tree_safe.csv or
  structure_tree_safe_2017.csv?
* **franklin_paxinos_structure_tree.csv** - Obtained from Supplementary Data 2 in reference [10].
* **beryl.npy** - A 306 x 1 int32 array of Allen CCF brain region IDs generated in MATLAB [*]_. See more information see
  `Mappings`_.
* **cosmos.npy** - A 10 x 1 int32 array of Allen CCF brain region IDs generated in MATLAB [*]_. See more information see
  `Mappings`_.
* **swanson_regions.npy** - A 1D array of length 323 containing the Allen CCF brain region IDs
* **mappings.pqt** - A table of mappings.  Each column defines a mapping, with the '-lr' suffix indicating a lateralized version.
  The rows contain the correspondence of each mapping to the int64 index of the lateralized Allen structure tree.  The table is
  generated by ibllib.atlas.regions.BrainRegions._compute_mappings.

Remote files
^^^^^^^^^^^^

* **annotation_<res_um>.nrrd** - A 3D volume containing indicies of the regions in the associated
  structure tree.  `res_um` indicates the isometric spacing in microns.  These uint16 indicies are
  known as the region 'index' in the structure tree, i.e. the position of the region in the
  flattened tree.
* **average_template_<res_um>.nrrd** - TODO Document
* **annotation_<res_um>_lut_<LUT_VERSION>.npz** - TODO Document
* **FranklinPaxinons/annotation_<res_um>.npz** - A 3D volume containing indices of the regions associated with the Franklin-
  Paxinos structure tree.
* **FranklinPaxinons/average_template_<res_um>.npz** - A 3D volume containing the Allen dwi image slices corresponding to
  the slices in the annotation volume [*] .
* **swansonpaths.json** - The paths of a vectorized Swanson flatmap image [*]. The vectorized version was generated
  from the Swanson bitmap image using the matlab contour function to find the paths for each region. The paths for each
  region were then simplified using the `Ramer Douglas Peucker algorithm <https://rdp.readthedocs.io/en/latest/>`_
* **swanson2allen.npz** - TODO Document who made this, its contents, purpose and data type
* **<flatmap_name>_<res_um>.nrrd** - TODO Document who made this, its contents, purpose and data type
* **gene-expression.pqt** - TODO Document who made this, its contents, purpose and data type
* **gene-expression.bin** - TODO Document who made this, its contents, purpose and data type.

.. [*] The annotation and average template volumes were created from the images provided in Supplemtary Data 4 of Chon et al. [3]_
   and stitched together as a single volume using SimpleITK.
.. [*] output of aggType 2 in https://github.com/cortex-lab/allenCCF/blob/master/Browsing%20Functions/aggregateAcr.m
.. [*] output of aggType 1 in https://github.com/cortex-lab/allenCCF/blob/master/Browsing%20Functions/aggregateAcr.m
.. [*] the paths were generated from a bitmap of the
   `BM3 rat flatmap 3.0 foldout poster <https://larrywswanson.com/wp-content/uploads/2015/03/BM3-flatmap-foldout.pdf>`_
   in `Swanson LW (2004) Brain Maps, 3rd ed. <https://larrywswanson.com/?page_id=164>`_ TODO where is code for this?


References
----------
.. [1] © 2015 Allen Institute for Brain Science. Allen Mouse Brain Atlas (2015) with region annotations (2017).
   Available from: http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/
.. [2] Paxinos G, and Franklin KBJ (2012) The Mouse Brain in Stereotaxic Coordinates, 4th edition (Elsevier Academic Press)
.. [3] Chon U et al (2019) Enhanced and unified anatomical labeling for a common mouse brain atlas
   [doi 10.1038/s41467-019-13057-w]
.. [4] Dorr AE, Lerch JP, Spring S, Kabani N, Henkelman RM (2008). High resolution three-dimensional brain atlas using an average
   magnetic resonance image of 40 adult C57Bl/6J mice. Neuroimage 42(1):60-9. [doi 10.1016/j.neuroimage.2008.03.037]
.. [5] Qiu, LR, Fernandes, DJ, Szulc-Lerch, KU et al. (2018) Mouse MRI shows brain areas relatively larger
   in males emerge before those larger in females. Nat Commun 9, 2615. [doi 10.1038/s41467-018-04921-2]
.. [6] International Brain Laboratory et al. (2022) Reproducibility of in-vivo electrophysiological measurements in mice.
   bioRxiv. [doi 10.1101/2022.05.09.491042]
.. [7] International Brain Laboratory et al. (2023) A Brain-Wide Map of Neural Activity during Complex Behaviour.
   bioRxiv. [doi 10.1101/2023.07.04.547681]
.. [8] Findling C et al. (2023) Brain-wide representations of prior information in mouse decision-making.
   bioRxiv. [doi 10.1101/2023.07.04.547684]
.. [9] Swanson LW (2018) Brain maps 4.0—Structure of the rat brain: An open access atlas with global nervous system nomenclature
   ontology and flatmaps. J Comp Neurol. [doi 10.1002/cne.24381]
.. [10] Allen Mouse Common Coordinate Framework Technical White Paper (October 2017 v3)
   http://help.brain-map.org/download/attachments/8323525/Mouse_Common_Coordinate_Framework.pdf

"""
from .atlas import *  # noqa
from .regions import regions_from_allen_csv
from .flatmaps import FlatMap
import warnings

warnings.warn('ibllib.atlas is deprecated. Please install iblatlas using "pip install iblatlas" and use '
              'this module instead', DeprecationWarning)
