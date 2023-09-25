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
import logging
from iblatlas import regions
from ibllib.atlas import deprecated_decorator

_logger = logging.getLogger(__name__)


@deprecated_decorator
def BrainRegions():
    return regions.BrainRegions()


@deprecated_decorator
def FranklinPaxinosRegions():
    return regions.FranklinPaxinosRegions()


@deprecated_decorator
def regions_from_allen_csv():
    """
    (DEPRECATED) Reads csv file containing the ALlen Ontology and instantiates a BrainRegions object.

    NB: Instantiate BrainRegions directly instead.

    :return: BrainRegions object
    """
    _logger.warning("ibllib.atlas.regions.regions_from_allen_csv() is deprecated. "
                    "Use BrainRegions() instead")
    return BrainRegions()
