# 2025 - Ageing and neural variability

[Download instructions, tag `2025_Q3_Zang_et_al_Aging`](https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html#Find-data-associated-with-a-release-or-publication)

This is the dataset associated with the publication [Age-related changes in behavioural and neural variability in a decision-making task](https://doi.org/10.1038/s41467-026-74227-1).

Behavioural, electrophysiology and video data were collected from C57BL/6 mice performing the [IBL visual decision-making task](https://pubmed.ncbi.nlm.nih.gov/34011433/). The dataset combines previously released brain-wide map recordings with additional lifespan recordings from older mice. Animals were 3–20 months old at the time of recording.

The [analysis repository](https://github.com/Fenying-Zang/Ageing_behavioral_and_neural_variability) contains the code, intermediate results and a query example used for the publication. The analysis code is also archived on [Zenodo](https://doi.org/10.5281/zenodo.20102008).

## Overview of the data

The study analysed 149 mice across 367 sessions and 503 Neuropixels probe insertions. Recordings covered 16 regions of interest in cortex, hippocampus, thalamus, midbrain, basal ganglia and olfactory areas. From 242,671 recorded units, including multi-unit activity, 18,755 neurons passed the quality-control criteria used in the publication.

| Dataset source | Project |
|:---------------|:--------|
| Lifespan recordings | `churchland_learninglifespan` |
| Brain-wide map recordings | `ibl_neuropixel_brainwide_01` |

The paper treats age as a continuous variable. For visualisation, mice were divided into younger and older groups using 7.6 months, the mean age in the dataset, as the cutoff.

## Data structure and download

The organisation of the data follows the standard IBL data structure. The following example queries all sessions included in the release, then separates the lifespan and brain-wide map components.

```python
from one.api import ONE

TAG = '2025_Q3_Zang_et_al_Aging'

one = ONE(base_url='https://openalyx.internationalbrainlab.org')

sessions = one.alyx.rest('sessions', 'list', tag=TAG)
lifespan_sessions = one.alyx.rest(
    'sessions', 'list', tag=TAG, projects='churchland_learninglifespan'
)
brainwide_sessions = one.alyx.rest(
    'sessions', 'list', tag=TAG, projects='ibl_neuropixel_brainwide_01'
)

print(f'All sessions: {len(sessions)}')
print(f'Lifespan sessions: {len(lifespan_sessions)}')
print(f'Brain-wide map sessions: {len(brainwide_sessions)}')
```

To go further, see:

* [The data structure guide](https://int-brain-lab.github.io/iblenv/notebooks_external/data_structure.html) to download an example session and learn the IBL data structure
* [The data download guide](https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html) to search for and download released datasets with ONE
* [The data-loading examples](https://int-brain-lab.github.io/iblenv/loading_examples.html) for specific loading functions
* [The publication's query example](https://github.com/Fenying-Zang/Ageing_behavioral_and_neural_variability/blob/main/query_example_notebook.ipynb) for the exact project-level queries used by the authors

Note:

* The tag associated with this release is `2025_Q3_Zang_et_al_Aging`.
* The release contains sessions from the `churchland_learninglifespan` and `ibl_neuropixel_brainwide_01` projects.

## How to cite this dataset

If you use this dataset in your research, please cite:

* Zang, F., Khanal, A., Förster, S. *et al.* [Age-related changes in behavioural and neural variability in a decision-making task](https://doi.org/10.1038/s41467-026-74227-1). *Nature Communications* **17**, 8156 (2026).
* For analysis code and intermediate results, Zang, F. & Rai, P. [Age-related changes in behavioral and neural variability in a decision-making task](https://doi.org/10.5281/zenodo.20102008). Zenodo (2026).
* If you use the brain-wide map component, follow the citation guidance on the [brain-wide map data release page](https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html#How-to-cite-this-dataset).

## Data release notes and changelog

### 2025 Q3: initial release

The release tag used for the publication was added, combining the lifespan recordings with the brain-wide map sessions included in the analysis.
