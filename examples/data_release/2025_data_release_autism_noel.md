# 2025 - Autism


This is the dataset associated with the publication __A common computational and neural anomaly across mouse models of autism__
Nature Neuroscience | Volume 28 | July 2025 | 1519–1532 1519 nature neuroscience
https://doi.org/10.1038/s41593-025-01965-8

The following data repository contains intermediate results and code to reproduce the analysis: https://osf.io/fap2s/ and https://osf.io/fap2s/wiki/home/.

## Notes

## Overview of the Data
We have released data from 198 Neuropixel experimental sessions, with single Neuropixel recordings, referred to as probe insertions. Those were obtained with 62 genetically modified subjects performing the IBL task.
As output of spike-sorting, there are XXX units; of which XXX are considered to be of good quality. In total, XXX brain regions were recorded in sufficient numbers for comparison with IBL’s analysises [(IBL et al. 2023)](https://doi.org/10.1101/2023.07.04.547681).


| Model | Subject Count | Neuropixel Count |
| --- |---------------| --- |
| Wildtype | 15 | 51|
| CS       | 17 | 52|
| SH       | 15 | 41|
| FM       | 15 | 54|

## Data structure and download
The organisation of the data follows the standard IBL data structure.

Please see

* [These instructions](https://int-brain-lab.github.io/iblenv/notebooks_external/data_structure.html) to download an example dataset for one session, and get familiarised with the data structure
* [These instructions](https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html) to learn how to use the ONE-api to search and download the released datasets
* [These instructions](https://int-brain-lab.github.io/iblenv/loading_examples.html) to get familiarised with specific data loading functions

Note:

* The tag associated to this release is `2025_Q3_Noel_et_al_Autism`


## How to cite this dataset
If you are using this dataset for your research please cite the paper [A common computational and neural anomaly across mouse models of autism](https://doi.org/10.1038/s41593-025-01965-8).


## Data release notes and changelog

### 2025-08-29: initial release

### Excluded sessions

The following sessions were excluded as the synchronisation stream from the bpod couldn't be found in the FPGA. As such the behaviour information couldn't be extracted to a clock synced to the electrophysiology.

| EID | Subject | Date | Number |
| --- | --- | --- | --- |
| 5d4e158b-7d6d-48fd-ad94-74ad4704e89f | CSP018 | 2021-01-20 | 001 |
| 8507e9f6-4da3-454a-b553-3fc8f6299bbb | SH014 | 2020-07-13 | 001 |
| e9620e9a-688a-45da-ba6e-33fce6753729 | SH008 | 2020-03-04 | 001 |
| 429de7e8-1fc9-4aa2-87a1-0800268935d7 | CSP023 | 2020-11-17 | 001 |


### Video data: QC report
We have only included video data where we could confidently align the frames timing to the main experiment clock.

### Electrophysiology: QC report

#### Quality of uppermost channels
XXX

#### CRITICAL: Missing spike sorting
The following insertions do not have any spike sorting available.

| PID | EID | Subject | Date | Number | Probe |
| --- | --- | --- | --- | --- | --- |
| bcb1dac7-6d2b-47ad-bbbe-a4aaf9774481 | b33a5ab8-7b51-4fe4-9ee2-584fb430b41b | FMR025 | 2021-09-02 | 001  | probe01 |
| a25d9370-a714-4662-88c5-ccf69b646bd5 | 1a4479ec-8511-48c9-9169-55a20f2626f9 | FMR031 | 2022-03-22 | 001  | probe01 |
| 7bf6ff8d-6487-481e-a174-d1cbe31db2c4 | baab8aae-273e-4f83-80a9-81e9b7f99185 | FMR032 | 2022-03-15 | 001  | probe01 |
| 65c6349d-6ec3-4a7d-aefc-7cfe328c1faf | 2fe8aa16-faab-49e3-8e13-aceb0b095a30 | SH006 | 2020-02-26 | 001  | probe00 |
| 7b00f29f-f67f-4fa9-b4c9-e844c44b7b6b | ccffff9f-c432-4377-b228-e2710bc109b6 | SH011 | 2020-02-06 | 001  | probe01 |
| 8ade2696-5a06-4e0e-8f73-674aaf60a6ca | 8658a5ad-58ce-4626-8d46-be68cd33581b | SH011 | 2020-02-07 | 001  | probe01 |
| 1606f00b-e4a0-434e-a81b-a492016b42d9 | 792fe5db-dd66-4fc7-ab96-accfce28b7f4 | SH025 | 2022-02-01 | 002  | probe00 |
| 2a32115c-bb3e-42ec-81a4-c24f11d1721f | 68c22775-1b40-44ff-8802-b95d0152d565 | SH025 | 2022-02-02 | 001  | probe01 |
| 474965f4-5751-497b-9423-cb130fc30644 | 2d792481-75e5-4e51-a189-20e7cbbcd8ad | SH025 | 2022-02-04 | 002  | probe00 |
| de7639c4-0f81-4370-84f2-70dde255eca7 | ae21bab0-65dc-40a2-bc6d-48a6a07e9c82 | SH027 | 2022-01-27 | 001  | probe01 |
| 1034a9c5-85cd-49c1-bf6b-0fb2cbc4ee67 | 576de022-4a2b-4423-8f7f-53f83b1b896e | CSP003 | 2019-11-22 | 001  | probe01 |
| ef8eb985-3731-48c5-ac62-38214f8d8ee4 | 95fa5278-3870-4087-9c7d-a306a068d334 | CSP031 | 2022-02-01 | 001  | probe01 |
| eddf08b2-2b1f-489c-bceb-f28592518f61 | bf708a95-5980-463d-b54f-329b49754313 | CSP032 | 2022-01-27 | 001  | probe00 |
| c31e3510-cc05-4992-ac30-808d3b3f0d81 | d133df68-dec9-4979-a500-a549719424d1 | CSP032 | 2022-01-28 | 001  | probe00 |
| 131716c1-515e-4a45-9158-cf1af6da39c7 | 86bef629-a95d-4a68-890b-2f7cabb58504 | NYU-57 | 2021-11-18 | 001  | probe00 |


#### CRITICAL: Missing Histology tracing

| PID | EID | Subject | Date | Number | Probe |
| --- | --- | --- | --- | --- | --- |
| 7d475dc7-a60e-4418-837d-fa9c43a91cff | fb5831ac-d15b-437b-98fe-47d03d7edc15 | SH002 | 2019-11-25 | 001  | probe01 |
| 2fe497f6-95e8-4349-9b92-47f26265b784 | 0c2f24d9-5184-43e6-97dc-17b31cb8cee9 | SH004 | 2020-02-10 | 002  | probe00 |
| ada946d0-1195-4f04-9d45-8e7e6adf7f60 | cdfb4449-30fb-4dd9-85cd-c8dd385ddf75 | SH013 | 2020-07-23 | 001  | probe00 |
| 2eeeee9a-678e-47c9-b2d9-22daec55ddbb | b73a16d7-555d-4c51-91a0-611d0ed0a975 | CSP003 | 2019-11-19 | 001  | probe00 |
| 5831925c-4b27-425f-9b7a-ff8b0691e9f3 | c1f807b4-7538-4f82-a9a6-52d658eb0bd2 | CSP017 | 2020-11-17 | 001  | probe01 |
| be8f5333-ea97-4a80-8574-ab937b2087cd | 5f1a76fa-a1b6-4140-83f2-e891db4e11a8 | CSP017 | 2020-11-18 | 001  | probe01 |
| d8ed8bbe-c2fc-46ad-b769-872d326a8179 | 579881ec-e2f8-4079-b109-100e9ddbe8c0 | CSP017 | 2020-11-19 | 001  | probe01 |
| 553b258e-e21d-48b2-8065-21246c82e51a | 9c257bc7-ac32-4255-bb03-8ff90dfc2547 | CSP017 | 2020-11-20 | 001  | probe00 |


#### ERROR: Missing alignments

Here we have spike sorting and histology tracing, but the channels haven´t  been aligned. We are not releasing those datasets for the time being.
Those would be good candidates for a future revision if the ephys atlas task force devises a tool to perform automatic alignments.

| PID | EID | Subject | Date | Number | Probe |
| --- | --- | --- | --- | --- | --- |
| 66887465-3ace-43d6-b609-5e5e1878e8bd | 41eaf4a1-62d6-445f-9284-840b082b31da | FMR030 | 2022-03-29 | 001  | probe01 |
| eacf11c1-47d5-4245-b715-0acc29ccec5c | 77d47a6a-6e05-4e3e-9c85-49a62feada2a | FMR030 | 2022-03-31 | 001  | probe01 |
| 131e4225-f661-4d2a-9e82-ce5c476ca33c | 813e255f-9958-4faf-939d-4d1fdc6536de | FMR030 | 2022-04-01 | 001  | probe01 |
| 58c392e4-1a2e-455e-89eb-65a3ceab8093 | 9c09d626-ef3f-4e51-b353-a578531a8a4b | FMR030 | 2022-04-04 | 001  | probe01 |
| d805f64c-87dc-459c-b23d-8970483c3127 | badc1a08-030c-4144-91f4-d002a05b6d1f | SH020 | 2021-09-22 | 001  | probe00 |
| f4fb1053-61a7-482a-9099-39ed436dd756 | 81adee43-8f38-459c-b9ec-f8fee29ddb60 | SH024 | 2021-09-28 | 001  | probe00 |
| 6f74c301-8d73-49c3-b2fd-6ad1a74528ce | d6492634-aa84-4b51-ab8e-2434648f2d83 | SH024 | 2021-09-29 | 001  | probe00 |
| af95a0e6-073e-416c-9209-7ab30da8ce02 | 66687fe4-c6f8-40db-96c1-9fc49e07d2b6 | SH024 | 2021-09-30 | 001  | probe00 |
| 27271de8-f1ff-4299-a547-05f1be477417 | 1862cb56-2241-42c6-9298-a0861d8fb175 | SH024 | 2021-10-01 | 001  | probe00 |
| 685fb0f1-ed10-4a8b-8e10-51b42a7d67eb | 54859a45-23eb-41ed-b0dc-aa6d6607f006 | SH026 | 2022-03-01 | 001  | probe01 |
| fcdc2472-1025-41ef-bc0f-2c026c35dd6e | a152d17b-b1e3-4994-89a0-86edbf28661e | SH026 | 2022-03-02 | 001  | probe01 |
| 94d1359a-3e87-40a5-b747-7e2b85a4330a | 340a0f2b-0201-4c8b-a491-57b8b6320498 | SH026 | 2022-03-03 | 001  | probe01 |
| 2793cec9-4422-47b4-a58c-4dc5a71baa41 | 7b8d00bf-d7d9-43a4-8f69-50217cfb284c | SH026 | 2022-03-04 | 001  | probe01 |
| 9dd7e4a7-391a-4f61-982f-0efa470ccf59 | fdd79794-88ea-4a9c-910b-524c150dec48 | CSP001 | 2019-11-20 | 001  | probe00 |
| 326c22d1-dbce-41dd-92f7-26d6e4a8d9dc | 147d9be2-ab3a-4dd6-a9b8-fdf6fc129d84 | CSP003 | 2019-11-21 | 002  | probe00 |
| a6b0d2db-c3fc-4967-bdbf-e3f338f3af5e | 9b304cf6-a878-4539-bfd0-aeebcf07f8d1 | CSP023 | 2020-11-18 | 001  | probe01 |
| c9eb42ff-3d53-4044-9468-93e89d870368 | 9854d7b0-ab63-48d5-a7d6-6b5bf0cf5a30 | CSP026 | 2021-06-08 | 001  | probe00 |
| 051a6c5f-4a75-42e2-aa82-b4e3cbe1cecf | 26ef4db9-d1b2-4df8-8de7-519accb9883c | CSP026 | 2021-06-09 | 002  | probe00 |
| 8e5c51e1-690d-4b50-88c4-37242d5ccb65 | 8d3b57dd-4651-4c5f-a4c3-8de8865fdca3 | CSP026 | 2021-06-10 | 001  | probe00 |
| fab06f3a-9ff5-405d-957c-6e628c25af3d | 7b3be2f3-e11c-4352-b73f-10e7813ccec9 | CSP026 | 2021-06-11 | 001  | probe00 |
| 8d6c9ffc-6606-4f11-89bf-cc422ce5022a | 50911ee6-732b-4b07-afe0-8c48a287c803 | NYU-49 | 2021-07-23 | 001  | probe01 |


## Possible future evolutions
- Spike sorting re-run with iblsorter or newer algorithm to have better yield and better pre-processing
- Perform the missing alignments with an automatic tool
