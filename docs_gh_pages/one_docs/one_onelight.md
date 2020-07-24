# Accessing data with ONE light

ONE light is an implementation of ONE that can be used independently of the full IBL data architecture. Providing data 
is hierachically organised into sub-folders and file-names that match the ONE convention, ONE light can use the same 
[basic ONE commands](../notebooks/one_basics/one_basics.ipynb) to search for and load data of interest. ONE light 
supports data that has been uploaded to a webserver, to figshare (a site offering free hosting of scientific data) or 
that is stored on a user's local machine. For an example implementation of ONE Light and more information about how to 
use this interface to automatically upload data to figshare, please refer to this 
[page](https://github.com/int-brain-lab/ibllib/tree/onelight/).

The behavioural data associated with the IBL paper: [A standardized and reproducible method to measure decision-making 
in mice](https://doi.org/10.1101/2020.01.17.909838) has been made available through the ONE Light interface. To get 
started use ONE Light with this data, download `**ibl-behavior-data-Dec2019.zip**` onto your local computer from 
[here](https://figshare.com/articles/A_standardized_and_reproducible_method_to_measure_decision-making_in_mice_Data/11636748)
and follow this [tutorial](https://mybinder.org/v2/gh/int-brain-lab/paper-behavior-binder/master?filepath=one_example.ipynb).

You can also check out this 
[google colab notebook](https://colab.research.google.com/drive/19BTZT1zsduLXdT9GGbVzIw1g3Lt38VfW?usp=sharing) that uses
ONE Light with the IBL behavioural data to replicate the figures presented in this 
[paper](https://doi.org/10.1101/2020.05.21.109678).










