# Introduction to Datajoint

Datajoint is a workflow management system that integrates a relational database with computational data pipelines that
are programmed and accessed directing using Matlab or Python. To find out more about datajoint please visit their 
website [https://datajoint.io/](https://datajoint.io/). The IBL has collaborated with Datajoint to develop the 
IBL-pipeline, a set of tables organised in a structured manner that contain most experimental data and metadata 
collected within the IBL. Any new data generated in the collaboration is ingested into the IBL-pipeline on a daily basis
and so this framework can be used to access the latest data collected within the IBL.


There are three ways in which internal IBL users can use Datajoint to interact with the data stored in the IBL-pipeline.

```{important}
To access the IBL-pipeline, you will need a Datajoint username and password as well as global the IBL Navigator login 
details. To find these out, please get in contact with a member of the IBL software team.
```

1)  **IBL Navigator website [https://djcompute.internationalbrainlab.org/](https://djcompute.internationalbrainlab.org/)**

    Datajoint provides a website that displays behavioural and electrophysiological plots generated from data 
    contained within the IBL-pipeline. New plots are generated on a daily basis and so users can use the website to get an 
    overview of the latest behavioural and ephys data available. 

    
2)  **[jupyter.internationalbrainlab.org](https://jupyter.internationalbrainlab.org)**

    Datajoint hosts a JupyterHub server with access to the IBL-pipeline. This platform can be used to programatically 
    explore and perform analysis on data stored within the IBL-pipeline without requiring a local install of dependant 
    packages. To use the platform you will need to register your github account as well as have access to your Datajoint 
    login details. To get started using the IBL-pipeline please proceed to the 
    [Datajoint basics tutorial](../notebooks/dj_basics/dj_basics.ipynb). 


3)  **Accessing IBL-pipeline on your local computer**

    The most flexible way to use Datajoint with IBL data is to install the IBL-pipeline package onto your local computer. 
    This package is automatically installed as part of the iblenv unified environment and so if you have set this up you are
    ready to start using Datajoint in this way. The only thing you will need to do is to set up some local credentials. The 
    [instructions on the next page](dj_credentials.md) will guide you through how to set these up. 

    
For external users looking to use Datajoint to access the public IBL dataset please proceed to this website 
[https://data.internationalbrainlab.org/](https://data.internationalbrainlab.org/). Here you will find overview plots of
all the behavioural data available as well as instructions of how to set up access the data.

