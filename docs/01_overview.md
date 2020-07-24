# IBL data structure
![Alyx data structure](./_static/IBL_data.png)

-   **Alyx**: database that stores meta-data in a relational manner. Lives on the cloud.
-   **Datajoint**: processing Pipeline for IBL neuroscience data.
-   **FlatIron**: storage of bulky raw data time-series. The Alyx database points to the files on this server. It is accessible through HTTP, FTP and Globus.
-   **ONE**: set of normalized functions to access the IBL data. It queries the Alyx database and downloads data files from the FLatIron.

## Getting started:
Ipython notebook  [here](./_static/one_demo.html)