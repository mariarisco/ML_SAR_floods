# AI for flood detection

As a result of climate change, extreme hydrometeorological events are becoming increasingly frequent. Flood rapid mapping products play an important role in informing flood emergency response and management. These maps are generated quickly from remote sensing data during or after an event to show the extent of the flooding. They provide important information for emergency response and damage assessment. 

[IEEE Data fusion contest 2024] (https://www.grss-ieee.org/community/technical-committees/2024-ieee-grss-data-fusion-contest/) 

# Major Datasets Used

Multi-source geospatial datasets provided by the [Institute of electrical and Electronics Engineers](https://www.ieee.org/) withing the framework of the 2024 IEEE GRSS Data Fusion Contest - Flood Rapid Mapping. 

Public access to data: [IEEE DataPort](https://ieee-dataport.org/competitions/2024-ieee-grss-data-fusion-contest-flood-rapid-mapping)

Dataset structure: 

![Data structure](https://github.com/mariarisco/ML_SAR_floods/blob/main/src/img/Data_structure.png)

- Copernicus/Sentinel-1: C-band synthetic aperture radar, 10m resolution (VV & VH backscattering)
- [Copernicus DEM](https://dataspace-copernicus-eu.translate.goog/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=sc) (30 m): it is a Digital Surface Model (DSM) that represents the surface of the Earth including buildings, infrastructure and vegetation
- [MERIT](https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/) (90 m): it is a digital terrain model widely used in the hydrology scientific community
- [Global Surface Water Occurrence](https://global-surface-water.appspot.com/): maps the location and temporal distribution of water surfaces at the global scale over the past 32 years and provides statistics on the extent and change of those water surfaces
- [ESA WorldCover](https://esa-worldcover.org/en/data-access): global land cover product at 10 m resolution
- Labeled training data: Flood extent labeled by the Copernicus Emergency Management Service

# Technical solution

![Workflow1](https://github.com/mariarisco/ML_SAR_floods/blob/main/src/img/Workflow1.png)

![Workflow2](https://github.com/mariarisco/ML_SAR_floods/blob/main/src/img/Workflow2.png)

# Repo structure

```bash
ML_SAR_floods/
│── src/                    # Main source code directory
│   │── data/               # Contains a sample of the images datasets for training and testing
│   │── models/             # Stores machine learning / deep learning models
│   │── notebooks/          # Draft notebooks
│   │── result_notebooks/   # Final notebooks for model development (Machine Learning and MLP models)
│   │── utils/              # Utility scripts for shared functionality
│── .gitignore              # Specifies files and folders to be ignored by Git
│── environment.yml         # Conda environment file with dependencies
│── README.md               # Project documentation and setup instructions
```
