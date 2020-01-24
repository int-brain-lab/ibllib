# Installing Brainbox

## Key Dependencies
Brainbox requires **Python 3.6 or higher**. Other dependencies can be installed using the `requirements.txt` or `environment.yml` files in included in brainbox using pip or conda, respectively.

## Brainbox requires ibllib
Brainbox is for the moment a submodule of ibllib, a suite of python code used by the International Brain Lab to support its experimental infrastructure and data analysis pipelines.

## Setting up a conda environment for ibllib
To install brainbox you must first install ibllib via the terminal. This process should be the same for Windows, OS X, and Linux.

```
conda create -n brainbox --python=python3.7
conda activate brainbox
```

## Clone ibllib into your machine
We will use git in the terminal to clone the GitHub repository for ibllib onto our machine. Windows does not come with git installed. (You can download it here.)[https://gitforwindows.org/]

