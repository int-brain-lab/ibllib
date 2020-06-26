# IBLLIB Python Developer installation guide

1. Install Git onto your machine

    -   **Linux** : Git comes with Linux.
    -   **Mac OS** : Download and install Git from here: https://git-scm.com/download/mac .
    -   **Windows** : Download and install Git from here: https://github.com/git-for-windows/git/releases/ .

2. Activate/install your target Python 3.6 or greater environment (latest always recommended). For example on Linux:

**Linux:**
```
cd ibllib/
virtualenv iblenv --python=python3.8
source ./venv/bin/activate
```

**Windows:**
```
conda create -n iblenv python=3.8
conda activate iblenv
```

**Mac OS:**
```
conda create -n iblenv python=3.8 anaconda
conda activate iblenv
```


3. Clone the ibllib repository and install ibllib in place:
```
git clone https://github.com/int-brain-lab/ibllib.git
cd ibllib
pip install -r requirements.txt
pip install -e .
```
    

4. Run tests
    -   In a shell terminal, in the ibllib folder with the proper environment activated:
        -   **Linux and Mac OS**: `source run_tests`
        -   **Windows**: `call run_tests.bat`
Console output should end with `OK`
