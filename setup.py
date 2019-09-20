from setuptools import setup, find_packages
import sys
from pathlib import Path

CURRENT_DIRECTORY = Path(__file__).parent.absolute()

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of idtrackerai requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

with open("README.md", 'r') as f:
    long_description = f.read()

"""
reasons not to read from the requirements.txt file here:
a) attempting to read from the requirements.txt file here is incompatible with installing from
the built tarball as in  `pip install ibllib-x.x.x.tar.gz`
b) requirements.txt may have a broader environment scope than the hard requirements listed here
c) git http dependencies are not allowed on a PyPi package
"""

install_requires = [
    'colorlog>=4.0.2',
    'dataclasses>=0.6',
    'globus-sdk>=1.7.1',
    'matplotlib>=3.0.3',
    'numpy>=1.16.4',
    'pandas>=0.24.2',
    'requests>=2.22.0',
    'scipy>=1.3.0',
    'seaborn>=0.9.0',
    'flake8>=3.7.8',
    'opencv-python>=4.1.1.26',
    'phylib',
]

setup(
    name='ibllib',
    version='1.1.9',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    description='IBL libraries',
    license="MIT",
    long_description=long_description,
    author='IBL Staff',
    url="https://www.internationalbrainlab.com/",
    packages=find_packages(exclude=['scratch']),  # same as name
    # external packages as dependencies
    install_requires=install_requires,
    scripts={}
)
