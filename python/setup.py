from setuptools import setup

with open("../README.md", 'r') as f:
    long_description = f.read()

setup(
   name='ibllib',
   version='0.1.2',
   description='IBL libraries',
   license="MIT",
   long_description=long_description,
   author='IBL Staff',
   url="https://www.internationalbrainlab.com/",
   packages=['ibllib'],  #same as name
   install_requires=['dataclasses', 'matplotlib', 'numpy', 'pandas',
                     'requests'], #external packages as dependencies
   scripts=[]
)