from setuptools import setup, find_packages

with open("../README.md", 'r') as f:
    long_description = f.read()

setup(
    name='ibllib',
    version='0.4.6',
    description='IBL libraries',
    license="MIT",
    long_description=long_description,
    author='IBL Staff',
    url="https://www.internationalbrainlab.com/",
    packages=find_packages(),  # same as name
    install_requires=['dataclasses', 'matplotlib', 'numpy', 'pandas',
                      'requests', 'scipy', 'seaborn', 'globus_sdk',
                      'ciso8601'],  # external packages as dependencies
    scripts=[]
)
