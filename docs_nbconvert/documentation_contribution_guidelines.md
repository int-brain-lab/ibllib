# Contributing to ibllib documentation

The documentation is built as a read the docs website at this address:
https://docs.internationalbrainlab.org/en/latest/

The website is generated partly using the markdown files in this `./docs` folders and partly from the docstrings in the source code.

Small contributions can be done directly through the github web-interface.
For larger contributions, it is recommended to push to the `feature/docs` branch that will then be merged into develop.

This is for 2 reasons:
1)  any contribution to the docstrings will be subject to linting/testing requirements of the code
2)  to build and preview the website locally before putting changes online     


## Contributing via Github
You can directly edit files whithin the `./docs` folder.

## Building the website locally
Before pushing the documentation to the website, it may be useful to preview the documentation on the local machine.

### Linux
```shell script
cd ./docs
rm -fR _build
make html
```
And them open the `./docs/_build/html/index.html` in a web browser.
