# ibllib gitflow and git commands for releasing
To use `git flow` install it by: `sudo apt install git-flow`  
## Create a release branch 
    git flow release start 0.4.35    |    git checkout -b release/0.4.35 develop

## Change and commit locally:
* ### Bump up version in setup.py
    ```python
    setup(
        name='ibllib',
        version='0.4.34',   -->   version='0.4.35'
        ...
    ```
* ### Flakify
* ### Docs if needed
* ### Make sure tests pass
        
**Committ changes normally to current release/0.4.35 branch**  
Normal push and pull for sharing an unfinished release branch apply

## Finalize a release branch
    git flow release finish 0.4.35  |    git checkout master
                                    |    git merge --no-ff release/0.4.35
                                    |    git tag -a 0.4.35
                                    |    git checkout develop
                                    |    git merge --no-ff release/0.4.35
                                    |    git branch -d release/0.4.35

## Push to repo
    git push origin master
    git push origin develop
    git push origin --tags
.  
.  
.  
.  
# ibllib deploy to PYPI
## Build
**First remove anything in ibllib/python/dist/***  
Then build
```shell
rm -R dist
python setup.py sdist bdist_wheel
```

## Test upload
```shell
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

## Upload
```shell
twine upload dist/*
```

## Install
As lib  
Activate environment and upgrade
```shell
conda activate iblenv
pip install ibllib --upgrade 
```
As code installed with `pip install -e .`  
```shell
cd /my/ibllib/repo/path
git reset --hard
git pull
```