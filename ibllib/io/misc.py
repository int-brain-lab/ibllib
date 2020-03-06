from pathlib import Path


# Remove empty folders
def delete_empty_folders(path, rglob_pattern='*', dry=True, recursive=False):
    """delete_empty_folders Will delete empty folders inside path, if recursive is set to True
    will delete all empty folders recusively untill all folders have a file inside
    recursive is ignored if dry==True

    :param path: path to check for empty folders
    :type path: str or pathlib.Path
    :param rglob_pattern: filter on folder names, defaults to '*'
    :type rglob_pattern: str, optional
    :param dry: dry run will simulate the action, defaults to True
    :type dry: bool, optional
    :param recursive: whether to recurse after the last level of empty folders
                      is deleted, defaults to False
    :type recursive: bool, optional
    :return: [description]
    :rtype: [type]
    """
    path = Path(path)
    all_dirs = {p for p in path.rglob(rglob_pattern) if p.is_dir()}
    empty_dirs = {p for p in all_dirs if not list(p.glob('*'))}
    print(f'Empty folders: {len(empty_dirs)}')
    if dry:
        print('Empty folder names:', empty_dirs)
    elif not dry:
        for d in empty_dirs:
            print('Deleting empty folder: ', d)
            d.rmdir()
        print(f'Deleted folders: {len(empty_dirs)}\n')
        if recursive:
            return delete_empty_folders(path, rglob_pattern=rglob_pattern, dry=dry)


if __name__ == "__main__":
    # Setup
    dirs = [
        Path('/home/nico/test'),
        Path('/home/nico/test/test_empty_parent'),
        Path('/home/nico/test/test_empty_parent/test_empty'),
        Path('/home/nico/test/test_empty'),
        Path('/home/nico/test/test_full'),
    ]
    file = Path('/home/nico/test/test_full/file.txt')

    _ = [x.mkdir() for x in dirs]
    file.touch()
    # Run
    {x: x.exists() for x in dirs}
    delete_empty_folders(dirs[0], dry=False, recursive=True)
    {x: x.exists() for x in dirs}
    # Cleanup
    file.unlink()
    Path('/home/nico/test/test_full').rmdir()
    Path('/home/nico/test').rmdir()
