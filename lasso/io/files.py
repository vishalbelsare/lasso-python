
import os
import glob
import unittest


def collect_files(dirpath, patterns, recursive=False):
    ''' Collect files from directories

    Parameters
    ----------
    dirpath : str or list(str)
        path to one or multiple directories to search through
    patterns : str or list(str)
        patterns to search for
    recursive : bool
        whether to also search subdirs

    Returns
    -------
    found_files : list(str) or list(list(str))
        returns the list of files found for every pattern specified

    Examples
    --------
        >>> png_images, jpeg_images = collect_files('./folder', ['*.png', '*.jpeg'])
    '''

    if not isinstance(dirpath, (list, tuple)):
        dirpath = [dirpath]
    if not isinstance(patterns, (list, tuple)):
        patterns = [patterns]

    found_files = []
    for pattern in patterns:

        files_with_pattern = []
        for current_dir in dirpath:
            # files in root dir
            files_with_pattern += glob.glob(
                os.path.join(current_dir, pattern))
            # subfolders
            if recursive:
                files_with_pattern += glob.glob(
                    os.path.join(current_dir, '**', pattern))

        found_files.append(sorted(files_with_pattern))

    if len(found_files) == 1:
        return found_files[0]
    else:
        return found_files
