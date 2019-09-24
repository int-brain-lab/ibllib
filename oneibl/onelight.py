# -*- coding: utf-8 -*-

"""ONE light."""


# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import os
from pathlib import Path
import re
import sys


# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return Bunch(super(Bunch, self).copy())


# -------------------------------------------------------------------------------------------------
# Base provider
# -------------------------------------------------------------------------------------------------

PATH_PATTERN = r'^{lab}/Subjects/{subject}/{date}/{session}/alf/{filename}$'
# SEARCH_TERMS = ('lab', 'subjects', 'date_range', 'number', 'dataset_types')


def _pattern_to_regex():
    return re.compile(re.sub(r'\{(\w+)\}', r'(?P<\1>[a-zA-Z0-9_\-\.]+)', PATH_PATTERN))


def _make_filter_regex(**kwargs):
    pattern = PATH_PATTERN

    for term in ('lab', 'subject', 'date', 'session', 'filename'):
        value = kwargs.get(term, None)
        if value:
            if not isinstance(value, str):
                value = '|'.join(value)
            if '*' in value:
                value = value.replace('*', r'[a-zA-Z0-9_\-\.]+')
            value = value.replace('.', r'\.')
            pattern = pattern.replace(r'{%s}' % term, '(?P<%s>%s)' % (term, value))
        else:
            pattern = pattern.replace(r'{%s}' % term, r'(?P<%s>[a-zA-Z0-9_\-\.]+)' % term)
    return re.compile(pattern)


class BaseONE:
    def get_config(self):
        pass

    def iter_files(self):
        """

        How to generate a complete listing of files:

            find /path/to/root -type f -printf '%P\n' > /path/to/root/listing.txt

        """
        with open('listing.txt', 'r') as f:
            for line in f:
                yield line

    def search(self, dataset_types, **kwargs):
        filter_kwargs = {
            'filename': dataset_types,
            'lab': kwargs.get('lab', None),
            'session': kwargs.get('number', None),
            'subject': kwargs.get('subject', None),
            'date': kwargs.get('date', None),
        }
        pattern = _make_filter_regex(**filter_kwargs)
        for path in self.iter_files():
            if pattern.match(path):
                yield path

    def download_file(self, dset_id):
        pass


# -------------------------------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------------------------------

def set_download_dir(path):
    pass


def search_terms():
    pass


def search(dataset_types, **kwargs):
    pass


def load_object(session, *dataset_types):
    pass


def load_dataset(session, dataset_type):
    pass
