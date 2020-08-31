#!/usr/bin/env python

import os
import unittest
import lasso
import numpy as np
from setuptools import setup, find_packages

def collect_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.', pattern='test_*.py')
    return test_suite

def main():

    setup(name='lasso-python',
          version='1.3.1',
          description='A next-generation CAE Python Library.',
          author='Lasso GmbH',
          author_email='lasso@lasso.de',
          url='https://github.com/lasso-gmbh/lasso-python',
          license="BSD-3",
          install_requires=[
              'numpy',
              'plotly',
              'matplotlib',
              'grpcio',
              'enum34',
              'protobuf',
              'flask',
              'h5py',
              'psutil',
          ],
          packages=find_packages(),
          package_data={
              '': ['*.png', '*.html', '*.js', '*.so', '*.dll', '*.txt','*.css'],
          },
          test_suite='setup.collect_tests',
          zip_safe=False,
          classifiers=[
            'Development Status :: 4 - Beta',
            'Topic :: Scientific/Engineering',
            'Intended Audience :: Science/Research',
            'Topic :: Utilities',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8']
          )


if __name__ == "__main__":
    main()
    