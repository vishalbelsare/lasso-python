#!/usr/bin/env python

import os
import unittest
from setuptools import setup, find_packages

def collect_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.', pattern='test_*.py')
    return test_suite

def get_requirements():
    with open("requirements.txt", "r") as fp:
        requirements = [line.strip() for line in fp if line.strip()]
        print(requirements)
        return requirements

def main():

    setup(name='lasso-python',
          version='1.5.2post1',
          description='A next-generation CAE Python Library.',
          author='Lasso GmbH',
          author_email='lasso@lasso.de',
          url='https://github.com/lasso-gmbh/lasso-python',
          license="BSD-3",
          packages=find_packages(),
          package_data={
              '': ['*.png', '*.html', '*.js', '*.so', '*.so.5', '*.dll', '*.txt','*.css'],
          },
          test_suite='setup.collect_tests',
          install_requires=get_requirements(),
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
    