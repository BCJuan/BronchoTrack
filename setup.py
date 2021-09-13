#!/usr/bin/env python
# https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/

from setuptools import setup, find_packages
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='BronchoTrack',
    version='0.1',
    description='Bronchoscopy tracking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Borrego-Carazo, Juan',
    author_email='juan.borrego@uab.cat',
    url=None,
    packages=find_packages(include=['exampleproject', 'exampleproject.*']),
    extras_require={
    'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    },
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    package_data={'DepthResolution': ['data/*.pkl']}
)
