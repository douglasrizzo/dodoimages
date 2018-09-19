#!/usr/bin/env python3

from setuptools import setup, find_packages
from dodoimages import __version__
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
    name='dodoimages',
    version=__version__,
    description='Image Preprocessor',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Douglas De Rizzo Meneghetti',
    author_email='douglasrizzom@gmail.com',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'numpy', 'pillow', 'imutils', 'scikit-image', 'tqdm'
    ],
    license='GPLv3'
)