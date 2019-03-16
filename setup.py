#!/usr/bin/env python
from setuptools import setup
import sys
import os

version = '0.0.1'

setup(name = 'gazedata',
      version = version,
      description = 'Module for gaze-data handling',
      long_description = """
This is a Python module to handle gaze data (eyetracking gata).
DISCLAIMER: This is a beta version. No warranty.
""",
      classifiers = [
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
      ],
      keywords = 'Tobii, Eye tracking, eyetracking, gaze',
      author = 'Kiri Kuroda',
      url = 'https://github.com/kirikuroda/gazedata/',
      licence = 'GNU GPL',
      packages = ['gazedata'],
      )