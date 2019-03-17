#!/usr/bin/env python
from setuptools import setup
import sys
import os

version = '0.0.2'

setup(name = 'gazedata',
      version = version,
      description = 'Module for pre-processing of gaze data',
      long_description = """
This is a Python module to pre-process gaze data (eyetracking gata).
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
      keywords = 'Tobii, eye-tracking, eyetracking, gaze, fixation',
      author = 'Kiri Kuroda',
      url = 'https://github.com/kirikuroda/ppgaze/',
      licence = 'GNU GPL',
      packages = ['ppgaze'],
      )