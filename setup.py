#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line

ext_modules = [Extension("inner_paragraph_vector", ["inner_paragraph_vector.pyx"])]

setup(
  name = 'inner_paragraph_vector',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],         # <---- New line
  ext_modules = ext_modules
)