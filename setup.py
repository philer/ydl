#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

from ydl import (__version__, __license__, __author__, __email__)

setup(name='ydl',
      description='An interactive CLI for youtube-dl',
      version=__version__,
      license=__license__,
      author=__author__,
      author_email=__email__,
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.8',
          'Operating System :: POSIX :: Linux',
          'Topic :: Desktop Environment',
          'Topic :: Utilities',
          'Natural Language :: English',
      ],
      packages=['ydl'],
      install_requires=[
          'docopt',
          'pyperclip',
          'requests',
          'urwid',
          'youtube_dl',
      ],
      entry_points={'console_scripts': ['ydl = ydl.__main__:main']})
