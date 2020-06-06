import os
from setuptools import setup, find_packages

setup(
  name = 'tollefTokenizer',
  packages = find_packages(),
  include_package_data=True,
  version = '0.1',
  description = 'nltk-based tokenizer',
  author = 'Tollef Jørgensen',
  author_email = 'tollefj@gmail.com',
  license = 'MIT License: http://opensource.org/licenses/MIT',
  url = '',  # TODO
  download_url = '',  # TODO
  install_requires = ['nltk'],
  keywords = ['nltk', 'tokenizer'],
  classifiers = ['Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License', 'Natural Language :: English',
                 'Programming Language :: Python :: 3.7', 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Information Analysis', 'Topic :: Text Processing :: Linguistic',
                 'Topic :: Text Processing :: General'],
)
