import os
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


NAME = 'toad'


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
VERSION_FILE = os.path.join(CURRENT_PATH, NAME, 'version.py')

def get_version():
    ns = {}
    with open(VERSION_FILE) as f:
        exec(f.read(), ns)
    return ns['__version__']


extensions = [
    Extension('toad.c_utils', sources = ['toad/c_utils.pyx'], include_dirs = [np.get_include()]),
    Extension('toad.merge', sources = ['toad/merge.pyx'], include_dirs = [np.get_include()]),
]

setup(
    name = NAME,
    version = get_version(),
    description = 'python utils for detect data',
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/amphibian-dev/toad',
    author = 'ESC Team',
    author_email = 'secbone@gmail.com',
    packages = find_packages(exclude = ['tests']),
    include_dirs = [np.get_include()],
    ext_modules = cythonize(extensions),
    include_package_data = True,
    python_requires = '>=3.5',
    setup_requires = [
        'setuptools',
        'Cython',
    ],
    install_requires = [
        'numpy >= 1.15',
        'pandas',
        'scipy',
        'scikit-learn',
        'statsmodels',
        'seaborn',
    ],
    tests_require=[
        'pytest'
    ],
    license = 'MIT',
    classifiers = [
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points = {
        'console_scripts': [
            'toad = toad.cli:main',
        ],
    },
)
