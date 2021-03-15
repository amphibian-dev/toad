import os
import numpy as np
from setuptools import setup, find_packages, Extension


NAME = 'toad'


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
VERSION_FILE = os.path.join(CURRENT_PATH, NAME, 'version.py')

def get_version():
    ns = {}
    with open(VERSION_FILE) as f:
        exec(f.read(), ns)
    return ns['__version__']


def get_ext_modules():
    from Cython.Build import cythonize

    extensions = [
        Extension('toad.c_utils', sources = ['toad/c_utils.pyx'], include_dirs = [np.get_include()]),
        Extension('toad.merge', sources = ['toad/merge.pyx'], include_dirs = [np.get_include()]),
    ]

    return cythonize(extensions)


def get_requirements(stage = None):
    file_name = 'requirements'

    if stage is not None:
        file_name = f"{file_name}-{stage}"
    
    requirements = []
    with open(f"{file_name}.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('-'):
                continue
            
            requirements.append(line)
    
    return requirements


setup(
    name = NAME,
    version = get_version(),
    description = 'Toad is dedicated to facilitating model development process, especially for a scorecard.',
    long_description = open('README.md', encoding = 'utf-8').read(),
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/amphibian-dev/toad',
    author = 'ESC Team',
    author_email = 'secbone@gmail.com',
    packages = find_packages(exclude = ['tests']),
    include_dirs = [np.get_include()],
    ext_modules = get_ext_modules(),
    include_package_data = True,
    python_requires = '>=3.6',
    install_requires = get_requirements(),
    extras_require = {
        'nn': get_requirements('nn')
    },
    tests_require = get_requirements('test'),
    license = 'MIT',
    classifiers = [
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points = {
        'console_scripts': [
            'toad = toad.cli:main',
        ],
    },
)
