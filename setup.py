from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
# from distutils.core import setup


extensions = [
    Extension('toad.merge', ['toad/merge.pyx']),
    Extension('toad.utils', ['toad/utils.pyx']),
]

setup(
    name='toad',
    version='0.0.1',
    description='python utils for detect data',
    author='Secbone',
    author_email='secbone@gmail.com',
    packages = find_packages(exclude = ['tests']),
    ext_modules = cythonize(extensions),
    install_requires = [
        'cython',
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
    ]
)
