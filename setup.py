from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
# from distutils.core import setup


extensions = [
    Extension('detector.merge', ['detector/merge.pyx']),
    Extension('detector.utils', ['detector/utils.pyx']),
]

setup(
    name='detector',
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
