from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
# from distutils.core import setup


extensions = [
    Extension('toad.merge', ['toad/merge.pyx']),
    Extension('toad.utils', ['toad/utils.pyx']),
]

setup(
    name='toad',
    version='0.0.3',
    description='python utils for detect data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Secbone',
    author_email='secbone@gmail.com',
    packages = find_packages(exclude = ['tests']),
    ext_modules = cythonize(extensions),
    python_requires=">=3.5",
    install_requires = [
        'cython',
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
    ]
)
