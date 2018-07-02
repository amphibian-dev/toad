from setuptools import setup, find_packages
# from distutils.core import setup

setup(
    name='detector',
    version='0.0.1',
    description='python utils for detect data',
    author='Secbone',
    author_email='secbone@gmail.com',
    packages = find_packages(exclude = ['tests']),
    install_requires = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
    ]
)
