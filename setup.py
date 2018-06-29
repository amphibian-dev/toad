from distutils.core import setup

setup(
    name='detector',
    version='0.0.1',
    description='python utils for detect data',
    author='Secbone',
    author_email='secbone@gmail.com',
    packages = ['detector'],
    requires = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
    ]
)
