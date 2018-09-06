from setuptools import setup, find_packages, Extension


setup(
    name = 'toad',
    version = '0.0.7',
    description = 'python utils for detect data',
    long_description = open('README.rst').read(),
    # long_description_content_type = 'text/markdown',
    url = 'https://github.com/Secbone/toad',
    author = 'ESC Team',
    author_email = 'secbone@gmail.com',
    packages = find_packages(exclude = ['tests']),
    python_requires = '>=3.5',
    setup_requires = [
        'setuptools',
    ],
    install_requires = [
        'numpy >= 1.15',
        'pandas',
        'scipy',
        'sklearn',
    ],
    license = 'MIT',
    classifiers = [
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
