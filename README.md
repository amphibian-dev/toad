# TOAD

[![Pypi version][pypi-image]][pypi-url]
[![Build Status][travis-image]][travis-url]
[![Version][version-image]][version-url]

ESC Team's data-detector for finance

## Install

```
pip install toad
```
or
```
make install
```
or
```
python setup.py install
```

## Usage

```
import toad


data = pd.read_csv('test.csv')

toad.detect(data)

toad.quality(data, target = 'TARGET', iv_only = True)

toad.IV(feature, target, method = 'dt', min_samples = 0.1)
```

## Documents

working...

[pypi-image]: https://img.shields.io/pypi/v/toad.svg?style=flat-square
[pypi-url]: https://pypi.org/project/toad/
[version-image]: https://img.shields.io/pypi/pyversions/toad.svg?style=flat-square
[version-url]: https://pypi.org/project/toad/
[travis-image]: https://img.shields.io/travis/Secbone/toad/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/Secbone/toad
