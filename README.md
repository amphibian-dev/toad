<div align="center">
    <img src="images/toadlogo.png" width="256px" />
</div>

# TOAD


[![PyPi version][pypi-image]][pypi-url]
[![Python version][python-image]][pypi-url]
[![Build Status][travis-image]][travis-url]


ESC Team's data-detector for credit risk

## Install


via pip

```bash
pip install toad
```

via source code

```bash
python setup.py install
```

## Usage

```python
import toad


data = pd.read_csv('test.csv')

toad.detect(data)

toad.quality(data, target = 'TARGET', iv_only = True)

toad.IV(feature, target, method = 'dt', min_samples = 0.1)
```

## Documents

A simple API [docs](docs/API.rst)


[pypi-image]: https://img.shields.io/pypi/v/toad.svg?style=flat-square
[pypi-url]: https://pypi.org/project/toad/
[python-image]: https://img.shields.io/pypi/pyversions/toad.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/Secbone/toad/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/Secbone/toad
