<div align="center">
    <img src="https://raw.githubusercontent.com/amphibian-dev/toad/master/images/toadlogo.png" width="256px" />
</div>

# TOAD


[![PyPi version][pypi-image]][pypi-url]
[![Python version][python-image]][docs-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Status][downloads-image]][docs-url]


**Toad is developed to facilitate the model development work - especially credit risk scorecard.**

It provides intuitive tools for

- data handling
- feature selection and WOE binning
- model selection
- results evaluation and model validation
- scorecard transformation.

A [basic tutorial](https://toad.readthedocs.io/en/latest/tutorial.html) is provided on readthedocs.

[中文指引](https://toad.readthedocs.io/en/dev/tutorial_chinese.html)

By the **ESC Team**

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

A simple API [docs][docs-url]


[pypi-image]: https://img.shields.io/pypi/v/toad.svg?style=flat-square
[pypi-url]: https://pypi.org/project/toad/
[python-image]: https://img.shields.io/pypi/pyversions/toad.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/amphibian-dev/toad/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/amphibian-dev/toad
[downloads-image]: https://img.shields.io/pypi/dm/toad?style=flat-square
[docs-url]: https://toad.readthedocs.io/
