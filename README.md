<div align="center">
    <img src="https://raw.githubusercontent.com/amphibian-dev/toad/master/images/toadlogo.png" width="256px" />
</div>

# TOAD


[![PyPi version][pypi-image]][pypi-url]
[![Python version][python-image]][docs-url]
[![Build Status][actions-image]][actions-url]
[![Downloads Status][downloads-image]][docs-url]


Toad is dedicated to facilitating model development process, especially for a scorecard. It provides intuitive functions of the entire process, from EDA, feature engineering and selection etc. to results validation and scorecard transformation. Its key functionality streamlines the most critical and time-consuming process such as feature selection and fine binning.

## Install
 
Pip

```bash
pip install toad
```

Conda

```bash
conda install toad --channel conda-forge
```

Source code

```bash
python setup.py install
```

## Upgrade

Pip

```bash
pip install -U toad
```

Conda

```bash
conda install -U toad --channel conda-forge
```

## Key features

- Simple IV calculation for all

```python
toad.quality(data,'target',iv_only=True)
```

- Optimised stepwise feature selection algorithm, and selection by criteria

```python
selected_data = toad.selection.select(data,target = 'target', empty = 0.5, iv = 0.02, corr = 0.7, return_drop=True, exclude=['ID','month'])

final_data = toad.selection.stepwise(data_woe,target = 'target', estimator='ols', direction = 'both', criterion = 'aic', exclude = to_drop)
```

- Reliable fine binning with visualisation 

```python
# Chi-squared fine binning
c = toad.transform.Combiner()
c.fit(data_selected.drop(to_drop, axis=1), y = 'target', method = 'chi', min_samples = 0.05) 
print(c.export())

# Visualisation to check binning results 
col = 'feature_name'
bin_plot(c.transform(data_selected[[col,'target']], labels=True), x=col, target='target')
```

- Intuitive model results presentation

```python
toad.metrics.KS_bucket(pred_proba, final_data['target'], bucket=10, method = 'quantile')
```

- One-click scorecard transformation 

```python
card = toad.ScoreCard(
    combiner = c,
    transer = transer,
    class_weight = 'balanced',
    C=0.1,
    base_score = 600,
    base_odds = 35 ,
    pdo = 60,
    rate = 2
)

card.fit(final_data[col], final_data['target'])
print(card.export())
```

## Documents 

- [Tutorial](https://toad.readthedocs.io/en/latest/tutorial.html)

- [中文指引](https://toad.readthedocs.io/en/latest/tutorial_chinese.html)

- [docs][docs-url]

## Community
We welcome public feedback and new PRs. We hold a WeChat group for questions and suggestions. 


## Dedicated by **The ESC Team**

[pypi-image]: https://img.shields.io/pypi/v/toad.svg?style=flat-square
[pypi-url]: https://pypi.org/project/toad/
[python-image]: https://img.shields.io/pypi/pyversions/toad.svg?style=flat-square
[actions-image]: https://img.shields.io/github/workflow/status/amphibian-dev/toad/Release?style=flat-square
[actions-url]: https://github.com/amphibian-dev/toad/actions
[downloads-image]: https://img.shields.io/pypi/dm/toad?style=flat-square
[docs-url]: https://toad.readthedocs.io/
