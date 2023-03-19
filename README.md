<div align="center">
    <img src="https://raw.githubusercontent.com/amphibian-dev/toad/master/images/toad_logo.png" width="350px" />
</div>

# TOAD


[![PyPi version][pypi-image]][pypi-url]
[![Python version][python-image]][docs-url]
[![Build Status][actions-image]][actions-url]
[![Downloads Status][downloads-image]][docs-url]


Toad is dedicated to facilitating model development process, especially for a scorecard. It provides intuitive functions of the entire process, from EDA, feature engineering and selection etc. to results validation and scorecard transformation. Its key functionality streamlines the most critical and time-consuming process such as feature selection and fine binning.

Toad 是专为工业界模型开发设计的Python工具包，特别针对评分卡的开发。Toad 的功能覆盖了建模全流程，从 EDA、特征工程、特征筛选 到 模型验证和评分卡转化。Toad 的主要功能极大简化了建模中最重要最费时的流程，即特征筛选和分箱。

## Install and Upgrade · 安装与升级
 
Pip

```bash
pip install toad # to install
pip install -U toad # to upgrade
```

Conda

```bash
conda install toad --channel conda-forge # to install
conda install -U toad --channel conda-forge # to upgrade
```

Source code

```bash
python setup.py install
```

## Key features · 主要功能

The following showcases some of the most popular features of toad, for more detailed demonstrations and user guidance, please refer to the tutorials.

以下部分简单介绍了toad最受欢迎的一些功能，具体的使用方法和使用教程，请详见文档部分。

- Simple IV calculation for all features · 一键算IV:

```python
toad.quality(data,'target',iv_only=True)
```

- Preliminary selection based on criteria · 根据特定条件的初步变量筛选; 
- and stepwise feature selection (with optimised algorithm) · 优化过的逐步回归:

```python
selected_data = toad.selection.select(data,target = 'target', empty = 0.5, iv = 0.02, corr = 0.7, return_drop=True, exclude=['ID','month'])

final_data = toad.selection.stepwise(data_woe,target = 'target', estimator='ols', direction = 'both', criterion = 'aic', exclude = to_drop)
```

- Reliable fine binning with visualisation · 分箱及可视化:

```python
# Chi-squared fine binning
c = toad.transform.Combiner()
c.fit(data_selected.drop(to_drop, axis=1), y = 'target', method = 'chi', min_samples = 0.05) 
print(c.export())

# Visualisation to check binning results 
col = 'feature_name'
bin_plot(c.transform(data_selected[[col,'target']], labels=True), x=col, target='target')
```

- Intuitive model results presentation · 模型结果展示:

```python
toad.metrics.KS_bucket(pred_proba, final_data['target'], bucket=10, method = 'quantile')
```

- One-click scorecard transformation · 评分卡转化:

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

## Documents · 文档

- [Tutorial](https://toad.readthedocs.io/en/latest/tutorial.html)

- [中文指引](https://toad.readthedocs.io/en/latest/tutorial_chinese.html)

- [docs][docs-url]

- [Contributing](CONTRIBUTING.md)

## Community · 社区
We welcome public feedback and new PRs. We hold a WeChat group for questions and suggestions. 

欢迎各位提PR，同时我们有toad使用交流的微信群，欢迎询问加群。

## Contributors

[![Contributors][contributor-image]][contributor-url]

------------

## Dedicated by **The ESC Team** 

[pypi-image]: https://img.shields.io/pypi/v/toad.svg?style=flat-square
[pypi-url]: https://pypi.org/project/toad/
[python-image]: https://img.shields.io/pypi/pyversions/toad.svg?style=flat-square
[actions-image]: https://img.shields.io/github/workflow/status/amphibian-dev/toad/Release?style=flat-square
[actions-url]: https://github.com/amphibian-dev/toad/actions
[downloads-image]: https://img.shields.io/pypi/dm/toad?style=flat-square
[docs-url]: https://toad.readthedocs.io/
[contributor-image]: https://contrib.rocks/image?repo=amphibian-dev/toad
[contributor-url]: https://github.com/amphibian-dev/toad/graphs/contributors
