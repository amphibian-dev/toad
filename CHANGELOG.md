# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] - 2024-11-03

### Add
- Added wheel package supported for `py3.12`
- Added `figsize` param in `toad.plot.bin_plot` function

### Changed
- Update `pandas` version to `>=1.5`
- Python `3.7` `3.8` is no longer supported

## [0.1.3] - 2023-12-10

### Add
- Added `performance` in `toad.utils` for test code performance
- Added `pickletracer` in `toad.utils` for infer requirements in pickle object

### Fixed
- Fixed `Value Error` in `select` and `drop_corr` method when using `pandas >= 2.0.x`

## [0.1.2] - 2023-04-09

### Add
- Added `ks_plot` for KS plot, [#102](https://github.com/amphibian-dev/toad/issues/102) thanks @kevin-meng
- Added `xgb_loss` decorator for convert a normal loss function to a xgb supported loss function
- Added `binary_focal_loss` function in `nn.functional`
- Added `event` module in `nn.trainer`, and changed `trainer` mode to event-based
- Added wheel package supported for `py3.9`, `py3.10` and `py3.11`

### Changed
- Now you can pass arguments to `DecisionTreeClassifier` in `merge` or `Combiner` when use `method = dt`

### Fixed
- Fixed `groupby` rewrited in `preprocessing`
- Fixed the expired deprecations of numpy types in `1.24.0`

## [0.1.1] - 2022-08-14

### Add
- Added `Progress` for `pandas.apply` by using `pandas_enable` and `pandas_disable`
- Added `feature_bin_stats` for feature bins, [#91](https://github.com/amphibian-dev/toad/issues/91) thanks @kevin-meng

### Changed
- `countBlank` can use customize missing value, [#101](https://github.com/amphibian-dev/toad/issues/101) thanks @kevin-meng
- remove ref of `merge` in `__init__` file



## [0.1.0] - 2021-10-08

### Add

- Added `backward_rounds` for `nn.Trainer.train`
- Added `evalute` func in `nn.Module`
- Added `get_reason` func in `ScoreCard`, [#79](https://github.com/amphibian-dev/toad/issues/79) thanks @qianweishuo
- Added dict type input support for `ScoreCard.predict` and `Combiner.transform`, [#79](https://github.com/amphibian-dev/toad/issues/79) thanks @qianweishuo
- Added iterator support for `Progress`

### Changed

- Change `callback` and `earlystopping` to python decorator


## [0.0.65] - 2021-06-30

### Breaking Changes

- Add new `lift` value and rename the old `lift` value to `cum_lift` in `KS_Bucket`
- Move `nn.autoencoder` to `nn.zoo.autoencoder`

### Add

- Added `label_smoothing`, `focal_loss` function in `nn` module
- Added some features in `nn.trainer`
- Added default `early_stopping` for `nn.Trainer`

### Changed

- Update `numpy` version to `>=1.20`
- Python `3.6` is no longer supported

### Fixed

- Fixed combiner error after `ScoreCard` reload. [#67](https://github.com/amphibian-dev/toad/issues/67)


## [0.0.64] - 2021-03-22

### Added

- Added `callback` param in `fit` method for `nn`
- Added `Trainer` and `EarlyStopping` in `nn.trainer` module

### Changed

- Use mean of loss in `nn.Module.fit` instead of the latest loss value
- Set default rotation for x tick labels

### Fixed

- Fixed dependence version of `numpy`
- Fixed `DistModule` module
- Fixed `ScoreCard` representation error

## [0.0.62] - 2021-02-19

### Added

- `save` and `load` method for nn module
- Added `lift` value in `KS_bucket` function
- Added checking duplicate keys in `Transformer`

### Changed

- `quality` method support `indicators`

### Fixed

- Fixed tadpole warning of legend. [#52](https://github.com/amphibian-dev/toad/issues/52)
- Fixed tadpole `title` and `x/y label` display for `UTF8` 
- Fixed default rule in RuleMixin.
- Fixed loss function of VAE model.
- Fixed `decimal` argument in `ScoreCard.export` function

### Enhancements

- Reduce memory usage when using `select` function

## [0.0.61] - 2020-06-24

### Added

- Support for calculating IV for each groups in a feature. [#25](https://github.com/amphibian-dev/toad/issues/25)
- Add `cpu_cores` for `quality` function
- Add `predict_proba` for `ScoreCard`
- Impute module
- NN module

### Changed

- The y axis of `badrate_plot` is starting with `0` now. [#23](https://github.com/amphibian-dev/toad/issues/23)
- `KS` is implemented using `ks2samp` instead

### Fixed

- Fixed `Preprocess` bugs

### Docs

- Add references for `Chi-Merge`, `Stepwise Regression`, `Scorecard Transformation`

## [0.0.60] - 2020-04-20

### Added

- Preprocess module.
- Annotation format for bin plot.
- KS bucket support split pointers as bucket. [#22](https://github.com/amphibian-dev/toad/issues/22)

### Changed

- Format_bins support ellipsis.
- Reverse cumulative columns in KS bucket
- Use correct order of score for auc and roc plot. [#21](https://github.com/amphibian-dev/toad/issues/21)

### Fixed

- Fixed number type of x axis of badrate plot. [#20](https://github.com/amphibian-dev/toad/issues/20)
- Fixed negative ks value in `KS_bucket`.

## [0.0.59] - 2020-02-07

### Added

- Combiner support empty separate.
- Confusion matrix function in metrics.
- support python 3.8.

### Changed

- Transform support y as string type.
- VIF independent statsmodels.


[Unreleased]: https://github.com/amphibian-dev/toad/compare/0.1.0...HEAD
[0.1.0]: https://github.com/amphibian-dev/toad/compare/0.0.65...0.1.0
[0.0.65]: https://github.com/amphibian-dev/toad/compare/0.0.64...0.0.65
[0.0.64]: https://github.com/amphibian-dev/toad/compare/0.0.62...0.0.64
[0.0.62]: https://github.com/amphibian-dev/toad/compare/0.0.61...0.0.62
[0.0.61]: https://github.com/amphibian-dev/toad/compare/0.0.60...0.0.61
[0.0.60]: https://github.com/amphibian-dev/toad/compare/0.0.59...0.0.60
[0.0.59]: https://github.com/amphibian-dev/toad/compare/0.0.58...0.0.59
