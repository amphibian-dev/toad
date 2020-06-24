# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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


[unreleased]: https://github.com/amphibian-dev/toad/compare/0.0.60...HEAD
[0.0.60]: https://github.com/amphibian-dev/toad/compare/0.0.59...0.0.60
[0.0.59]: https://github.com/amphibian-dev/toad/compare/0.0.58...0.0.59
