# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Preprocess module.
- Annotation format for bin plot.
- KS bucket support split pointers as bucket

### Changed

- Format_bins support ellipsis.
- Reverse cumulative columns in KS bucket
- Use correct order of score for auc and roc plot. fixed #21

### Fixed

- Fix number type of x axis of badrate plot. fixed #20

## [0.0.59] - 2020-02-07

### Added

- Combiner support empty separate.
- Confusion matrix function in metrics.
- support python 3.8.

### Changed

- Transform support y as string type.
- VIF independent statsmodels.


[unreleased]: https://github.com/amphibian-dev/toad/compare/0.0.59...HEAD
[0.0.59]: https://github.com/amphibian-dev/toad/compare/0.0.58...0.0.59
