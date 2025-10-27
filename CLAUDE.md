# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Toad is a Python package for scorecard model development, particularly focused on credit scoring and risk modeling. It provides tools for the entire modeling pipeline: EDA, feature engineering, feature selection, binning, model validation, and scorecard transformation.

**Key characteristics:**
- Python 3.9+ required
- **Uses Rust (via PyO3/maturin) for performance-critical binning/merging algorithms**
- Scikit-learn compatible transformers and estimators
- Supports both Chinese and English documentation

## Build and Development Commands

### Setup and Installation

**Prerequisites:**
- Rust toolchain (install with: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Python 3.9+

```bash
make install          # Install package in development mode (builds Rust extensions first)
make build           # Build Rust extensions using maturin
make build_deps      # Install build dependencies (including maturin)
```

**Alternative build methods:**
```bash
# Using maturin directly (requires virtualenv)
maturin develop --release

# Or build wheel and install
maturin build --release
pip install target/wheels/toad-*.whl
```

### Testing
```bash
make test                        # Run all tests with pytest
make test toad/xxxx_test.py     # Run specific test file
```

### Building Distribution
```bash
make dist            # Build source distribution (.tar.gz)
make dist_wheel      # Build wheel distribution
make clean           # Remove build artifacts, .pyc, .c, .so files
```

### Documentation
```bash
make docs            # Build Sphinx documentation (outputs to docs/build/)
```

## Rust Components

**Critical:** The `toad.merge.ChiMerge` function is implemented in Rust for performance.

- Source files: `src/lib.rs`, `src/c_utils.rs`, `src/merge.rs`
- After modifying Rust files, run `maturin develop --release` or `make build` to recompile
- Rust extension module name: `toad_rust` (defined in Cargo.toml)
- The Rust extension provides:
  - `toad_rust.merge.chi_merge` - High-performance Chi-square based binning
  - `toad_rust.c_utils` - Utility functions for array operations
- Python binning algorithms (DTMerge, StepMerge, QuantileMerge, KMeansMerge) remain in pure Python
- Import in Python: `import toad_rust` (module is installed to site-packages by maturin)

## Architecture and Core Components

### Main Modules (toad/)

**stats.py** - Statistical metrics for feature evaluation:
- `IV()` - Information Value calculation
- `WOE()` - Weight of Evidence transformation
- `VIF()` - Variance Inflation Factor for multicollinearity
- `quality()` - Comprehensive feature quality metrics
- `gini()`, `entropy()` - Distribution measures

**transform.py** - Feature transformation pipeline:
- `Combiner` - Binning transformer (supports chi-merge, tree-based, quantile methods)
- `WOETransformer` - WOE encoding for categorical/binned features
- Both are sklearn-compatible transformers (fit/transform interface)
- Inherit from `RulesMixin` and `BinsMixin` for rule persistence

**selection.py** - Feature selection:
- `select()` - Multi-criteria feature filtering (IV, correlation, missing rate, etc.)
- `stepwise()` - Stepwise regression with AIC/BIC criteria
- `StatsModel` - Wrapper for sklearn estimators with statistical tests

**scorecard.py** - Scorecard generation:
- `ScoreCard` - End-to-end scorecard class
- Integrates Combiner → WOETransformer → LogisticRegression
- Converts log-odds to scores using PDO (Points to Double Odds) scaling
- `export()` method generates human-readable scorecard tables

**merge.py** - Binning algorithms:
- `ChiMerge()` - Chi-squared statistic based merging (Rust implementation)
- `DTMerge()` - Decision tree based binning (Python)
- `StepMerge()`, `QuantileMerge()`, `KMeansMerge()` - Alternative binning methods (Python)

**metrics.py** - Model evaluation:
- `KS()`, `KS_bucket()` - Kolmogorov-Smirnov statistic
- `AUC()`, `F1()`, `PSI()` - Standard ML metrics

**nn/** - Neural network utilities (optional dependency):
- PyTorch-based components
- Includes distributed training, LoRA, quantization support
- trainer/, zoo/, functional.py, module.py, loss.py

**utils/** - Shared utilities:
- `decorator.py` - Function decorators for pandas DataFrame handling
- `func.py` - Array manipulation, binning utilities
- `mixin.py` - `RulesMixin`, `BinsMixin` for rule serialization (save/load)
- `progress/` - Progress bar utilities

### Typical Workflow

1. **Feature Quality Analysis**: Use `toad.quality()` to calculate IV and other metrics
2. **Preliminary Selection**: Use `toad.selection.select()` to filter by IV, correlation, missing rate
3. **Binning**: Use `Combiner.fit()` with method='chi' or 'tree' for automatic binning
4. **WOE Encoding**: Use `WOETransformer.fit_transform()` to convert to WOE values
5. **Stepwise Selection**: Use `toad.selection.stepwise()` for final feature selection
6. **Scorecard Building**: Use `ScoreCard.fit()` to train and `export()` to generate card
7. **Validation**: Use `toad.metrics.KS_bucket()` for performance visualization

## Testing Strategy

- Tests colocated with source: `*_test.py` files in same directory as implementation
- Uses pytest framework
- Test files follow pattern: `toad/module_test.py` tests `toad/module.py`
- Run full test suite before commits to catch regressions

## Dependencies

**Core dependencies** (requirements.txt):
- pandas >= 1.5
- scikit-learn >= 0.21
- scipy, numpy (version depends on Python version)
- joblib, seaborn

**Build dependencies**:
- maturin >= 1.0 (for building Rust extensions)
- Rust toolchain (rustc, cargo)

**Optional dependencies**:
- requirements-nn.txt: PyTorch and torchvision for neural network components
- requirements-tools.txt: Additional tooling

**Install with optional dependencies:**
```bash
pip install .[nn]      # Install with neural network support
pip install .[all]     # Install all optional dependencies
```

## Platform-Specific Notes

- **macOS M1/M2**: Special handling in requirements for numpy < 2.0
- **Cython version**: Different constraints for Python < 3.10 vs >= 3.10
- CI/CD runs on Linux, macOS, Windows (see .github/workflows/)

## Release Process

- Releases triggered by pushing git tags
- GitHub Actions workflows:
  - `linux.yml` - Tests on multiple Python versions, builds manylinux wheels using maturin-action
  - `macos.yml`, `windows.yml` - Platform-specific Rust builds
  - `release.yml` - Creates GitHub release from tag
- PyPI publishing automated via `gh-action-pypi-publish`
- All workflows include Rust toolchain setup for building native extensions

## Using uv for Development

This project supports `uv` for fast dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and build Rust extensions
uv sync

# Run tests
uv run pytest toad/

# Install in development mode
uv pip install -e . --no-build-isolation

# Build wheel
uv run maturin build --release
```
