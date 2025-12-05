.PHONY: build test install uninstall test_deps dist_deps dist dist_wheel upload clean docs ensure-uv install-nn

SHELL = /bin/bash

# Always use uv
PYTHON = uv run python3
PIP = uv pip
UV_SYSTEM_FLAG = --system

# Check if uv is installed, if not install it
ensure-uv:
	@command -v uv >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$$HOME/.local/bin:$$PATH")

SPHINXOPTS =
SPHINXBUILD = sphinx-build
SPHINXPROJ = toad
DOCSDIR = docs
SOURCEDIR := $(DOCSDIR)/source
BUILDDIR := $(DOCSDIR)/build

install: ensure-uv build
	@echo "Installing with uv..."
	$(PIP) install $(UV_SYSTEM_FLAG) -e .

install-nn: ensure-uv build
	@echo "Installing with neural network support..."
	$(PIP) install $(UV_SYSTEM_FLAG) -e .[nn]

uninstall:
	cat files.txt | xargs rm -rf

test_deps: ensure-uv
	@echo "Installing test dependencies with uv..."
	$(PIP) install $(UV_SYSTEM_FLAG) -r requirements-test.txt

test: test_deps
	$(eval TARGET := $(filter-out $@, $(MAKECMDGOALS)))
	@if [ -z "$(TARGET)" ]; then \
		$(PYTHON) -m pytest -x toad --ignore=toad/nn; \
	else \
		$(PYTHON) -m pytest -s $(TARGET); \
	fi

test-nn: test_deps install-nn
	@echo "Running tests with neural network support..."
	$(PYTHON) -m pytest -x toad

build_deps: ensure-uv
	@echo "Installing build dependencies with uv..."
	$(PIP) install $(UV_SYSTEM_FLAG) -r requirements.txt
	$(PIP) install $(UV_SYSTEM_FLAG) maturin

build: build_deps
	@echo "Building with uv and maturin..."
	$(PYTHON) -m maturin build --release
	@echo "Extracting and installing toad_core module..."
	@rm -rf /tmp/toad_wheel_extract && mkdir -p /tmp/toad_wheel_extract
	@cd /tmp/toad_wheel_extract && unzip -q $(CURDIR)/target/wheels/toad-*.whl "toad_core/*"
	@rm -rf .venv/lib/python*/site-packages/toad_core
	@cp -r /tmp/toad_wheel_extract/toad_core .venv/lib/python*/site-packages/
	@echo "toad_core module installed successfully"

dist_deps: ensure-uv
	@echo "Installing distribution dependencies with uv..."
	$(PIP) install $(UV_SYSTEM_FLAG) -U -r requirements-dist.txt

dist: build dist_deps
	$(PYTHON) -m maturin build --release

dist_wheel: build dist_deps
	$(PYTHON) -m maturin build --release

upload:
	$(PYTHON) -m twine check dist/*
	@$(PYTHON) -m twine upload dist/*  -u $(TWINE_USER) -p $(TWINE_PASS)

clean:
	@rm -rf build/ dist/ *.egg-info/ **/__pycache__/
	@rm -rf toad/*.c toad/*.so target/ Cargo.lock
	@rm -rf toad/*.pyx toad/*.pxd

docs: build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%:
	@: