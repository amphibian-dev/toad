.PHONY: build test install uninstall test_deps dist_deps dist dist_wheel upload clean docs ensure-uv install-nn

SHELL = /bin/bash

# Detect CI environment
ifdef CI
    # In CI: use system Python and pip
    PYTHON = python3
    PIP = pip3
    PIP_FLAGS =
else
    # Locally: use uv if available, otherwise fall back to system
    PYTHON = python3
    PIP = pip3
    PIP_FLAGS =
    # Check if uv is available
    ifneq ($(shell command -v uv 2> /dev/null),)
        PIP = uv pip
        PIP_FLAGS = --system
    endif
endif

SPHINXOPTS =
SPHINXBUILD = sphinx-build
SPHINXPROJ = toad
DOCSDIR = docs
SOURCEDIR := $(DOCSDIR)/source
BUILDDIR := $(DOCSDIR)/build

# Check if uv is installed, if not install it
ensure-uv:
	@command -v uv >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$$HOME/.local/bin:$$PATH")

install: ensure-uv build
	@echo "Installing with $(PIP)..."
	$(PIP) install $(PIP_FLAGS) -e .

install-nn: ensure-uv build
	@echo "Installing with neural network support..."
	$(PIP) install $(PIP_FLAGS) -e .[nn]

uninstall:
	cat files.txt | xargs rm -rf

test_deps: ensure-uv
	@echo "Installing test dependencies..."
	$(PIP) install $(PIP_FLAGS) -r requirements-test.txt

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
	@echo "Installing build dependencies..."
	$(PIP) install $(PIP_FLAGS) -r requirements.txt
	$(PIP) install $(PIP_FLAGS) maturin

build: build_deps
	@echo "Building Rust extension with maturin..."
	$(PYTHON) -m maturin build --release
	@echo "Installing toad_core from wheel..."
ifdef CI
	# In CI: directly install wheel
	$(PIP) install --force-reinstall --no-deps target/wheels/toad-*.whl
else
	# Locally: extract and copy toad_core to preserve editable install
	@rm -rf /tmp/toad_wheel_extract && mkdir -p /tmp/toad_wheel_extract
	@cd /tmp/toad_wheel_extract && unzip -q $(CURDIR)/target/wheels/toad-*.whl "toad_core/*" 2>/dev/null || true
	@if [ -d "/tmp/toad_wheel_extract/toad_core" ]; then \
		rm -rf .venv/lib/python*/site-packages/toad_core 2>/dev/null || true; \
		mkdir -p .venv/lib/python*/site-packages/ 2>/dev/null || true; \
		cp -r /tmp/toad_wheel_extract/toad_core .venv/lib/python*/site-packages/ 2>/dev/null || true; \
		echo "toad_core module installed to .venv"; \
	else \
		echo "Warning: Could not extract toad_core, may need manual installation"; \
	fi
endif

dist_deps: ensure-uv
	@echo "Installing distribution dependencies..."
	$(PIP) install $(PIP_FLAGS) -U -r requirements-dist.txt

dist: build dist_deps
	$(PYTHON) -m maturin build --release

dist_wheel: build_deps dist_deps
	@rm -rf target/wheels/*
	$(PYTHON) -m maturin build --release

upload:
	$(PYTHON) -m twine check dist/*
	@$(PYTHON) -m twine upload dist/*  -u $(TWINE_USER) -p $(TWINE_PASS)

clean:
	@rm -rf build/ dist/ *.egg-info/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf toad/*.c toad/*.so toad/*.pyx toad/*.pxd
	@rm -rf target/
	@rm -rf .venv/lib/python*/site-packages/toad_core 2>/dev/null || true

docs: build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%:
	@: