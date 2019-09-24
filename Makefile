.PHONY: build test


PYTHON ?= python
PIP ?= pip
SUDO ?=

SPHINXOPTS =
SPHINXBUILD = sphinx-build
SPHINXPROJ = toad
DOCSDIR = docs
SOURCEDIR := $(DOCSDIR)/source
BUILDDIR := $(DOCSDIR)/build



install:
	$(SUDO) $(PIP) install numpy pytest Cython
	$(SUDO) $(PIP) install -e .

uninstall:
	cat files.txt | xargs rm -rf

test:
	$(PYTHON) -m pytest -x ./tests

build_deps:
	$(SUDO) $(PIP) install -U wheel setuptools twine

build: build_deps
	$(PYTHON) setup.py build_ext --inplace

dist: build
	$(SUDO) $(PYTHON) setup.py sdist

dist_wheel: build
	$(SUDO) $(PYTHON) setup.py bdist_wheel --universal

patchelf:
	sudo apt-get update
	sudo apt-get install patchelf

dist_manylinux: build patchelf
	$(SUDO) $(PIP) install -U auditwheel
	$(SUDO) $(PYTHON) setup.py sdist bdist_wheel --universal
	auditwheel repair dist/*.whl

upload:
	twine check dist/*
	@twine upload dist/*  -u $(TWINE_USER) -p $(TWINE_PASS)

clean:
	@rm -rf build/ dist/ *.egg-info/

docs: build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
