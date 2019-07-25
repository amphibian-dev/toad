.PHONY: build test


PYTHON ?= python
PIP ?= pip
PIP_USER ?=

SPHINXOPTS =
SPHINXBUILD = sphinx-build
SPHINXPROJ = toad
DOCSDIR = docs
SOURCEDIR := $(DOCSDIR)/source
BUILDDIR := $(DOCSDIR)/build

PIP_INSTALL := $(PIP) install

ifdef PIP_USER
PIP_INSTALL += --user
endif



install:
	$(PIP_INSTALL) numpy pytest Cython
	$(PIP_INSTALL) -e .

uninstall:
	cat files.txt | xargs rm -rf

test:
	$(PYTHON) -m pytest -x ./tests

build_deps:
	$(PIP_INSTALL) -U wheel setuptools twine

build: build_deps
	$(PYTHON) setup.py build_ext --inplace

dist: build
	$(PYTHON) setup.py sdist

dist_wheel: build
	$(PYTHON) setup.py bdist_wheel --universal

upload:
	twine check dist/*
	@twine upload dist/*  -u $(TWINE_USER) -p $(TWINE_PASS)

clean:
	@rm -rf build/ dist/ *.egg-info/

docs: build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
