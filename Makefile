.PHONY: build test


ifeq ($(PYTHON),)
PYTHON = python
endif

ifeq ($(PIP),)
PIP = pip
endif

SPHINXOPTS =
SPHINXBUILD = sphinx-build
SPHINXPROJ = toad
DOCSDIR = docs
SOURCEDIR = $(DOCSDIR)/source
BUILDDIR = $(DOCSDIR)/build


install:
	$(PIP) install numpy pytest Cython
	$(PIP) install -e .

uninstall:
	cat files.txt | xargs rm -rf

test:
	$(PYTHON) -m pytest -x ./tests

build_deps:
	$(PIP) install -U wheel setuptools twine

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
