.PHONY: build test

PYTHON = python3
PIP = pip3
SPHINXOPTS =
SPHINXBUILD = sphinx-build
SPHINXPROJ = toad
DOCSDIR = docs
SOURCEDIR = $(DOCSDIR)/source
BUILDDIR = $(DOCSDIR)/build


install:
	$(PYTHON) setup.py install --record files.txt

uninstall:
	cat files.txt | xargs rm -rf

test:
	$(PYTHON) -m pytest -x ./tests

build_deps:
	$(PIP) install wheel setuptools twine==1.12.0

build: build_deps
	$(PYTHON) setup.py build_ext --inplace

dist: build
	$(PYTHON) setup.py sdist

dist_wheel: build
	$(PYTHON) setup.py bdist_wheel --universal

upload:
	twine check dist/*
	twine upload dist/*  -u $(TWINE_USER) -p $(TWINE_PASS)

clean:
	@rm -rf build/ dist/ *.egg-info/

docs: build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
