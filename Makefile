.PHONY: build test

PYTHON = python3
SPHINXOPTS =
SPHINXBUILD = sphinx-build
SPHINXPROJ = toad
DOCSDIR = docs
SOURCEDIR = $(DOCSDIR)/source
BUILDDIR = $(DOCSDIR)/build

build:
	$(PYTHON) setup.py build_ext --inplace

install:
	$(PYTHON) setup.py install --record files.txt

uninstall:
	cat files.txt | xargs rm -rf

test:
	$(PYTHON) -m unittest discover -s ./tests

publish:
	$(PYTHON) setup.py sdist bdist_wheel --universal
	twine upload dist/*

docs: build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
