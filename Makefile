.PHONY: build test

SHELL = /bin/bash

PYTHON = python3
PIP = pip3
SUDO ?=

SPHINXOPTS =
SPHINXBUILD = sphinx-build
SPHINXPROJ = toad
DOCSDIR = docs
SOURCEDIR := $(DOCSDIR)/source
BUILDDIR := $(DOCSDIR)/build


ifeq ('$(shell type -P python3)','')
    PYTHON = python
endif

ifeq ('$(shell type -P pip3)','')
    PIP = pip
endif


install: build
	$(SUDO) $(PIP) install -e .

uninstall:
	cat files.txt | xargs rm -rf

test_deps:
	$(SUDO) $(PIP) install -r requirements-test.txt

test: test_deps
	$(eval TARGET := $(filter-out $@, $(MAKECMDGOALS)))
	@if [ -z $(TARGET) ]; then \
		$(PYTHON) -m pytest -x toad; \
	else \
		$(PYTHON) -m pytest -s $(TARGET); \
	fi

build_deps:
	$(SUDO) $(PIP) install -r requirements.txt

build: build_deps
	$(PYTHON) setup.py build_ext --inplace

dist_deps:
	$(SUDO) $(PIP) install -U -r requirements-dist.txt

dist: build dist_deps
	$(SUDO) $(PYTHON) setup.py sdist

dist_wheel: build dist_deps
	$(SUDO) $(PYTHON) setup.py bdist_wheel --universal

upload:
	twine check dist/*
	@twine upload dist/*  -u $(TWINE_USER) -p $(TWINE_PASS)

clean:
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf toad/*.c toad/*.so

docs: build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%:
	@:
