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

PATCHELF_VERSION = 0.10

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
	$(SUDO) $(PIP) install pytest

test: test_deps
	$(eval TARGET := $(filter-out $@, $(MAKECMDGOALS)))
	@if [ -z $(TARGET) ]; then \
		$(PYTHON) -m pytest -x toad; \
	else \
		$(PYTHON) -m pytest -s toad/$(TARGET)_test.py; \
	fi

build_deps:
	$(SUDO) $(PIP) install numpy Cython setuptools

build: build_deps
	$(PYTHON) setup.py build_ext --inplace

dist_deps:
	$(SUDO) $(PIP) install -U wheel twine

dist: build dist_deps
	$(SUDO) $(PYTHON) setup.py sdist

dist_wheel: build dist_deps
	$(SUDO) $(PYTHON) setup.py bdist_wheel --universal

patchelf:
	wget http://nixos.org/releases/patchelf/patchelf-$(PATCHELF_VERSION)/patchelf-$(PATCHELF_VERSION).tar.bz2
	tar xf patchelf-$(PATCHELF_VERSION).tar.bz2
	cd patchelf-$(PATCHELF_VERSION) && ./configure && sudo make install

manylinux_docker:
	docker pull $(DOCKER_IMAGE)

dist_manylinux: dist manylinux_docker
	docker run --rm -e PLAT=$(PLAT) -v $(shell pwd):/io $(DOCKER_IMAGE) $(PRE_CMD) /io/scripts/build_wheels.sh

upload:
	twine check dist/*
	@twine upload dist/*  -u $(TWINE_USER) -p $(TWINE_PASS)

clean:
	@rm -rf build/ dist/ *.egg-info/

docs: build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%:
	@:
