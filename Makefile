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


install:
	$(SUDO) $(PIP) install numpy pytest Cython
	$(SUDO) $(PIP) install -e .

uninstall:
	cat files.txt | xargs rm -rf

test:
	$(eval TARGET := $(filter-out $@, $(MAKECMDGOALS)))
	@if [ -z $(TARGET) ]; then \
		$(PYTHON) -m pytest tests; \
	else \
		$(PYTHON) -m pytest tests/test_$(TARGET).py; \
	fi

build_deps:
	$(SUDO) $(PIP) install -U wheel setuptools twine

build: build_deps
	$(PYTHON) setup.py build_ext --inplace

dist: build
	$(SUDO) $(PYTHON) setup.py sdist

dist_wheel: build
	$(SUDO) $(PYTHON) setup.py bdist_wheel --universal

patchelf:
	wget http://nixos.org/releases/patchelf/patchelf-$(PATCHELF_VERSION)/patchelf-$(PATCHELF_VERSION).tar.bz2
	tar xf patchelf-$(PATCHELF_VERSION).tar.bz2
	cd patchelf-$(PATCHELF_VERSION) && ./configure && sudo make install

manylinux_docker:
	docker pull $(DOCKER_IMAGE)

dist_manylinux: build dist manylinux_docker
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
