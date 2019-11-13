#!/bin/bash
set -e -x


# Compile wheels
for PYBIN in /opt/python/cp3[5678]*/bin; do
    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel --no-deps /io/ -w /dist/
done

# Bundle external shared libraries into the wheels
for whl in /dist/toad*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/dist/
done