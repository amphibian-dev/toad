TOAD
^^^^

.. image:: https://img.shields.io/pypi/v/toad.svg?style=flat-square
  :target: https://pypi.org/project/toad/
  :alt: Latest version on PyPi
.. image:: https://img.shields.io/pypi/pyversions/toad.svg?style=flat-square
  :target: https://pypi.org/project/toad/
  :alt: Supported Python versions
.. image:: https://img.shields.io/travis/Secbone/toad/master.svg?style=flat-square
  :target: https://travis-ci.org/Secbone/toad
  :alt: Travis-CI build status


ESC Team's data-detector for credit risk

Install
-------

**via pip**

.. code-block:: bash

    pip install toad


**via source code**

.. code-block:: bash

    python setup.py install


Usage
-----

.. code-block:: python

    import toad


    data = pd.read_csv('test.csv')

    toad.detect(data)

    toad.quality(data, target = 'TARGET', iv_only = True)

    toad.IV(feature, target, method = 'dt', min_samples = 0.1)


Documents
---------

A simple API `docs <docs/API.rst>`_
