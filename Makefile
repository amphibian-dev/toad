build:
	python3 setup.py build_ext --inplace

install:
	python3 setup.py install

uninstall:
	python3 setup.py install --record files.txt
	cat files.txt | xargs rm -rf

test:
	python3 -m unittest discover -s ./tests
