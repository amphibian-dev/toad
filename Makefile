install:
	python setup.py install

uninstall:
	python setup.py install --record files.txt
	cat files.txt | xargs rm -rf
