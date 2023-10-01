.PHONY: build
build:
	python -m pipenv install --dev -e .
	sphinx-build -b html docs/source/ docs/build/html
	python -m build
	twine check dist/*

.PHONY: buildt
buildt:
	python -m pipenv install --dev -e .
	rm tests/data/*
	python tests/build_data.py

.PHONY: clean
clean:
	rm -rf docs/build/html
	rm -rf dist
	rm -rf src/imgfilt.egg-info
	rm -rf tests/__pycache__
	rm -rf tests/*.pyc
	rm -rf src/imgfilt/__pycache__
	rm -rf src/imgfilt/*.pyc
	rm -rf src/imgfilt/pattern/__pycache__
	rm -f *.log
	rm -f *.json
	rm -f *.jpg
	python -m pipenv uninstall imgfilt
	python -m pipenv install --dev -e .

.PHONY: docs
docs:
	rm -rf docs/build/html
	sphinx-build -b html docs/source/ docs/build/html

.PHONY: pre
pre:
	python precommit.py
	git status

.PHONY: test
test:
	python -m pytest --capture=fd


.PHONY: testv
testv:
	python -m pytest -vv  --capture=fd
