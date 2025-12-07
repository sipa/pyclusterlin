default: test

test_mypy:
	mypy --strict --extra-checks *.py

test_lint:
	pylint --enable-all-extensions -d W0149,R1702,R2004,R1260,R0914,R0912,R0915 *.py

test_py3unittest:
	pypy3 -m unittest *.py

test_pypy3unittest:
	python3 -m unittest *.py

test: test_mypy test_lint test_py3unittest test_pypy3unittest
