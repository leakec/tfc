# Get version from the version.py file
SRC_DIR = ../src

VERSION=$(shell cd ../src/tfc; python -c "from version import __version__; print(__version__)")

PYTHON_PKG_FILES=$(shell find $(SRC_DIR))
PYTHON_WHEEL=tfc-$(VERSION)-*.whl
PYTHON_WHEEL_DIST=../dist/$(PYTHON_WHEEL)

../dist:
	mkdir -p ../dist

install: ../dist
	cd ../; python setup.py bdist_wheel
	pip uninstall -y tfc; pip install ../dist/*.whl

clean-python:
	rm -f ../dist/*.whl
	rm -rf ../build

clean: clean-python


