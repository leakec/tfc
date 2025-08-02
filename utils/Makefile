# Get version from the version.py file
SRC_DIR = ../src

VERSION=$(shell cd ../src/tfc; python -c "from version import __version__; print(__version__)")

PYTHON_PKG_FILES=$(shell find $(SRC_DIR))
PYTHON_WHEEL=tfc-$(VERSION)-*.whl
PYTHON_WHEEL_DIST=../dist/$(PYTHON_WHEEL)

ifdef VIRTUAL_ENV
	USE_VENV := true
else
	USE_VENV := false
endif
../dist:
	mkdir -p ../dist

ifeq ($(USE_VENV),true)
install: ../dist
	cd ../; python setup.py bdist_wheel
	uv pip uninstall tfc; uv pip install ../dist/*.whl
else
install: ../dist
	cd ../; python setup.py bdist_wheel
	pip uninstall -y tfc; pip install ../dist/*.whl
endif

clean-python:
	rm -f ../dist/*.whl
	rm -rf ../build

clean: clean-python

format: 
	@cd ../src/tfc && fd -e cc -e h | xargs clang-format -i --verbose --style="file:../../clang-format.yaml"
	@cd ../src/tfc && black .
