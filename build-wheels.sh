#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}


# Install system packages
#yum install -y python38u-devel
#yum install -y swig

# Compile wheels
for PYBIN in /opt/python/cp3{8,7}*/bin; do
    "${PYBIN}/pip" install cmake numpy 
	PYTHON_INCLUDE_PATH="${PYBIN}/../include/python3.8"
	PYTHON_LIB_PATH="${PYBIN}/../lib/python3.8"
	#"${PYBIN}/python" /io/test.py
    "${PYBIN}/cmake" /io/ /io/build -DPYTHON_INCLUDE_PATH=$PYTHON_INCLUDE_PATH -DPYTHON_LIBRARY:FILEPATH=$PYTHON_LIB_PATH
	#"${PYBIN}/python" /io/setup.py bdist_wheel
    #"${PYBIN}/pip" install tfc
    #"${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
    #"${PYBIN}/pip" install -r /io/requirements.txt
done
#
## Bundle external shared libraries into the wheels
#for whl in wheelhouse/*.whl; do
#    repair_wheel "$whl"
#done

# Install packages and test
#for PYBIN in /opt/python/*/bin/; do
#    "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
#    (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
#done

