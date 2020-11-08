#!/bin/bash
set -e -u -x

## Run docker
DOCKER_IMAGE=quay.io/pypa/manylinux2010_x86_64
PLAT=manylinux2010_x86_64
#DOCKER_IMAGE=quay.io/pypa/manylinux1_x86_64
#PLAT=manylinux1_x86_64
docker pull $DOCKER_IMAGE
docker run --rm -e PLAT=$PLAT -v `pwd`:/io $DOCKER_IMAGE io/build-wheels.sh
