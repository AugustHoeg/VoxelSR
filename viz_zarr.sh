#!/bin/bash

ROOT=W:
ZARR_PATH=${ROOT}/bone_2_ome_super.zarr
ZARR_GROUP=HR

# activate the virtual environment
VENV_DIR=.
VENV_NAME=venv
if [ -d "${VENV_DIR}/${VENV_NAME}" ]; then
    echo "Activating virtual environment: ${VENV_DIR}/${VENV_NAME}"
    source "${VENV_DIR}/${VENV_NAME}/bin/activate"
else
    echo "Virtual environment not found: ${VENV_DIR}/${VENV_NAME}"
    exit 1
fi

echo "Visualizing: ${ZARR_PATH}"

# Run the visualization script
qim3d viz ${ZARR_PATH}/${ZARR_GROUP}






