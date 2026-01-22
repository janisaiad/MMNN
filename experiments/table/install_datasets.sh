#!/bin/bash
# we install all PDE benchmark datasets in one command

echo "installing PDE benchmark datasets..."

# we install core dependencies
pip install datasets pdebench || uv pip install datasets pdebench

# we install PDEArena
if [ ! -d "pdearena" ]; then
    git clone https://github.com/pdearena/pdearena.git
    cd pdearena && pip install -e . || uv pip install -e . && cd ..
fi

# we install PDEGym (Poseidon)
if [ ! -d "poseidon" ]; then
    git clone https://github.com/camlab-ethz/poseidon.git
    cd poseidon && pip install -e . || uv pip install -e . && cd ..
fi

echo "datasets installed! note: PINNacle may need manual setup - check their repository"
