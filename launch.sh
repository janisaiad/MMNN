# check if uv installed and if not install it

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# create venv
uv venv

# activate venv


uv venv
source .venv/bin/activate

uv pip install -e .
uv cache prune

uv run tests/test_env.py