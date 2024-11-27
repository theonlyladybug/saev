docs: lint
    rm -rf docs/saev docs/contrib
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True saev contrib
    uv run python scripts/docs.py --pkg-names saev contrib --fpath docs/llms.txt

test: lint
    uv run pytest --cov saev probing -n auto saev

lint: fmt
    fd -e py | xargs ruff check

fmt:
    fd -e py | xargs isort
    fd -e py | xargs ruff format --preview

clean:
    uv run python -c 'import datasets; print(datasets.load_dataset("ILSVRC/imagenet-1k").cleanup_cache_files())'
