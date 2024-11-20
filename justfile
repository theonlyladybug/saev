docs: lint
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True saev probing

test: lint
    uv run pytest saev probing

lint: fmt
    fd -e py | xargs ruff check

fmt:
    fd -e py | xargs isort
    fd -e py | xargs ruff format --preview

clean:
    uv run python -c 'import datasets; print(datasets.load_dataset("ILSVRC/imagenet-1k").cleanup_cache_files())'
