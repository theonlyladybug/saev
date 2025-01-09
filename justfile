docs: lint
    rm -rf docs/saev docs/contrib
    uv run python scripts/docs.py --pkg-names saev contrib --fpath docs/llms.txt
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True saev contrib

test: lint
    uv run pytest --cov saev -n auto saev

lint: fmt
    fd -e py | xargs ruff check

fmt:
    fd -e py | xargs isort
    fd -e py | xargs ruff format --preview
    fd -e elm | xargs elm-format --yes

clean:
    uv run python -c 'import datasets; print(datasets.load_dataset("ILSVRC/imagenet-1k").cleanup_cache_files())'

build-semgseg: fmt
    cd web && elm make apps/semseg/Main.elm --output apps/semseg/dist/app.js --debug
    cd web && tailwindcss --input apps/semseg/main.css --output apps/semseg/dist/main.css

build-classification: fmt
    cd web && elm make src/Classification.elm --output apps/classification/dist/app.js --optimize
    cd web && tailwindcss --input apps/classification/main.css --output apps/classification/dist/main.css --minify

build-comparison: fmt
    cd web && elm make src/Comparison.elm --output apps/comparison/dist/app.js --debug
    cd web && tailwindcss --input apps/comparison/main.css --output apps/comparison/dist/main.css

deploy: build-classification
    uv run python scripts/deploy.py
