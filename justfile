docs: lint
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True saev

lint: fmt
    ruff check saev/ main.py webapp.py

fmt:
    fd -e py | xargs isort
    fd -e py | xargs ruff format --preview
