.PHONY: install test eval quick
install: ; pip install -e ".[dev]"
test: ; python -m pytest -q
eval: ; python run_eval.py
quick: ; python run_eval.py --quick
