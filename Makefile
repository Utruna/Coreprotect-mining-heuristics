PYTHON := .venv/Scripts/python.exe

.PHONY: install lint format test run init clean

install:
	$(PYTHON) -m pip install -e .[dev,viz]

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m xray_detector

init:
	$(PYTHON) -m xray_detector init

clean:
	Remove-Item -Recurse -Force build, dist, .pytest_cache, .ruff_cache, __pycache__ -ErrorAction SilentlyContinue
