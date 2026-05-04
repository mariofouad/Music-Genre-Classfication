# macOS often has `python3` but no `python`; override with `make PYTHON=python` on Windows if needed.
PYTHON ?= python3
POETRY := $(PYTHON) -m poetry
CONFIG ?= configs/project.yaml

.PHONY: install data train evaluate pipeline dashboard serve test lint build ci clean

install:
	$(POETRY) install

data:
	$(POETRY) run music-genre-data --config $(CONFIG)

train:
	$(POETRY) run music-genre-train --config $(CONFIG)

evaluate:
	$(POETRY) run music-genre-evaluate --config $(CONFIG)

pipeline: data train evaluate

dashboard:
	$(POETRY) run streamlit run src/music_genre/dashboard.py

serve: pipeline dashboard

test:
	$(POETRY) run pytest

lint:
	$(POETRY) run ruff check src tests

build:
	$(POETRY) check --lock
	$(POETRY) build

ci: lint test build

clean:
	powershell -NoProfile -Command "Remove-Item -Recurse -Force data/interim,data/processed,reports/*.json,models/*.joblib -ErrorAction SilentlyContinue"
