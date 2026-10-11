PY = .venv/bin/python

.PHONY: setup data train horizon plots app test

setup:
	uv venv --python 3.12 .venv
	uv pip install -p $(PY) -r requirements.txt pytest

data:    ## join EIA + weather (needs data/raw/PJM.xlsx)
	cd src && ../$(PY) fetch_grid.py && ../$(PY) build_dataset.py

train:   ## features, baselines, model, metrics
	cd src && ../$(PY) features.py && ../$(PY) train.py

horizon: ## accuracy by forecast distance
	cd src && ../$(PY) horizon_eval.py

plots:
	cd src && ../$(PY) report_figures.py

app:
	$(PY) app.py

test:
	$(PY) -m pytest -q
