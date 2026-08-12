PYTHON ?= python3
PYTHONPATH := src
CLAIMS ?= data/AraFacts.csv
CONTENT ?= data/AraFacts_content.csv

.PHONY: test train report api predict

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train.py --claims $(CLAIMS) --content $(CONTENT) --output-dir artifacts --model-dir models
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/make_report.py --artifacts artifacts

api:
	MIZAN_MODEL_PATH=models/classifier.joblib MIZAN_INDEX_PATH=models/retriever.joblib uvicorn mizan.api:app --app-dir src --host 0.0.0.0 --port 8000

predict:
	@test -n "$(CLAIM)" || (echo 'Usage: make predict CLAIM="..."' && exit 1)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/predict.py "$(CLAIM)" --model-dir models
