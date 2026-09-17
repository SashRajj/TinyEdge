VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: setup train convert benchmark all clean

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

train:
	$(PY) -m tinyedge.train

convert:
	$(PY) -m tinyedge.convert

benchmark:
	$(PY) -m tinyedge.benchmark

all: train convert benchmark

clean:
	rm -rf models/*.tflite models/*.keras results/*.json
