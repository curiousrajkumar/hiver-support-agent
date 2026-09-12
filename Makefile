.PHONY: run eval test label clean

run:
	python run_pipeline.py

eval:
	python run_pipeline.py --mode eval

test:
	pytest -v

label:
	python data/labeler.py

clean:
	rm -rf __pycache__ src/__pycache__ eval/__pycache__ tests/__pycache__ data/__pycache__ .pytest_cache
