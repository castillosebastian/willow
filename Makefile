.PHONY: all env install_models

all: env install_models

env:
	@echo "Creating virtual environment..."
	python3 -m venv ~/.willow
	@echo "Activating virtual environment..."
	. ~/.willow/bin/activate
	@echo "Installing requirements..."
	pip install -r requirements.txt

install_models:
	@echo "Downloading spaCy model..."
	python -m spacy download es_core_news_sm
	@echo "Downloading FastText model..."
	curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec
	@echo "Moving FastText model to models directory..."
	mkdir -p models
	mv wiki.es.vec models/
