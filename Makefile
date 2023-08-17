.PHONY: all env install_models install_db

all: env install_models install_db

env:
	@echo "Creating virtual environment..."
	python3 -m venv ~/.willow
	@echo "Activating virtual environment..."
	. ~/.willow/bin/activate
	@echo "Installing requirements..."
	pip install -r requirements.txt
	@echo "Installing mongodb..."
	sudo apt-get install gnupg curl
	curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
		sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
		--dearmor
	echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
	sudo apt-get update
	sudo apt-get install -y mongodb-org

install_models:
	@echo "Downloading spaCy model..."
	python -m spacy download es_core_news_sm
	@echo "Downloading FastText model..."
	curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec
	@echo "Moving FastText model to models directory..."
	mkdir -p models
	mv wiki.es.vec models/

install_db:
	@echo "create mongodb database"
	mkdir -p mongodb
	mongod --dbpath /mongodb
	
	
