# Define variables
MINIO_DIR=./services/minio
MILVUS_DIR=./services/milvus
MODEL_SERVING_DIR=./services/model-serving
VALKEY_DIR=./services/valkey
MONGODB_DIR=./services/mongodb
FALKORDB_DIR=./services/falkordb
NEO4J_DIR=./services/neo4j
CHATBOT_DIR=./services/chatbot
STREAMLIT_DIR=./services/streamlit
ENVIROMENT_DIR=./environment
BACKUP_DIR=./backup
NETWORK_NAME=human-chatbot

# Target to all services
up-minio:
	@echo "Starting Minio service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MINIO_DIR)/docker-compose.yaml up -d

up-milvus:
	@echo "Starting Milvus service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MILVUS_DIR)/docker-compose.yaml up -d

up-embedder:
	@echo "Starting Embedder service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml build model-serving-embedder
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml up -d model-serving-embedder

up-reranker:
	@echo "Starting Reranker service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml build model-serving-reranker
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml up -d model-serving-reranker

up-llm:
	@echo "Starting LLM service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml up -d model-serving-llm

up-model:
	@echo "Starting Model Serving service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml up -d

up-valkey:
	@echo "Starting Valkey service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(VALKEY_DIR)/docker-compose.yaml up -d

up-mongodb:
	@echo "Starting MongoDB service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MONGODB_DIR)/docker-compose.yaml up -d

up-neo4j:
	@echo "Starting Neo4j service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(NEO4J_DIR)/docker-compose.yaml up -d

up-falkordb:
	@echo "Starting FalkorDB service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(FALKORDB_DIR)/docker-compose.yaml up -d

up-chatbot:
	@echo "Starting Chatbot service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(CHATBOT_DIR)/docker-compose.yaml build chatbot-server
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(CHATBOT_DIR)/docker-compose.yaml up -d chatbot-server

up-setup:
	@echo "Setting up services..."
	@$(MAKE) up-embedder
	@$(MAKE) up-model
	@$(MAKE) up-valkey
	@$(MAKE) up-mongodb
	@$(MAKE) up-falkordb

up-setup-with-llm:
	@echo "Setting up services with LLM..."
	@$(MAKE) up-embedder
	@$(MAKE) up-llm
	@$(MAKE) up-model
	@$(MAKE) up-valkey
	@$(MAKE) up-mongodb
	@$(MAKE) up-falkordb

up-server:
	@echo "Starting Chatbot server..."
	@$(MAKE) up-chatbot

down-minio:
	@echo "Stopping Minio service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MINIO_DIR)/docker-compose.yaml down

down-milvus:
	@echo "Stopping Milvus service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MILVUS_DIR)/docker-compose.yaml down

down-embedder:
	@echo "Stopping Embedder service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml down model-serving-embedder

down-reranker:
	@echo "Stopping Reranker service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml down model-serving-reranker

down-llm:
	@echo "Stopping LLM service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml down model-serving-llm

down-model:
	@echo "Stopping Model Serving service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MODEL_SERVING_DIR)/docker-compose.yaml down

down-valkey:
	@echo "Stopping Valkey service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(VALKEY_DIR)/docker-compose.yaml down

down-mongodb:
	@echo "Stopping MongoDB service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(MONGODB_DIR)/docker-compose.yaml down

down-neo4j:	
	@echo "Stopping Neo4j service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(NEO4J_DIR)/docker-compose.yaml down

down-falkordb:
	@echo "Stopping FalkorDB service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(FALKORDB_DIR)/docker-compose.yaml down

down-chatbot:
	@echo "Stopping Chatbot service..."
	@docker compose --env-file $(ENVIROMENT_DIR)/.env -f $(CHATBOT_DIR)/docker-compose.yaml down

# Targets for network
create-network:
	@echo "Creating network $(NETWORK_NAME)..."
	@docker network create $(NETWORK_NAME)
	@$(MAKE) update-env

inspect-network:
	@echo "Network subnet: $(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*')"

update-env:
	@echo "Updating $(ENVIROMENT_DIR)/.env file with NETWORK_SUBNET..."
	@mkdir -p $(ENVIROMENT_DIR) && touch $(ENVIROMENT_DIR)/.env
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' $(ENVIROMENT_DIR)/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' $(ENVIROMENT_DIR)/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> $(ENVIROMENT_DIR)/.env; \
	fi

remove-network:
	@echo "Removing network $(NETWORK_NAME)..."
	@docker network rm $(NETWORK_NAME)
	@echo "Removing NETWORK_SUBNET from $(ENVIROMENT_DIR)/.env..."
	@sed -i '/^NETWORK_SUBNET=/d' $(ENVIROMENT_DIR)/.env

# Target to setup volumes folder
setup-volumes: 
	@echo "Creating volumes folder..."
	@$(MAKE) setup-volumes-minio
	@$(MAKE) setup-volumes-milvus
	@$(MAKE) setup-volumes-model
	@$(MAKE) setup-volumes-valkey
	@$(MAKE) setup-volumes-mongodb
	@$(MAKE) setup-volumes-neo4j
	@$(MAKE) setup-volumes-falkordb

setup-volumes-minio:
	@echo "Creating volumes folder for minio..."
	@mkdir -p $(MINIO_DIR)/.data
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(MINIO_DIR)/.data

setup-volumes-milvus:
	@echo "Creating volumes folder for milvus..."
	@mkdir -p $(MILVUS_DIR)/.data/etcd $(MILVUS_DIR)/.data/milvus
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(MILVUS_DIR)/.data/etcd $(MILVUS_DIR)/.data/milvus

setup-volumes-model:
	@echo "Creating volumes folder for models..."
	@mkdir -p $(MODEL_SERVING_DIR)/embedder/prometheus_multiproc $(MODEL_SERVING_DIR)/embedder/logs \
		$(MODEL_SERVING_DIR)/re-ranker/prometheus_multiproc $(MODEL_SERVING_DIR)/re-ranker/logs \
		$(MODEL_SERVING_DIR)/.cache
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(MODEL_SERVING_DIR)/embedder/prometheus_multiproc $(MODEL_SERVING_DIR)/embedder/logs \
		$(MODEL_SERVING_DIR)/re-ranker/prometheus_multiproc $(MODEL_SERVING_DIR)/re-ranker/logs \
		$(MODEL_SERVING_DIR)/.cache

setup-volumes-valkey:
	@echo "Creating volumes folder for valkey..."
	@mkdir -p $(VALKEY_DIR)/.data
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(VALKEY_DIR)/.data
	@sudo chmod 777 $(VALKEY_DIR)/valkey.conf

setup-volumes-mongodb:
	@echo "Creating volumes folder for mongodb..."
	@mkdir -p $(MONGODB_DIR)/.data $(MONGODB_DIR)/logs
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(MONGODB_DIR)/.data $(MONGODB_DIR)/logs

setup-volumes-neo4j:
	@echo "Creating volumes folder for neo4j..."
	@mkdir -p $(NEO4J_DIR)/.data/data $(NEO4J_DIR)/.data/logs $(NEO4J_DIR)/.data/plugins
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(NEO4J_DIR)/.data

setup-volumes-falkordb:
	@echo "Creating volumes folder for falkordb..."
	@mkdir -p $(FALKORDB_DIR)/.data
	@echo "Setting permissions..."
	@sudo chmod -R 777 $(FALKORDB_DIR)/.data
	@sudo chmod 777 $(FALKORDB_DIR)/falkordb.conf
	@sudo chmod 777 $(FALKORDB_DIR)/start.sh

# Target to clean up the volumes
clean:
	@echo "Cleaning up..."
	@$(MAKE) clean-milvus
	@$(MAKE) clean-minio
	@$(MAKE) clean-model
	@$(MAKE) clean-valkey
	@$(MAKE) clean-mongodb
	@$(MAKE) clean-neo4j
	@$(MAKE) clean-falkordb

clean-milvus:
	@echo "Cleaning up Milvus volumes..."
	@docker run --rm -v $(MILVUS_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-minio:
	@echo "Cleaning up Minio volumes..."
	@docker run --rm -v $(MINIO_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-model:
	@echo "Cleaning up Model Serving volumes..."
	@docker run --rm -v $(MODEL_SERVING_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-valkey:
	@echo "Cleaning up Valkey volumes..."
	@docker run --rm -v $(VALKEY_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-mongodb:
	@echo "Cleaning up MongoDB volumes..."
	@docker run --rm -v $(MONGODB_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-neo4j:
	@echo "Cleaning up Neo4j volumes..."
	@docker run --rm -v $(NEO4J_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-falkordb:
	@echo "Cleaning up Falkordb volumes..."
	@docker run --rm -v $(FALKORDB_DIR):/data alpine sh -c "rm -rf /data/.data"

# Target to backup the database
backup:
	@echo "Backing up database..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/mongodb $(BACKUP_DIR)/${FOLDER}/neo4j $(BACKUP_DIR)/${FOLDER}/valkey
	@sudo cp -r $(MONGODB_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/mongodb
	@sudo cp -r $(MONGODB_DIR)/logs $(BACKUP_DIR)/$(FOLDER)/mongodb
	@sudo cp -r $(NEO4J_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/neo4j
	@sudo cp -r $(VALKEY_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/valkey
	@sudo cp -r $(FALKORDB_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/falkordb

# >>> make backup FOLDER=23-11-24

backup-mongodb:
	@echo "Backing up MongoDB data..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/mongodb
	@sudo cp -r $(MONGODB_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/mongodb
	@sudo cp -r $(MONGODB_DIR)/logs $(BACKUP_DIR)/$(FOLDER)/mongodb

backup-neo4j:
	@echo "Backing up Neo4j data..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/neo4j
	@sudo cp -r $(NEO4J_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/neo4j

backup-valkey:
	@echo "Backing up Valkey data..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/valkey
	@sudo cp -r $(VALKEY_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/valkey

backup-falkordb:
	@echo "Backing up FalkorDB data..."
	@mkdir -p $(BACKUP_DIR)/${FOLDER}/falkordb
	@sudo cp -r $(FALKORDB_DIR)/.data $(BACKUP_DIR)/$(FOLDER)/falkordb

# Target to restore the database
restore:
	@echo "Restoring data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/mongodb/.data/ $(MONGODB_DIR)/.data/
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/mongodb/logs/ $(MONGODB_DIR)/logs/
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/neo4j/.data/ $(NEO4J_DIR)/.data/
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/valkey/.data/ $(VALKEY_DIR)/.data/
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/falkordb/.data/ $(FALKORDB_DIR)/.data/
	@sudo chmod -R 777 $(MONGODB_DIR)/.data $(MONGODB_DIR)/logs $(NEO4J_DIR)/.data $(VALKEY_DIR)/.data $(FALKORDB_DIR)/.data

# >>> make restore FOLDER=23-11-24

restore-mongodb:
	@echo "Restoring MongoDB data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/mongodb/.data/ $(MONGODB_DIR)/.data/
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/mongodb/logs/ $(MONGODB_DIR)/logs/
	@sudo chmod -R 777 $(MONGODB_DIR)/.data $(MONGODB_DIR)/logs

restore-neo4j:
	@echo "Restoring Neo4j data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/neo4j/.data/ $(NEO4J_DIR)/.data/
	@sudo chmod -R 777 $(NEO4J_DIR)/.data

restore-valkey:
	@echo "Restoring Valkey data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/valkey/.data/ $(VALKEY_DIR)/.data/
	@sudo chmod -R 777 $(VALKEY_DIR)/.data

restore-falkordb:
	@echo "Restoring FalkorDB data..."
	@sudo rsync -a --delete $(BACKUP_DIR)/${FOLDER}/falkordb/.data/ $(FALKORDB_DIR)/.data/
	@sudo chmod -R 777 $(FALKORDB_DIR)/.data

# Target to check GPU
check-gpu:
	@echo "Checking GPU..."
	@docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
