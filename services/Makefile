# Define variables
MINIO_DIR=./minio
MILVUS_DIR=./milvus
MODEL_SERVING_DIR=./model-serving
VALKEY_DIR=./valkey
STREAMLIT_DIR=./streamlit
NETWORK_NAME=human-chatbot

# Target to all services
up-minio:
	@echo "Starting Minio service..."
	@docker compose -f $(MINIO_DIR)/docker-compose.yaml up -d

up-milvus:
	@echo "Starting Milvus service..."
	@docker compose -f $(MILVUS_DIR)/docker-compose.yaml up -d

up-model:
	@echo "Starting Model Serving service..."
	@docker compose -f $(MODEL_SERVING_DIR)/docker-compose.yaml up -d

up-valkey:
	@echo "Starting Valkey service..."
	@docker compose -f $(VALKEY_DIR)/docker-compose.yaml up -d

down-minio:
	@echo "Stopping Minio service..."
	@docker compose -f $(MINIO_DIR)/docker-compose.yaml down

down-milvus:
	@echo "Stopping Milvus service..."
	@docker compose -f $(MILVUS_DIR)/docker-compose.yaml down

down-model:
	@echo "Stopping Model Serving service..."
	@docker compose -f $(MODEL_SERVING_DIR)/docker-compose.yaml down

down-valkey:
	@echo "Stopping Valkey service..."
	@docker compose -f $(VALKEY_DIR)/docker-compose.yaml down

# Targets for network
create-network:
	@echo "Creating network $(NETWORK_NAME)..."
	@docker network create $(NETWORK_NAME)
	@$(MAKE) update-env-milvus
	@$(MAKE) update-env-minio
	@$(MAKE) update-env-model-serving
	@$(MAKE) update-env-rag-core
	@$(MAKE) update-env-valkey
	@$(MAKE) update-env-streamlit

inspect-network:
	@echo "Network subnet: $(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*')"

update-env-minio:
	@echo "Updating minio/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' minio/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' minio/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> minio/.env; \
	fi

update-env-milvus:
	@echo "Updating milvus/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' milvus/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' milvus/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> milvus/.env; \
	fi

update-env-model-serving:
	@echo "Updating model-serving/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' model-serving/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' model-serving/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> model-serving/.env; \
	fi

update-env-rag-core:
	@echo "Updating rag-core/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' rag-core/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' rag-core/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> rag-core/.env; \
	fi

update-env-valkey:
	@echo "Updating valkey/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' valkey/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' valkey/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> valkey/.env; \
	fi

update-env-streamlit:
	@echo "Updating streamlit/.env file with NETWORK_SUBNET..."
	@SUBNET=$(shell docker network inspect $(NETWORK_NAME) | grep -oP '(?<="Subnet": ")[^"]*') && \
	if grep -q '^NETWORK_SUBNET=' streamlit/.env; then \
		sed -i 's/^NETWORK_SUBNET=.*/NETWORK_SUBNET=$$SUBNET/' streamlit/.env; \
	else \
		echo "NETWORK_SUBNET=$$SUBNET" >> streamlit/.env; \
	fi

# Target to setup volumes folder
setup-volumes: 
	@echo "Creating volumes folder..."
	@mkdir -p milvus/.data/etcd milvus/.data/milvus
	@mkdir -p minio/.data
	@mkdir -p model-serving/cache
	@mkdir -p valkey/.data
	@echo "Setting permissions..."
	@chmod -R 767 milvus/.data/ minio/.data/ valkey/ model-serving/cache/ model-serving/nginx

# Target to clean up the volumes
clean:
	@echo "Cleaning up..."
	@$(MAKE) clean-milvus
	@$(MAKE) clean-minio
	@$(MAKE) clean-valkey

clean-milvus:
	@echo "Cleaning up Milvus volumes..."
	@docker run --rm -v $(MILVUS_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-minio:
	@echo "Cleaning up Minio volumes..."
	@docker run --rm -v $(MINIO_DIR):/data alpine sh -c "rm -rf /data/.data"

clean-valkey:
	@echo "Cleaning up Valkey volumes..."
	@docker run --rm -v $(VALKEY_DIR):/data alpine sh -c "rm -rf /data/.data"
