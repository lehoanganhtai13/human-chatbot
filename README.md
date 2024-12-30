# Human-like Assistant Chatbot System 🗨️

Hi 👋 Welcome to the official repository for our **Human-like Assistant Chatbot System**!  

This project is dedicated to building a personalized, human-like assistant chatbot that evolves into a true companion. By leveraging advanced **GraphRAG** technology, our system allows users to create their own personalized assistant starting from an initial instruction that defines the assistant's personal background (identification, education, professional work, family, etc.).  

## Key Features ✨  
- **Emotionally Rich Conversations**:  
  The chatbot engages in natural, emotionally relatable dialogues, always maintaining a warm, empathetic tone. It listens attentively, supports the user emotionally, and adapts to the dynamics of the conversation.  

- **Human-like Personalization**:  
  The assistant stores memories and progressively understands the user’s behavior, preferences, and habits. This creates a deeply personalized experience that feels like interacting with a real human friend.  

- **Dynamic Adaptability**:  
  From cheerful encouragement to thoughtful nostalgia, the assistant can switch conversational styles to match the user's emotional state, ensuring every interaction feels meaningful.  

- **Advanced Memory System**:  
  Inspired by human memory, the chatbot incorporates a memory storage mechanism that retains key conversational moments. This feature enables it to "remember" past interactions and grow its understanding of the user over time. 

- **Specialized Knowledge Integration** *(Coming Soon)*:  
  Alongside the assistant’s personalized background defined through the initial instruction, users will be able to equip their assistant with specialized knowledge in any field by providing relevant documents (e.g., PDFs, DOCX files). This feature will enable the assistant to act as a domain expert or guide in areas like medicine, technology, or education, enhancing its utility and versatility. 

- **In-depth Domain Analysis** *(Coming Soon)*:  
  The assistant will utilize an **agent-based mechanism** to enhance its reasoning capabilities, allowing it to perform in-depth analysis of complex problems within the knowledge domains provided by the user. This approach empowers the assistant to decompose intricate queries, reason step-by-step, and provide insightful solutions, making it a reliable tool for tackling challenging problems and supporting logical decision-making.

This project utilizes the **Retrieval-Augmented Generation (RAG)** framework enhanced with graph-knowledge-based techniques to achieve high-quality, contextual, and personalized responses.  

---

### Progress 🔄

![](https://geps.dev/progress/60)

| Feature                          | Status         | Details                                                                 |
|----------------------------------|----------------|-------------------------------------------------------------------------|
| Emotionally Rich Conversations   | ✅ Completed   | Engaging, warm, and empathetic dialogue.                               |
| Human-like Personalization       | ✅ Completed   | Adaptive memory and tone personalization.                              |
| Dynamic Adaptability             | ✅ Completed   | Switch conversational styles dynamically.                              |
| Advanced Memory System           | 🚧 In Progress | Persistent memory for long-term personalization.                       |
| Specialized Knowledge Integration| 🚧 In Progress | Integrate knowledge from user-provided documents (e.g., PDFs, DOCX).   |
| In-depth Domain Analysis         | 🚧 In Progress | Agent-based reasoning for complex problem analysis and solutions.      |

---

## GPU Requirements 🖥️

### Memory Requirements
- To run this system, you need a minimum GPU memory of `3GB` to host the local embedding model and use the **OpenAI API** for LLM.
- If you want to use the **local LLM** instead of the OpenAI API, you need a minimum GPU memory of `12GB`.
- For optimal results while saving costs, you can combine both the local LLM and the OpenAI API by adjusting the [llm_config.json](./services/chatbot/config/llm_config.json) file to select the provider for each LLM according to your preference. This file allows you to specify whether to use the local LLM or the OpenAI API, as well as other parameters like `temperature` and `max_new_tokens` (**not recommended to adjust these parameters**).

### Compute Capability
- Your GPU must have a compute capability of at least `7.5`.
- If your GPU has a compute capability of `8.0` or higher, set `LOCAL_LLM_DTYPE` in the `.env` file to `bfloat16`.
- If your GPU has a compute capability of `7.5`, set `LOCAL_LLM_DTYPE` in the `.env` file to `float16`.

### Memory Utilization
- The `GPU_MEMORY_UTILIZATION` value in the `.env` file should be set such that the total GPU memory multiplied by this value meets the minimum required GPU memory.
- For example, if you have an RTX 4060 Ti with **16GB** of memory:
  - To host the local LLM, set `GPU_MEMORY_UTILIZATION` to at least `0.55` to ensure **Vllm** has at least **9GB** memory to host the model.
  - Higher values will result in higher throughput for the LLM.

### CUDA Driver
- A minimum CUDA Driver version of `12.1` is required to host the local LLM.

---

## Quick setup 🚀

The following steps will help you to get the system up and running:

- Create network for the whole system. This will create network `human-chatbot` and create an `.env` file with the corresponding value of the network subnet:
    ```bash
    make create-network
    ```
- Setup folders for containers' volume:
    ```bash
    make setup-volumes
    ```
- Setup [.env](./environment/.env) file based on [.template.env](./environment/.template.env) with value of all variables of each service based on your system configuration design.
- You will get the folder `services` structured like below:
   ```bash
   services
   ├── milvus
   │   ├── .data
   │   │   ├── etcd
   │   │   └── milvus
   │   └── docker-compose.yaml
   ├── minio
   │   ├── .data
   │   └── docker-compose.yaml
   ├── model-serving
   │   ├── .cache
   │   ├── embedder
   │   │   ├── logs
   │   │   ├── prometheus_multiproc
   │   │   ├── embedder-app.py
   │   │   └── gunicorn_conf.py
   │   ├── re-ranker
   │   │   ├── logs
   │   │   ├── prometheus_multiproc
   │   │   ├── reranker-app.py
   │   │   └── gunicorn_conf.py
   │   ├── docker-compose.yaml
   │   ├── models.py
   │   └── requirements.txt
   ├── valkey
   │   ├── .data
   │   ├── docker-compose.yaml
   │   └── valkey.conf
   ├── mongodb
   │   ├── .data
   │   ├── logs
   │   └── docker-compose.yaml
   ├── falkordb
   │   ├── .data
   │   ├── docker-compose.yaml
   │   └── falkordb.conf
   └── chatbot
       ├── OpenAI
       ├── client
       ├── config
       ├── data
       ├── prompt
       ├── query
       ├── server
       ├── utils
       ├── chatbot_app.py
       ├── Dockerfile
       ├── docker-compose.yaml
       └── requirements.txt
   ```
- Setup all the base services, including `storage`, `vector database`, `graph database` and `model serving`:
  - Use this command if you only host embedding model:
    ```bash
    make up-setup
    ```
  - Use this command if you want to host both embedding model and your own local LLM:
    ```bash
    make up-setup-with-llm
    ```

- Then, when all the base services are ready, we can now start the server:
   ```bash
   make up-server
   ```

- Finally, we'll run this command to check if all the containers are ready:
    ```bash
    docker ps
    ```
    You will get the following result, which means that the system is all setup:
    ```bash
    IMAGE                                       STATUS                      PORTS                                                                                      NAMES
    chatbot/chatbot-server:latest               Up 25 seconds               0.0.0.0:8050->8000/tcp, :::8050->8000/tcp                                                  chatbot-server
    vllm/vllm-openai:v0.6.5                     Up 23 seconds               0.0.0.0:8013->8000/tcp, :::8011->8000/tcp                                                  model-serving-llm
    chatbot/model-serving-reranker:latest       Up 23 seconds               0.0.0.0:8012->8000/tcp, :::8012->8000/tcp                                                  model-serving-reranker
    chatbot/model-serving-embedder:latest       Up 23 seconds               0.0.0.0:8011->8000/tcp, :::8011->8000/tcp                                                  model-serving-embedder
    falkordb/falkordb:edge                      Up 20 seconds               0.0.0.0:3000->3000/tcp, :::3000->3000/tcp, 0.0.0.0:6380->6379/tcp, :::6380->6379/tcp       falkordb
    mongo:latest                                Up 20 seconds               0.0.0.0:27017->27017/tcp, :::27017->27017/tcp                                              mongodb
    valkey/valkey:8.0.1                         Up 20 seconds               0.0.0.0:6379->6379/tcp, :::6379->6379/tcp                                                  valkey
    zilliz/attu:v2.4.7                          Up 14 seconds               0.0.0.0:3000->3000/tcp, :::3000->3000/tcp                                                  milvus-attu-container
    milvusdb/milvus:v2.4.9                      Up 14 seconds (healthy)     0.0.0.0:9091->9091/tcp, :::9091->9091/tcp, 0.0.0.0:19530->19530/tcp, :::19530->19530/tcp   milvus-standalone-container
    quay.io/coreos/etcd:v3.5.5                  Up 14 seconds (healthy)     2379-2380/tcp                                                                              milvus-etcd-container
    minio/minio:RELEASE.2024-08-29T01-40-52Z    Up 10 seconds (healthy)     0.0.0.0:9000->9000/tcp, :::9000->9000/tcp, 0.0.0.0:9001->9001/tcp, :::9001->9001/tcp       minio-storage-container
    ```
- You can access to the management console of these services:
    - Storage **MinIO**: `http://localhost:9001/`
    - Vector database **Milvus**: `http://localhost:3001/`
    - Graph database **FalkorDB**: `http://localhost:3000/`
    - Chat history database **MongoDB** - **Valkey**: [MongoDB Compass](https://www.mongodb.com/products/tools/compass) - [Redis Insight](https://redis.io/insight/)
    - Model serving: `http://localhost:8011/docs`, `http://localhost:8012/docs`, `http://localhost:8013/docs`
    - Chatbot server: `http://localhost:8050/docs`

---

## Framework Overview 🛠️

Our **GraphRAG-powered Chatbot System** integrates:  
1. **Initial Setup**: Users define an assistant's instruction and background.  
2. **Contextual Understanding**: RAG is enhanced with graph-based reasoning to provide highly contextual responses.  
3. **Memory System**: Persistent memory enables adaptive, user-specific interactions.  
4. **Document-Based Knowledge** *(In Development)*: Process and integrate knowledge from user-provided documents to enrich the assistant's expertise in specific fields.  
5. **Deep Problem Analysis** *(In Development)*: Utilize domain knowledge and agent-based reasoning mechanisms for performing step-by-step in-depth analysis of complex user queries, supporting logical reasoning and personalized problem-solving.

---

We hope you enjoy exploring our project! If you have questions, feel free to open an issue or contribute to this repository. 😊 
