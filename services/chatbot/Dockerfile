FROM python:3.9-slim

LABEL maintainer="Tai Le"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY ./requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends iputils-ping && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install 'uvicorn[standard]'

COPY ./data /app/chatbot/data
COPY ./config /app/chatbot/config
COPY ./prompt /app/chatbot/prompt
COPY ./query /app/chatbot/query
COPY ./utils /app/chatbot/utils
COPY ./OpenAI /app/chatbot/OpenAI
COPY ./server /app/chatbot/server
COPY ./chatbot_app.py /app/chatbot/chatbot_app.py

EXPOSE 8000

# CMD ["python", "chatbot_app.py"]