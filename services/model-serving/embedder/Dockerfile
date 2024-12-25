FROM python:3.10-slim

LABEL maintainer="Tai Le"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends iputils-ping && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

COPY ../requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt --no-cache-dir

# Directory where source code is located
ARG SERVICE_DIR

COPY ../${SERVICE_DIR} /app
COPY ../models.py /app/models.py

EXPOSE 8011

# CMD ["python", "app.py"]