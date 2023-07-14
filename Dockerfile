FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD} 

RUN apt update && apt install build-essential gcc git wget -y

COPY ./pyproject.toml  pyproject.toml
COPY ./poetry.lock poetry.lock
RUN pip3 install poetry && poetry config virtualenvs.create false && poetry install

COPY ./ /app

# ENTRYPOINT [ "gunicorn",  "app:app" "-k", "uvicorn.workers.UvicornWorker", "--timeout 1500" ]
