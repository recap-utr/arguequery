# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# https://github.com/microsoft/vscode-dev-containers/blob/master/containers/python-3/.devcontainer/Dockerfile
# https://github.com/nautobot/nautobot/blob/develop/docker/Dockerfile

ARG POETRY_VERSION=1.1.12
ARG PYTHON_VERSION=3.9

FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt update && \
    apt install -y curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS -o /wait https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh && \
    chmod u+x /wait

ENV PATH="/root/.local/bin:${PATH}"
RUN curl -sS -o /tmp/install-poetry.py https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py && \
    python /tmp/install-poetry.py && \
    rm -f /tmp/install-poetry.py && \
    poetry config virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root
