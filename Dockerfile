# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# https://github.com/microsoft/vscode-dev-containers/blob/master/containers/python-3/.devcontainer/Dockerfile

FROM python:3.10-slim
ENV POETRY_VERSION=1.1.12

WORKDIR /app

RUN apt update \
    && apt install -y wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O /wait https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh \
    && chmod u+x /wait

RUN pip install "poetry==${POETRY_VERSION}" \
    && poetry config virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi

RUN pip install nlp-service[server] \
    && python -m spacy download en_core_web_lg
