# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# https://github.com/microsoft/vscode-dev-containers/blob/master/containers/python-3/.devcontainer/Dockerfile

FROM python:3.7-slim
ENV POETRY_VERSION=0.12.17

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}" \
    && poetry config settings.virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi
