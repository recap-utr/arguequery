# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# https://github.com/microsoft/vscode-dev-containers/blob/master/containers/python-3/.devcontainer/Dockerfile

FROM python:3.7-buster

ARG POETRY_VERSION=0.12.17

# Avoid warnings by switching to noninteractive
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Configure apt and install packages
RUN apt-get update \
    # VSCode suggestions
    && apt-get -y install --no-install-recommends apt-utils 2>&1 \
    && apt-get -y install git procps lsb-release \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    #
    # Install poetry
    && pip install "poetry==${POETRY_VERSION}" \
    && poetry config settings.virtualenvs.create false

# Install poetry and packages
COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

CMD [ "bash" ]
