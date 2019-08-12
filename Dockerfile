# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# https://github.com/microsoft/vscode-dev-containers/blob/master/containers/python-3/.devcontainer/Dockerfile

FROM python:3.7-buster

# Avoid warnings by switching to noninteractive
ENV POETRY_VERSION=0.12.17 \
    #
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    DEBIAN_FRONTEND=noninteractive

# Copy requirements.txt (if found) to a temp locaition so we can install it. Also
# copy "noop.txt" so the COPY instruction does not fail if no requirements.txt exists.
WORKDIR /setup/
COPY poetry.lock* pyproject.toml ./

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git procps lsb-release \
    #
    && pip install "poetry==$POETRY_VERSION" \
    && poetry config settings.virtualenvs.create false \
    #
    # Update Python environment based on requirements.txt (if presenet)
    && if [ -f "pyproject.toml" ]; then poetry install --no-interaction --no-ansi; fi \
    && rm -rf /setup \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
