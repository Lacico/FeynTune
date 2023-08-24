FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# configure docker environment
ENV YOUR_ENV="dev" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PYENV_ROOT=/usr/local/.pyenv \
  POETRY_HOME=/opt/poetry \
  POETRY_REQUESTS_TIMEOUT=120 \
  PIP_NO_CACHE_DIR=1 \
  POETRY_VIRTUALENVS_IN_PROJECT=1 \
  # create global env with poetry for selected python version
  POETRY_VIRTUALENVS_CREATE=0 \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  DEBIAN_FRONTEND=noninteractive \ 
  PYTHON_VERSION=3.10.8 \
  WORKDIR=/home/feyntune

# install system dependencies (incl build python)
RUN apt update \
  && apt -y install netcat gcc curl make openssl systemd git build-essential \
    curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev wget \
    xz-utils zlib1g-dev \
  && apt clean

# create user with normal privileges
RUN mkdir -p $WORKDIR && addgroup feyntune && useradd -d $WORKDIR -g feyntune feyntune \
  && chown feyntune:feyntune $WORKDIR

# set working directory
WORKDIR $WORKDIR

# -- python
# Set-up necessary Env vars for PyEnv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
# Install pyenv
RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash

# install poetry
ENV PATH $POETRY_HOME/bin:$PATH
RUN curl -sSL https://install.python-poetry.org | python3 -

# cache python requirements in docker layer
COPY pyproject.toml poetry.lock* ./
# cache project source code (required to be installed by poetry)
COPY feyntune* README.md ./

# install python dependencies
RUN poetry install \
  $(test "$YOUR_ENV" == production && echo "--no-dev") --no-interaction --no-ansi

# non-root user
USER feyntune

# move remaining local code over
COPY --chown=feyntune:feyntune . .

CMD ["poetry", "run", "poe", "jl"]