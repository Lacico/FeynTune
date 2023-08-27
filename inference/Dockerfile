FROM nvcr.io/nvidia/pytorch:22.12-py3

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
  WORKDIR=/home/feyntune-inference

# install system dependencies (incl build python)
RUN apt update \
  && apt -y install netcat gcc curl make openssl systemd git build-essential \
    curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev wget \
    xz-utils zlib1g-dev \
  && apt clean

# create user with normal privileges
RUN mkdir -p $WORKDIR && addgroup feyntune-inference && useradd -d $WORKDIR -g feyntune-inference feyntune-inference \
  && chown feyntune-inference:feyntune-inference $WORKDIR

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
COPY feyntune-inference* README.md ./

# install python dependencies
#RUN pip uninstall torch -y 
RUN poetry install \
  $(test "$YOUR_ENV" == production && echo "--no-dev") --no-interaction --no-ansi

# move remaining local code over
COPY --chown=feyntune-inference:feyntune-inference . .

# build vllm from source
RUN cd vllm && pip install -e . && cd ..
RUN python -c "import vllm; assert vllm"

# non-root user
USER feyntune-inference

# for testing
ENV INFERENCE_SERVER_MODEL=bigscience/bloomz-560m \
  INFERENCE_SERVER_PORT=6969 \
  INFERENCE_SERVER_HOST=0.0.0.0
EXPOSE $INFERENCE_SERVER_PORT

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
  "--host", $INFERENCE_SERVER_HOST, \
  "--port", $INFERENCE_SERVER_PORT, \
  "--model", $INFERENCE_SERVER_MODEL]