# syntax = docker/dockerfile:1
FROM feyntune-base as cache
ARG HUGGINGFACE_KEY
ENV WORKDIR=/home/feyntune-train \
    HUGGINGFACE_KEY=$HUGGINGFACE_KEY

# create user with normal privileges
RUN mkdir -p $WORKDIR && addgroup feyntune-train && useradd -d $WORKDIR -g \
    feyntune-train feyntune-train \
    && chown feyntune-train:feyntune-train $WORKDIR

# set working directory
WORKDIR $WORKDIR

COPY pyproject-core.toml pyproject.toml
COPY README.md .
COPY feyntune/model_paths.py feyntune/model_paths.py
COPY feyntune/cache_models.py feyntune/cache_models.py
# apparently poetry really fails here (as it often does with pytorch), 
# so add torch dependency explicitly before poetry install to avoid:
# ImportError: libcupti.so.11.7: cannot open shared object file: No such file or directory
RUN poetry run pip install transformers[torch]
RUN python feyntune/cache_models.py
RUN poetry install
COPY feyntune/data_loader.py feyntune/data_loader.py
RUN poetry run python feyntune/data_loader.py

FROM cache as builder
COPY pyproject.toml poetry.lock* ./
RUN poetry install
COPY feyntune feyntune

FROM builder as finetuner 
ENTRYPOINT ["python", "feyntune/main.py"]