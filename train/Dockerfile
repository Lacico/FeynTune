# syntax = docker/dockerfile:1
FROM feyntune-base as cache
ARG HUGGINGFACE_KEY
ENV HUGGINGFACE_KEY=$HUGGINGFACE_KEY
COPY cache_requirements.txt requirements.txt
COPY src/model_paths.py src/model_paths.py
COPY src/cache_models.py src/cache_models.py
RUN pip install -r requirements.txt
RUN python src/cache_models.py
COPY src/data_loader.py src/data_loader.py
RUN python src/data_loader.py

FROM cache as builder
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY src src

FROM builder as finetuner 
ENTRYPOINT ["python", "src/main.py"]