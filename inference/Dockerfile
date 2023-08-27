FROM feyntune-base

# configure docker environment
ENV ENV="dev" \
  WORKDIR=/home/feyntune-inference \
  INFERENCE_SERVER_MODEL=bigscience/bloomz-560m \
  INFERENCE_SERVER_PORT=6969 \
  INFERENCE_SERVER_HOST=0.0.0.0

# create user with normal privileges
RUN mkdir -p $WORKDIR && addgroup feyntune-inference && useradd -d $WORKDIR -g feyntune-inference feyntune-inference \
  && chown feyntune-inference:feyntune-inference $WORKDIR

# set working directory
WORKDIR $WORKDIR

# cache python requirements in docker layer
COPY pyproject.toml poetry.lock* ./
# cache project source code (required for local pkg to be installed by poetry)
COPY feyntune-inference* README.md ./

# install python dependencies
#RUN pip uninstall torch -y 
RUN poetry install \
  $(test "$ENV" == production && echo "--no-dev") --no-interaction --no-ansi

# move remaining local code over
COPY --chown=feyntune-inference:feyntune-inference . .

# build vllm from source
RUN cd vllm && pip install -e . && cd ..
RUN python -c "import vllm; assert vllm"

# non-root user
USER feyntune-inference

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
  "--host", $INFERENCE_SERVER_HOST, \
  "--port", $INFERENCE_SERVER_PORT, \
  "--model", $INFERENCE_SERVER_MODEL]