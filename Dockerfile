 FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

ARG PROD_ENV
ARG commit_hash=git_commit_default
ARG DEBIAN_FRONTEND=noninteractive

ENV PROD_ENV=${PROD_ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.0.0 \
    COMMIT_HASH=$commit_hash \
    MLFLOW_S3_UPLOAD_EXTRA_ARGS='{"ACL": "bucket-owner-full-control"}' \
    MLFLOW_S3_ENDPOINT_URL=https://s3.eu-central-1.amazonaws.com/ \
    PYTHONPATH=/app/src 


RUN apt-get update && apt-get install -y \
    curl \
    default-libmysqlclient-dev \
    gcc \
    g++ \
    htop \
    locales \
    python3-dev \
    git \
    python3-pip \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* 

RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex &&\
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN python -m spacy download en_core_web_sm

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python3 && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock /app/ 

RUN poetry install $(if [ $PROD_ENV = "production" ]; then echo --no-dev; fi) --no-interaction --no-ansi

COPY src /app/src


# TODO set master addr and port, and set node ranks as argument to docker setup - and then how to scale these? 
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html use this instead
# https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-customize-training-script-pt.html
CMD ["python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr='192.168.1.1' --master_port=1234 src/train.py"]
