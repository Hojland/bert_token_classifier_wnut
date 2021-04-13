%load_ext autoreload
%autoreload 2
import logging
import os
import re
from pathlib import Path

import boto3
import mlflow
import torch

import settings
from model import bert, model_utils
from utils import utils


def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

# TODO  maybe to do this distributed learning correctly we need to use click? not certain 
def main():
    logger = utils.get_logger(__name__)
    texts, tags = read_wnut('../example_data/wnut17train.conll')
    #mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)

    train_texts, test_texts, train_tags, test_tags = model_utils.train_test_split(texts, tags, test_size=.2)

    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    params = bert.BertParams(unique_tags=unique_tags, tag2id=tag2id, id2tag=id2tag, epochs=100)
    trainer = bert.BertTokenTrainer(params=params, num_labels=len(unique_tags))

    trainer.train(train_texts, train_tags)

if __name__ == '__main__':
    main()
