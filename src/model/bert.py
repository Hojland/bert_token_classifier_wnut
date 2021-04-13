import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import List

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    TokenClassificationPipeline,
)
import mlflow.pytorch

from model import model_utils, text_utils

# TODO sagemaker serve
# TODO doccano
# TODO get data


@dataclass
class BertParams:
    """Class for keeping track of an params."""

    unique_tags: set = field(default_factory=set)
    id2tag: dict = field(default_factory=dict)
    tag2id: dict = field(default_factory=dict)
    epochs: int = 10
    n_cpu: int = 8
    max_len: int = 128
    learn_rate: float = 5e-5
    batch_size: int = 64  # would set to 32 if not for fp16=True

    def to_dict(self):
        return asdict(self)

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


class NerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BertTokenBase:
    def __init__(self, params: BertParams = BertParams()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.tokenizer = self.load_tokenizer()
        self.sentencizer = self.load_sentencizer()
        self.config = self.load_config()

    def load_tokenizer(self):
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased", padding="max_length", max_length=self.params.max_len, truncation=True, is_split_into_words=True
        )
        return tokenizer

    def load_sentencizer(self):
        nlp = spacy.load("en_core_web_sm", exclude=["parser"])
        nlp.enable_pipe("senter")  # can disable later when speed is not important
        return nlp

    def load_config(self):
        config = DistilBertConfig(label2id=self.params.tag2id, id2label=self.params.id2tag).from_pretrained(
            "distilbert-base-uncased", finetuning_task="ner"
        )
        return config

    def preprocess(self, text: str):
        text = str(text.encode("utf-8"), "utf-8")
        text = self.clean_text(text)
        sent_encoding_list = []
        sent_offset_mapping_list = []
        sent_text_list = []
        for sent in self.sentencizer(text).sents:
            encodings = self.tokenizer(
                sent.text, is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True
            )
            offset_mapping = encodings.pop("offset_mapping")
            sent_offset_mapping_list.append(offset_mapping)
            sent_text_list.append([sent.text[start:stop] for start, stop in offset_mapping])
            sent_encoding_list.append(encodings)
        return sent_encoding_list, sent_offset_mapping_list, sent_text_list

    def clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text.
        Applies the native BERT Tokenizer text cleaning pipeline to given text.
        BERT changes your text before predictions are made. This function
        returns the text, as it is given to BERT
        """
        text = text_utils._run_strip_modifiers(text)
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or text_utils._is_control(char):  # Deletes char at position idx.
                continue
            if text_utils._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class BertTokenTrainer(BertTokenBase):
    # https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
    def __init__(self, params: BertParams = BertParams(), num_labels: int = None):
        super().__init__(params)
        self.num_labels = num_labels
        self.model = self.load_model(num_labels=num_labels)

    def load_model(self, num_labels: int = None):
        model = DistilBertForTokenClassification(self.config).from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels, label2id=self.params.tag2id, id2label=self.params.id2tag
        )
        return model

    def encode_tags(self, tags, encodings):
        labels = [[self.params.tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels

    def hyperparameter_search(self, texts: List, labels: List, model_name: str = "bert_ner_test"):
        training_args = TrainingArguments(
            output_dir=model_name,
            num_train_epochs=self.params.epochs,
            per_device_train_batch_size=self.params.batch_size,
            warmup_steps=500,
            eval_steps=500,
            fp16=True,  # activate bit 16 trainer - allowing  a double of batch size https://github.com/nvidia/apex
            learning_rate=self.params.learn_rate,
            disable_tqdm=True,
            label_names=self.params.unique_tags,
        )
        self.trainer = Trainer(
            model_init=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            compute_metrics=model_utils.compute_metrics,
        )
        """
        To use this method, you need to have provided a model_init when
        initializing your Trainer: we need to reinitialize the model
        at each new run. This is incompatible with the optimizers argument,
        so you need to subclass Trainer and override the method
        create_optimizer_and_scheduler() for custom optimizer/scheduler.
        """
        best_trial = self.trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            n_samples=10,  # number of trials
        )

    def train(self, texts: List, tags: List, model_name: str = "bert_ner_test", last_checkpoint: str = None):
        """[train the model]

        Args:
            texts (List): [list of text sequences]
            tags (List): [list of list of tags to train  on]
            model_name (str, optional): [name  of the model, used for outputdir and mlflow name]. Defaults to "bert_ner_test".
            last_checkpoint (str, optional): [modelname string  if we want to continue training on that checkpoint]. Defaults to None.
        """
        train_texts, val_texts, train_tags, val_tags = model_utils.train_test_split(texts, tags, test_size=0.2)
        train_encodings = self.tokenizer(
            train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True
        )
        train_labels = self.encode_tags(train_tags, train_encodings)
        train_offset_mapping = train_encodings.pop("offset_mapping")
        train_data = NerDataset(train_encodings, train_labels)

        val_encodings = self.tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        val_labels = self.encode_tags(val_tags, val_encodings)
        val_offset_mapping = val_encodings.pop("offset_mapping")
        val_data = NerDataset(val_encodings, val_labels)

        fp16 = True if self.device.type == "gpu" else False
        # distribute https://stackoverflow.com/questions/63017931/using-huggingface-trainer-with-distributed-data-parallel
        # https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
        # TODO how to distribute on dagster or kubernetes? ask david? with local_rank and workers
        training_args = TrainingArguments(
            output_dir=model_name,
            num_train_epochs=self.params.epochs,
            per_device_train_batch_size=self.params.batch_size,
            warmup_steps=500,
            eval_steps=500,
            fp16=fp16,  # activate bit 16 trainer - allowing  a double of batch size
            learning_rate=self.params.learn_rate,
            disable_tqdm=True,
            label_names=self.params.unique_tags,
        )
        self.trainer = Trainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            tokenizer=self.tokenizer,
            compute_metrics=model_utils.compute_metrics,
        )

        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            checkpoint = None
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        self.trainer.save_model(model_name)  # Saves the tokenizer too for easy upload

        # or save mlflow like. See also article below to easily deploy model with mlflow.
        # Save both tokenizer and model in context in the way huggingface would save and not just pickled
        # self.log_model(
        #    artifact_path=f"obscure/{model_name}",
        #    registered_model_name=model_name,
        # )
        # https://www.alexanderjunge.net/blog/mlflow-sagemaker-deploy/
        # TODO for this project, I should do this instead of FastAPI

        max_train_samples = len(train_data)
        metrics["train_samples"] = min(max_train_samples, len(train_data))

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def eval(self, texts: List, tags: List):
        test_encodings = self.tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        test_labels = self.encode_tags(tags, test_encodings)
        eval_data = TensorDataset(torch.Tensor(test_encodings), torch.Tensor(test_labels))

        logger.info("*** Evaluate ***")
        metrics = self.trainer.evaluate(eval_data)
        # maybe try clasification report by github/seqeval

        max_val_samples = len(eval_data)
        metrics["eval_samples"] = min(max_val_samples, len(eval_data))

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def log_model(self, **kwargs):
        # revised with https://www.alexanderjunge.net/blog/mlflow-sagemaker-deploy/
        mlflow.pytorch.log_model(**kwargs, pytorch_model=self.model, conda_env="utils/resources/conda-env.json")


class BertTokenPredictor(BertTokenBase):
    def __init__(self, params: BertParams = BertParams(), model_name: str = "bert_ner_test"):
        super().__init__(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.config, self.tokenizer = self.load_model(model_name)
        self.model.eval()  # Put model in evaluation mode.
        self.model.to(self.device)

    def load_model(self, model_name: str = "bert_ner_test"):
        # TODO model loaded from mlflow
        # Load model and tokenizer.
        config = DistilBertConfig.from_pretrained(model_name)
        model = DistilBertForTokenClassification(config).from_pretrained(model_name)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        return model, config, tokenizer

    def pipeline(self, text: str):
        # TODO maybe this needs senctenizing
        device = -1 if self.device.type == "cpu" else 0  # set to other device id if more (see how when you have a gpu)
        nlp = TokenClassificationPipeline(model=self.model, tokenizer=self.tokenizer, task="ner", device=device)
        res = nlp(inputs=text)
        return res

    def _predict(self, text: str):
        """[deprecated predict helper. Now just using pipeline from huggingface]

        Args:
            text (str): [text to predict]

        Returns:
            [type]: [prediction scores and words]
        """
        if not text:
            return []

        sent_encoding_list, sent_offset_mapping_list = self.preprocess(text)

        predictions = []
        for sent in sent_encoding_list:
            input_ids = torch.Tensor(sent["input_ids"], device=self.device).reshape(len(sent["input_ids"]), 1).long()
            att_masks = torch.Tensor(sent["attention_mask"], device=self.device)
            with torch.no_grad():
                entities = self.model(input_ids=input_ids, attention_mask=att_masks)[0][0].detach().cpu().numpy()
            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)
            predictions.extend([[logit for logit in logits] for logits in batch_logits])
        # Zip to remove excess padding in predictions.
        res = []
        for t_seq, p_seq in zip(sent_encoding_list, predictions):
            tp = list(zip(t_seq, (self.params.id2tag[p] for p in p_seq)))
            res.append(tp)
        return chain.from_iterable(res)

    def predict(self, text: str, keep_vals: List[str] = None):
        """
        TDOO Text deprecated for now
        Given text, returns dictionary containing predictions from the BERT model
        in various useful forms. Explicitly,
        dict = {
            'processed_text':       Text cleaned by the BERTtokenizer pipe. I.e. the input fed to BERT.
            'entities':             List of entities found. The are triples (start, end, label).
            'entity_dict':          Dictionary of entities. Keys are labels, values are lists of entities of labels.
            'raw_bert_output':      The output from BertPredictor.predict(text).
            'token_label_pairs':    The BERT output, where wordpieces are compressed back into tokens. Easier on the eyes.
        }
        """
        text = str(text.encode("utf-8"), "utf-8")
        text = self.clean_text(text)
        res = self.pipeline(text)

        # TODO depends what we want to do  with the output. Should be another model  for our  usecase
        # fx use tokenizer to compose back into text
        return res


if __name__ == "__main__":
    text = "hello niels you little stupid fuck. Hi Emma you seem nice.  "
    bert_pred = BertTokenPredictor(model_name="bert_ner_test")

    bert_pred.predict(text)
