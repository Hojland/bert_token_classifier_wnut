import json
import sys
from pathlib import Path
from zipfile import ZipFile

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel

from cachetools import LRUCache, cached
from common.bert_predict import BertSequencePredictor
from fastapi import APIRouter, HTTPException

router = APIRouter()

cache = LRUCache(maxsize=4)


class PredictInput(BaseModel):
    text: str
    service: str
    model: str


@router.post("/")
def predict_service(_input: PredictInput):

    # Fetch model
    try:
        model = get_model(_input.service, _input.model)
    except OSError as e:
        cache.clear()
        raise HTTPException(status_code=500, detail=e.strerror)

    # Predict on text
    prediction = model.predict(_input.text)

    # Return prediction result
    result = {"text": _input.text, "prediction": prediction}

    return result


@router.get("/clear_cache/")
def clear_cache():
    cache.clear()
    return True


@cached(cache=cache)
def get_model(service: str, model: str):

    # Get path to model
    model_path = Path("data", service, model)
    model_path = model_path.parent / model_path.stem

    # Check that model is downloaded
    if not model_path.exists():
        # Download model from S3
        pull_from_s3(service, model)

    # Build predictor object
    predictor_class = determine_predictor_class(model_path)
    model = predictor_class(str(model_path))

    return model


def determine_predictor_class(model_path: Path):

    # Read model config as json
    config_file = model_path / "model_config.json"
    with config_file.open() as f:
        model_config = json.load(f)

    # Return class based on model config
    predictor_class = getattr(sys.modules[__name__], model_config["class"])

    return predictor_class


def pull_from_s3(service_folder: str, object_name: str):

    # Connect to S3 client
    s3 = boto3.client("s3")

    # Make file paths to S3 and local objects
    s3_object_path = Path(service_folder, object_name)
    local_object_path = Path("data", service_folder, object_name)

    # Make directory for model files
    Path("data", service_folder).mkdir(parents=True, exist_ok=True)

    # Get folder containing model
    try:
        # Download zip file from S3
        s3.download_file(Bucket="bert", Key=str(s3_object_path), Filename=str(local_object_path))
    except ClientError as e:
        raise HTTPException(status_code=e.response["ResponseMetadata"]["HTTPStatusCode"], detail=e.response["Error"]["Message"])

    # Unzip model folder
    with ZipFile(local_object_path) as f:
        f.extractall(local_object_path.parent / local_object_path.stem)  # Save model to folder without .zip extension

    # Delete zip folder from disk
    Path.unlink(local_object_path)
