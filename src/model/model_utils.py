from typing import List

import numpy as np


def train_test_split(texts: List[List], labels: List[List], test_size: float = 0.2, random_state: int = None):
    if 0 <= test_size <= 1:
        test_size = int(np.floor(test_size * len(texts)))

    test_idx = np.random.choice(len(texts), size=test_size, replace=False)
    texts = np.array(texts)
    labels = np.array(labels)

    texts_train, texts_test, labels_train, labels_test = (
        list(texts[~test_idx]),
        list(texts[test_idx]),
        list(labels[~test_idx]),
        list(labels[test_idx]),
    )

    return texts_train, texts_test, labels_train, labels_test


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)
    ]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if data_args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
