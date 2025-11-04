

import json
from lib.utils.constants import GOLDEN_DATASET_PATH


def load_evaluation_dataset():
    with open(GOLDEN_DATASET_PATH, "r") as f:
        golden_dataset = json.load(f)

    return golden_dataset["test_cases"]
