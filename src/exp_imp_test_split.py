import json
import argparse
from tqdm import tqdm
import re
import os

# https://academicguides.waldenu.edu/writingcenter/grammar/prepositions
# https://github.com/gcollell/spatial-commonsense/blob/master/code/pre-process_data.py#L32
explicit_rels = [
    "on",
    "next to",
    "above",
    "over",
    "below",
    "behind",
    "along",
    "through",
    "in",
    "in front of",
    "near",
    "beyond",
    "with",
    "by",
    "inside of",
    "on top of",
    "down",
    "up",
    "beneath",
    "inside",
    "left",
    "right",
    "under",
    "across from",
    "underneath",
    "atop",
    "across",
    "beside",
    "around",
    "outside",
    "next",
    "against",
    "at",
    "between",
    "front",
    "aside",
    "adjacent",
]


def contains_word(word, text):
    # https://stackoverflow.com/a/45587730/3987085
    pattern = r"(^|[^\w]){}([^\w]|$)".format(word)
    pattern = re.compile(pattern, re.IGNORECASE)
    matches = re.search(pattern, text)
    return bool(matches)


def split_dataset(load_test_dataset_path: str, dump_test_dataset_splits: str):
    test_dataset = json.load(open(load_test_dataset_path))
    dataset_splits = {
        "explicit_0-25": [],
        "explicit_25-50": [],
        "explicit_50-75": [],
        "explicit_75-100": [],
    }
    for scene in tqdm(test_dataset):
        total = 0
        with_explicit = 0
        for sentences_set in scene["sentences"].values():
            for sentence in sentences_set:
                total += 1
                for relation in explicit_rels:
                    if contains_word(relation, sentence):
                        with_explicit += 1
                        break

        ratio = with_explicit / total
        if ratio <= 0.25:
            dataset_splits["explicit_0-25"].append(scene)
        elif ratio > 0.25 and ratio <= 0.5:
            dataset_splits["explicit_25-50"].append(scene)
        elif ratio > 0.5 and ratio <= 0.75:
            dataset_splits["explicit_50-75"].append(scene)
        elif ratio > 0.75 and ratio <= 1.0:
            dataset_splits["explicit_75-100"].append(scene)
        else:
            raise ValueError("Impossible!!")

    for split_name, split_scenes in dataset_splits.items():
        print(f"Dumping {split_name} with {len(split_scenes)} scenes")
        dump_path = os.path.join(
            dump_test_dataset_splits, f"test_dataset_{split_name}.json"
        )
        print(f"Dumping at {dump_path}")
        json.dump(split_scenes, open(dump_path, "w"))
        print("============================================")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Creates a split of the test set.")
    parser.add_argument(
        "--load_test_dataset_path",
        type=str,
        default="data/datasets_new/test_dataset.json",
        help="Path to the full test dataset.",
    )
    parser.add_argument(
        "--dump_test_dataset_splits",
        type=str,
        default="data/datasets_new/",
        help="Where to dump the dataset splits.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    split_dataset(args.load_test_dataset_path, args.dump_test_dataset_splits)


if __name__ == "__main__":
    main()
