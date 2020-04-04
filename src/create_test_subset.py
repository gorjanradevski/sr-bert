import json
import argparse
from tqdm import tqdm
import re

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


def split_dataset(
    load_test_dataset_path: str, dump_test_dataset_path: str, explicit_rels_ratio: float
):
    test_dataset = json.load(open(load_test_dataset_path))
    dump_test_dataset = []
    for scene in tqdm(test_dataset):
        total = 0
        with_explicit = 0
        for sentences in scene["sentences"].values():
            for sentence in sentences:
                total += 1
                for relation in explicit_rels:
                    if contains_word(relation, sentence):
                        with_explicit += 1
                        break
        if with_explicit / total > explicit_rels_ratio:
            dump_test_dataset.append(scene)
            continue

    print(f"The size of the filtered dataset is {len(dump_test_dataset)}")
    print(f"Dumping at {dump_test_dataset_path}")
    json.dump(dump_test_dataset, open(dump_test_dataset_path, "w"))


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Creates a split of the test set.")
    parser.add_argument(
        "--load_test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the full test dataset.",
    )
    parser.add_argument(
        "--dump_test_dataset_path",
        type=str,
        default="data/test_dataset_split.json",
        help="Path to the split of the test dataset.",
    )
    parser.add_argument(
        "--explicit_rels_ratio",
        type=float,
        default=0.5,
        help="The ratio of explicit relations.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    split_dataset(
        args.load_test_dataset_path,
        args.dump_test_dataset_path,
        args.explicit_rels_ratio,
    )


if __name__ == "__main__":
    main()
