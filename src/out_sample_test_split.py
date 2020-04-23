import json
import argparse
from tqdm import tqdm
import os


def split_dataset(
    load_train_dataset_path: str,
    load_test_dataset_path: str,
    dump_test_dataset_splits: str,
):
    train_dataset = json.load(open(load_train_dataset_path))
    test_dataset = json.load(open(load_test_dataset_path))
    training_relations = set()
    for scene in tqdm(train_dataset):
        for relation_set in scene["relations"].values():
            for relation in relation_set:
                training_relations.add(relation)

    dataset_splits = {"0-25": [], "25-50": [], "50-75": [], "75-100": []}
    for scene in tqdm(test_dataset):
        count = 0
        total = 0
        for relation_set in scene["relations"].values():
            for relation in relation_set:
                if relation not in training_relations:
                    count += 1
                total += 1
        ratio = count / total
        if ratio <= 0.25:
            dataset_splits["0-25"].append(scene)
        elif ratio > 0.25 and ratio <= 0.5:
            dataset_splits["25-50"].append(scene)
        elif ratio > 0.5 and ratio <= 0.75:
            dataset_splits["50-75"].append(scene)
        elif ratio > 0.75 and ratio <= 1.0:
            dataset_splits["75-100"].append(scene)
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
        "--load_train_dataset_path",
        type=str,
        default="data/datasets_new/train_dataset.json",
        help="Path to the full train dataset.",
    )
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
    split_dataset(
        args.load_train_dataset_path,
        args.load_test_dataset_path,
        args.dump_test_dataset_splits,
    )


if __name__ == "__main__":
    main()
