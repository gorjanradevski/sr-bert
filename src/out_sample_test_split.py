import json
import argparse
from tqdm import tqdm
import os
from copy import deepcopy


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
            training_relations.add("=".join(relation_set))

    dataset_splits = {"more_than_2": [], "more_than_2_filtered": [], "less_than_2": []}
    for scene in tqdm(test_dataset):
        count = 0
        # total = 0
        for relation_set in scene["relations"].values():
            relation = "=".join(relation_set)
            if relation not in training_relations:
                count += 1
        if count >= 2:
            dataset_splits["more_than_2"].append(scene)
            copy_scene = deepcopy(scene)
            for name, relation_set in copy_scene["relations"].items():
                relation = "=".join(relation_set)
                if relation in training_relations:
                    copy_scene["sentences"].pop(name)
            dataset_splits["more_than_2_filtered"].append(copy_scene)
        elif count < 2:
            dataset_splits["less_than_2"].append(scene)
        else:
            raise ValueError("Impossible!!")

    for split_name, split_scenes in dataset_splits.items():
        print(f"Dumping {split_name} with {len(split_scenes)} scenes")
        dump_path = os.path.join(
            dump_test_dataset_splits, f"test_dataset_unseen_{split_name}.json"
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
        default="data/train_dataset.json",
        help="Path to the full train dataset.",
    )
    parser.add_argument(
        "--load_test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the full test dataset.",
    )
    parser.add_argument(
        "--dump_test_dataset_splits",
        type=str,
        default="data/specific_splits",
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
