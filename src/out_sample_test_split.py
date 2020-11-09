import argparse
import json
import os
from copy import deepcopy

from tqdm import tqdm


def split_dataset(args):
    train_dataset = json.load(open(args.load_train_dataset_path))
    val_dataset = json.load(open(args.load_val_dataset_path))
    test_dataset = json.load(open(args.load_test_dataset_path))
    training_relations = set()
    for scene in tqdm(train_dataset):
        for relation_set in scene["relations"].values():
            training_relations.add("=".join(relation_set))

    dataset_splits = {
        "more_than_2": [],
        "more_than_2_oo_sample": [],
        "more_than_2_in_sample": [],
        "less_than_2": [],
    }
    for scene in tqdm(test_dataset):
        count = 0
        for relation_set in scene["relations"].values():
            relation = "=".join(relation_set)
            if relation not in training_relations:
                count += 1
        if count >= 2:
            dataset_splits["more_than_2"].append(scene)
            out_scene = deepcopy(scene)
            in_scene = deepcopy(scene)
            for name, relation_set in scene["relations"].items():
                relation = "=".join(relation_set)
                if relation in training_relations:
                    out_scene["relations"].pop(name)
                    out_scene["sentences"].pop(name)
                else:
                    in_scene["sentences"].pop(name)
                    in_scene["relations"].pop(name)
            dataset_splits["more_than_2_oo_sample"].append(out_scene)
            dataset_splits["more_than_2_in_sample"].append(in_scene)
        elif count < 2:
            dataset_splits["less_than_2"].append(scene)
        else:
            raise ValueError("Impossible!!")

    for split_name, split_scenes in dataset_splits.items():
        print(f"Dumping {split_name} with {len(split_scenes)} scenes")
        dump_path = os.path.join(
            args.dump_test_dataset_splits, f"test_dataset_unseen_{split_name}.json"
        )
        print(f"Dumping at {dump_path}")
        json.dump(split_scenes, open(dump_path, "w"))
        print("============================================")

    # Filtering the training and validation set
    more_than_2_relations = set()
    for scene in tqdm(dataset_splits["more_than_2"]):
        for relation_set in scene["relations"].values():
            more_than_2_relations.add("=".join(relation_set))

    # Filtering the training set
    dump_train_filtered_path = os.path.join(
        args.dump_test_dataset_splits, "train_dataset_filtered.json"
    )
    dump_filtered_set(more_than_2_relations, train_dataset, dump_train_filtered_path)

    # Filtering the validation set
    dump_val_filtered_path = os.path.join(
        args.dump_test_dataset_splits, "val_dataset_filtered.json"
    )
    dump_filtered_set(more_than_2_relations, val_dataset, dump_val_filtered_path)


def dump_filtered_set(relations, dataset, dump_path):
    filtered_dataset = []
    for scene in dataset:
        filtered_scene = deepcopy(scene)
        for name, relation_set in scene["relations"].items():
            relation = "=".join(relation_set)
            if relation in relations:
                filtered_scene["sentences"].pop(name)
                filtered_scene["relations"].pop(name)

        filtered_dataset.append(filtered_scene)

    print(f"Dumping at {dump_path}")
    json.dump(filtered_dataset, open(dump_path, "w"))
    print("============================================")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Creates a oo-split split of the test set."
    )
    parser.add_argument(
        "--load_train_dataset_path",
        type=str,
        default="data/train_dataset.json",
        help="Path to the full train dataset.",
    )
    parser.add_argument(
        "--load_val_dataset_path",
        type=str,
        default="data/val_dataset.json",
        help="Path to the full val dataset.",
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
    split_dataset(args)


if __name__ == "__main__":
    main()
