import json
import argparse
from tqdm import tqdm


def split_dataset(
    load_train_dataset_path: str,
    load_test_dataset_path: str,
    dump_test_present_dataset_path: str,
    dump_test_absent_dataset_path: str,
):
    train_dataset = json.load(open(load_train_dataset_path))
    test_dataset = json.load(open(load_test_dataset_path))
    dump_present_test_dataset = []
    dump_absent_test_dataset = []
    training_relations = set()
    for scene in tqdm(train_dataset):
        for relation_set in scene["relations"].values():
            for relation in relation_set:
                training_relations.add(relation)

    for scene in tqdm(test_dataset):
        total_relations = 0
        train_contained_relations = 0
        for relation_set in scene["relations"].values():
            for relation in relation_set:
                if relation in training_relations:
                    train_contained_relations += 1
                total_relations += 1

        if train_contained_relations / total_relations <= 0.5:
            dump_absent_test_dataset.append(scene)
        else:
            dump_present_test_dataset.append(scene)

    print(
        f"The size of the present relations dataset is {len(dump_present_test_dataset)}"
    )
    print(f"Dumping at {dump_test_present_dataset_path}")
    json.dump(dump_present_test_dataset, open(dump_test_present_dataset_path, "w"))
    print("============================================")
    print(
        f"The size of the absent relations dataset is {len(dump_absent_test_dataset)}"
    )
    print(f"Dumping at {dump_test_absent_dataset_path}")
    json.dump(dump_absent_test_dataset, open(dump_test_absent_dataset_path, "w"))


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
        "--dump_test_present_dataset_path",
        type=str,
        default="data/datasets_new/test_dataset_present.json",
        help="Path to the present relations split of the test dataset.",
    )
    parser.add_argument(
        "--dump_test_absent_dataset_path",
        type=str,
        default="data/datasets_new/test_dataset_absent.json",
        help="Path to the absent relations split of the test dataset.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    split_dataset(
        args.load_train_dataset_path,
        args.load_test_dataset_path,
        args.dump_test_present_dataset_path,
        args.dump_test_absent_dataset_path,
    )


if __name__ == "__main__":
    main()
