import json
from transformers import BertTokenizer
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset(dump_datasets_path: str, abstract_scenes_path: str, train_size: int):
    sentences = [
        sentence
        for sentence in open(
            os.path.join(abstract_scenes_path, "Sentences_1002.txt"), "r"
        )
    ]
    dataset_train = []
    dataset_val = []
    with open(os.path.join(abstract_scenes_path, "Scenes_10020.txt")) as scenes:
        _ = scenes.readline()
        for i, sentence in enumerate(sentences):
            for j in range(10):
                scene = {}
                scene["sentence"] = sentence.rstrip("\n").lower()
                scene["elements"] = []
                num_visuals = int(scenes.readline().split()[-1])
                if num_visuals == 0:
                    logger.warning(f"Skipping scene {i},{j}")
                    continue
                for _ in range(num_visuals):
                    visual_name, _, _, x, y, z, flip = scenes.readline().split()
                    scene["elements"].append(
                        {
                            "visual_name": visual_name,
                            "x": int(x),
                            "y": int(y),
                            "z": int(z),
                            "flip": int(flip),
                        }
                    )
                if i < train_size:
                    dataset_train.append(scene)
                else:
                    dataset_val.append(scene)

    json.dump(
        dataset_train, open(os.path.join(dump_datasets_path, "dataset_train.json"), "w")
    )
    json.dump(
        dataset_val, open(os.path.join(dump_datasets_path, "dataset_val.json"), "w")
    )

    logger.info("Train and validation datasets dumped.")

    # Dump visual2index json file
    excluded = {
        "background.png",
        "selected.png",
        "buttons.png",
        "MikeJenny.png",
        "title.png",
    }
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    visual2index = {}
    index = 0
    pngs_file_path = os.path.join(abstract_scenes_path, "Pngs")
    for filename in os.listdir(pngs_file_path):
        if filename in excluded:
            continue
        filename, extension = filename.split(".")
        visual2index[filename + "_0_0" + "." + extension] = len(tokenizer) + index
        visual2index[filename + "_0_1" + "." + extension] = len(tokenizer) + index + 1
        visual2index[filename + "_1_0" + "." + extension] = len(tokenizer) + index + 2
        visual2index[filename + "_1_1" + "." + extension] = len(tokenizer) + index + 3
        visual2index[filename + "_2_0" + "." + extension] = len(tokenizer) + index + 4
        visual2index[filename + "_2_1" + "." + extension] = len(tokenizer) + index + 5
        index += 6
    json.dump(
        visual2index, open(os.path.join(dump_datasets_path, "visual2index.json"), "w")
    )

    logger.info("visual2index json file dumped.")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Creates a train and val json datasets."
    )
    parser.add_argument(
        "--dump_datasets_path",
        type=str,
        default="data/",
        help="Where to dump the dataset files.",
    )
    parser.add_argument(
        "--abstract_scenes_path",
        type=str,
        default="data/AbstractScenes_v1.1",
        help="Path to the abstract scenes dataset.",
    )
    parser.add_argument(
        "--train_size",
        type=str,
        default=902,
        help="Number of sentences to include in the train set.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    create_dataset(args.dump_datasets_path, args.abstract_scenes_path, args.train_size)


if __name__ == "__main__":
    main()
