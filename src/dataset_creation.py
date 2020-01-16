import json
import os
import argparse
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_sentences(
    abstract_scenes_path: str,
    sententences_file_name: str,
    index2sentence: Dict[int, str] = None,
):
    if index2sentence is None:
        index2sentence = {}
    with open(
        os.path.join(
            abstract_scenes_path,
            os.path.join("SimpleSentences", sententences_file_name),
        )
    ) as sentences:
        lines = sentences.readlines()
        for line in lines:
            if line == "\n":
                continue
            index = int(line.split("\t")[0])
            if index not in index2sentence:
                index2sentence[index] = ""
            index2sentence[index] += line.split("\t")[-1].rstrip("\n").lower()

    return index2sentence


def create_dataset(
    dump_dataset_path: str,
    dump_visual2index_path: str,
    abstract_scenes_path: str,
    increase_sentences: bool,
):
    index2sentence = parse_sentences(abstract_scenes_path, "SimpleSentences1_10020.txt")
    if increase_sentences:
        index2sentence = parse_sentences(
            abstract_scenes_path, "SimpleSentences2_10020.txt", index2sentence
        )
    index2scene = {}
    with open(os.path.join(abstract_scenes_path, "Scenes_10020.txt")) as scenes:
        _ = scenes.readline()
        for i in range(10020):
            scene = {}
            scene["elements"] = []
            num_visuals = int(scenes.readline().split()[-1])
            if num_visuals == 0:
                print(f"Skipping scene {i}")
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
            index2scene[i] = scene

    # Combine the scenes and sentences
    for index in index2scene.keys():
        if index in index2sentence:
            index2scene[index]["sentence"] = index2sentence[index]

    dataset = [
        index2scene[index]
        for index in index2scene.keys()
        if "sentence" in index2scene[index]
    ]
    # Delete the scenes that have no sentence available
    json.dump(dataset, open(dump_dataset_path, "w"))

    logger.info("Dataset dumped.")

    # Dump visual2index json file
    if dump_visual2index_path is not None:
        excluded = {
            "background.png",
            "selected.png",
            "buttons.png",
            "MikeJenny.png",
            "title.png",
        }
        visual2index = {}
        index = 3  # Starting from 3 because 0 = PAD, 1 = SEP, 2 = MASK
        pngs_file_path = os.path.join(abstract_scenes_path, "Pngs")
        for filename in os.listdir(pngs_file_path):
            if filename in excluded:
                continue
            visual2index[filename] = index
            index += 1
        json.dump(visual2index, open(dump_visual2index_path, "w"))

        logger.info("Visual2index json file dumped.")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Creates a train and val json datasets."
    )
    parser.add_argument(
        "--dump_dataset_path",
        type=str,
        default="data/dataset.json",
        help="Where to dump the dataset file.",
    )
    parser.add_argument(
        "--dump_visual2index_path",
        type=str,
        default=None,
        help="Where to dump the visual2index file.",
    )
    parser.add_argument(
        "--abstract_scenes_path",
        type=str,
        default="data/AbstractScenes_v1.1",
        help="Path to the abstract scenes dataset.",
    )
    parser.add_argument(
        "--increase_sentences", action="store_true", help="Generate visual2index"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    create_dataset(
        args.dump_dataset_path,
        args.dump_visual2index_path,
        args.abstract_scenes_path,
        args.increase_sentences,
    )


if __name__ == "__main__":
    main()
