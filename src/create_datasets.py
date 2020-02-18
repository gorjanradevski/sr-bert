import json
import os
import argparse
import logging
from typing import Dict

from constants import remapped

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_sentences(
    abstract_scenes_path: str,
    sententences_file_name: str,
    index2sentences: Dict[int, str] = None,
):
    if index2sentences is None:
        index2sentences = {}
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
            scene_index = int(line.split("\t")[0])
            sentence_index = int(line.split("\t")[1])
            if scene_index not in index2sentences:
                index2sentences[scene_index] = {}
            if sentence_index not in index2sentences[scene_index]:
                index2sentences[scene_index][sentence_index] = []
            sentence = line.split("\t")[-1].rstrip("\n").rstrip().lower()
            if not sentence.endswith((".", "!", "?", '"')):
                sentence += "."
            index2sentences[scene_index][sentence_index].append(sentence)

    return index2sentences


def flip_scene(scene):
    scene["elements"] = [
        {
            "visual_name": element["visual_name"],
            "x": abs(500 - element["x"]),
            "y": element["y"],
            "z": element["z"],
            "flip": abs(1 - element["flip"]),
        }
        for element in scene["elements"]
    ]

    return scene


def detect_ambiguity(scene):
    for element in scene["elements"]:
        if element["visual_name"].startswith("hb0"):
            mike_element = element
        elif element["visual_name"].startswith("hb1"):
            jenny_element = element
        else:
            continue

    if "jenny_element" not in locals() or "mike_element" not in locals():
        return False
    if mike_element["x"] > jenny_element["x"]:
        return True

    return False


def create_datasets(
    dump_train_dataset_path: str,
    dump_test_dataset_path: str,
    dump_visual2index_path: str,
    abstract_scenes_path: str,
    test_size: int,
):
    index2sentences = parse_sentences(
        abstract_scenes_path, "SimpleSentences1_10020.txt"
    )
    index2sentences = parse_sentences(
        abstract_scenes_path, "SimpleSentences2_10020.txt", index2sentences
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
                if visual_name in remapped:
                    visual_name = remapped[visual_name]
                scene["elements"].append(
                    {
                        "visual_name": visual_name,
                        "x": int(x),
                        "y": int(y),
                        "z": int(z),
                        "flip": int(flip),
                    }
                )
            if detect_ambiguity(scene):
                scene = flip_scene(scene)
            index2scene[i] = scene

    # Combine the scenes and sentences
    for index in index2scene.keys():
        if index in index2sentences:
            index2scene[index]["sentences"] = index2sentences[index]

    dataset = [
        index2scene[index]
        for index in index2scene.keys()
        if "sentences" in index2scene[index]
    ]

    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]
    # Delete the scenes that have no sentence available
    json.dump(train_dataset, open(dump_train_dataset_path, "w"))
    logger.info(f"Train dataset dumped {dump_train_dataset_path}")
    json.dump(test_dataset, open(dump_test_dataset_path, "w"))
    logger.info(f"Test dataset dumped {dump_test_dataset_path}")

    # Dump visual2index json file
    if dump_visual2index_path is not None:
        excluded = {
            "background.png",
            "selected.png",
            "buttons.png",
            "MikeJenny.png",
            "title.png",
            "hb0_0s.png",
            "hb0_1s.png",
            "hb0_3s.png",
            "hb0_4s.png",
            "hb0_5s.png",
            "hb0_6s.png",
            "hb0_8s.png",
            "hb0_9s.png",
            "hb0_10s.png",
            "hb0_11s.png",
            "hb0_13s.png",
            "hb0_14s.png",
            "hb0_15s.png",
            "hb0_16s.png",
            "hb0_18s.png",
            "hb0_19s.png",
            "hb0_20s.png",
            "hb0_21s.png",
            "hb0_23s.png",
            "hb0_24s.png",
            "hb0_25s.png",
            "hb0_26s.png",
            "hb0_28s.png",
            "hb0_29s.png",
            "hb0_30s.png",
            "hb0_31s.png",
            "hb0_33s.png",
            "hb0_34s.png",
            "hb1_0s.png",
            "hb1_1s.png",
            "hb1_3s.png",
            "hb1_4s.png",
            "hb1_5s.png",
            "hb1_6s.png",
            "hb1_8s.png",
            "hb1_9s.png",
            "hb1_10s.png",
            "hb1_11s.png",
            "hb1_13s.png",
            "hb1_14s.png",
            "hb1_15s.png",
            "hb1_16s.png",
            "hb1_18s.png",
            "hb1_19s.png",
            "hb1_20s.png",
            "hb1_21s.png",
            "hb1_23s.png",
            "hb1_24s.png",
            "hb1_25s.png",
            "hb1_26s.png",
            "hb1_28s.png",
            "hb1_29s.png",
            "hb1_30s.png",
            "hb1_31s.png",
            "hb1_33s.png",
            "hb1_34s.png",
        }
        visual2index = {}
        index = 1
        pngs_file_path = os.path.join(abstract_scenes_path, "Pngs")
        for filename in sorted(os.listdir(pngs_file_path)):
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
        "--dump_train_dataset_path",
        type=str,
        default="data/train_dataset.json",
        help="Where to dump the train dataset file.",
    )
    parser.add_argument(
        "--dump_test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Where to dump the test dataset file.",
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
        "--test_size", type=int, default=500, help="Size of the test set."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    create_datasets(
        args.dump_train_dataset_path,
        args.dump_test_dataset_path,
        args.dump_visual2index_path,
        args.abstract_scenes_path,
        args.test_size,
    )


if __name__ == "__main__":
    main()
