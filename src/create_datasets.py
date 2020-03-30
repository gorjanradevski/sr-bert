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


def create_datasets(
    dump_train_dataset_path: str,
    dump_val_dataset_path: str,
    dump_test_dataset_path: str,
    dump_visual2index_path: str,
    abstract_scenes_path: str,
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
            index, num_visuals = scenes.readline().split()
            index, num_visuals = int(index), int(num_visuals)
            scene = {"index": index}
            scene["elements"] = []
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
        if index in index2sentences:
            index2scene[index]["sentences"] = index2sentences[index]

    dataset = [
        index2scene[index]
        for index in index2scene.keys()
        if "sentences" in index2scene[index]
    ]
    # Generate train-val-test datasets
    cur_index = 0
    train_dataset = []
    val_dataset = []
    test_dataset = []
    same_index_scenes = []
    for scene in dataset:
        if scene["index"] != cur_index:
            train_dataset += same_index_scenes[:-2]
            val_dataset.append(same_index_scenes[-2])
            test_dataset.append(same_index_scenes[-1])
            same_index_scenes = []
            cur_index += 1
        same_index_scenes.append(scene)
    # Add the remaining ones
    train_dataset += same_index_scenes[:-2]
    val_dataset.append(same_index_scenes[-2])
    test_dataset.append(same_index_scenes[-1])

    # Delete the scenes that have no sentence available
    json.dump(train_dataset, open(dump_train_dataset_path, "w"))
    logger.info(f"Train dataset dumped {dump_train_dataset_path}")
    json.dump(val_dataset, open(dump_val_dataset_path, "w"))
    logger.info(f"Val dataset dumped {dump_val_dataset_path}")
    json.dump(test_dataset, open(dump_test_dataset_path, "w"))
    logger.info(f"Test dataset dumped {dump_test_dataset_path}")
    json.dump(dataset, open("data/datasets_new/full_dataset.json", "w"))

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
        "--dump_val_dataset_path",
        type=str,
        default="data/val_dataset.json",
        help="Where to dump the val dataset file.",
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

    return parser.parse_args()


def main():
    args = parse_args()
    create_datasets(
        args.dump_train_dataset_path,
        args.dump_val_dataset_path,
        args.dump_test_dataset_path,
        args.dump_visual2index_path,
        args.abstract_scenes_path,
    )


if __name__ == "__main__":
    main()
