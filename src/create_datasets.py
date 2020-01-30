import json
import os
import argparse
import logging
from typing import Dict
from transformers import BertTokenizer
from tqdm import tqdm
import nltk
from nltk.stem import PorterStemmer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_relations(
    abstract_scenes_path: str,
    relations_file_name: str,
    index2relations: Dict[int, str] = None,
):
    if index2relations is None:
        index2relations = {}
    with open(
        os.path.join(
            abstract_scenes_path,
            os.path.join(
                "SimpleSentences", os.path.join("tuples", relations_file_name)
            ),
        )
    ) as relations:
        for line in relations:
            scene_index = int(line.split("\t")[0])
            sentence_index = int(line.split("\t")[1])
            if scene_index not in index2relations:
                index2relations[scene_index] = {}
            if sentence_index not in index2relations[scene_index]:
                index2relations[scene_index][sentence_index] = []
            relations = [
                relation.rstrip("\n").rstrip().lower()
                for relation in line.split("\t")[2:]
            ]
            index2relations[scene_index][sentence_index] += relations

    return index2relations


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


def reformat_dataset(old_dataset):
    stemmer = PorterStemmer()
    new_dataset = []
    for scene in tqdm(old_dataset):
        for sentence_index in scene["sentences"].keys():
            if sentence_index in scene["relations"]:
                relations = scene["relations"][sentence_index]
                for sentence in scene["sentences"][sentence_index]:
                    potential_masked_words = [
                        word
                        for word in nltk.word_tokenize(sentence)
                        if stemmer.stem(word) in relations or word in relations
                    ]
                    for masked_word in potential_masked_words:
                        test_sentence = []
                        count = 0
                        for word in nltk.word_tokenize(sentence):
                            if word != masked_word:
                                test_sentence.append(word)
                            else:
                                count += 1
                                test_sentence.append("[MASK]")
                        if count == 1:
                            new_dataset.append(
                                {
                                    "sentence": " ".join(test_sentence),
                                    "label": masked_word,
                                    "relations": scene["relations"][sentence_index],
                                    "elements": scene["elements"],
                                }
                            )
    return new_dataset


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
    index2relations = parse_relations(abstract_scenes_path, "TuplesText1_10020.txt")
    index2relations = parse_relations(
        abstract_scenes_path, "TuplesText2_10020.txt", index2relations
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
        if index in index2sentences:
            index2scene[index]["sentences"] = index2sentences[index]

    # Combine the scenes and the relations
    for index in index2scene.keys():
        if index in index2relations:
            index2scene[index]["relations"] = index2relations[index]

    dataset = [
        index2scene[index]
        for index in index2scene.keys()
        if "sentences" in index2scene[index]
    ]

    train_dataset = dataset[:-test_size]
    logger.info("Reformating test dataset...")
    test_dataset = reformat_dataset(dataset[-test_size:])

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
        }
        visual2index = {}
        index = len(BertTokenizer.from_pretrained("bert-base-uncased"))
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
