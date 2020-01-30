import json
import argparse
import logging
from nltk.stem import PorterStemmer
import nltk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_test_dataset(load_old_dataset_path: str, dump_new_dataset_path: str):
    stemmer = PorterStemmer()
    old_dataset = json.load(open(load_old_dataset_path))
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
    json.dump(new_dataset, open(dump_new_dataset_path, "w"))


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Creates a train and val json datasets."
    )
    parser.add_argument(
        "--dump_new_dataset_path",
        type=str,
        default="data/new_test_dataset.json",
        help="Where to dump the train dataset file.",
    )
    parser.add_argument(
        "--load_old_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Where to dump the test dataset file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    extract_test_dataset(args.load_old_dataset_path, args.dump_new_dataset_path)


if __name__ == "__main__":
    main()
