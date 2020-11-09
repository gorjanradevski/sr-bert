import json
from typing import Dict, Tuple

import nltk
import numpy as np
import torch
from scene_layouts.datasets import BUCKET_SIZE, X_MASK, Y_MASK
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

MAX_FREQ = 5


def build_vocab(dataset_file):
    # Obtain word frequencies
    word2freq = {}
    for scene in tqdm(dataset_file):
        for sentence in scene["sentences"].values():
            for word in nltk.word_tokenize(sentence[0]):
                if word not in word2freq:
                    word2freq[word] = 0
                word2freq[word] += 1
    # Obtain word2index
    word2index = {"<pad>": 0, "<unk>": 1}
    index = 2
    for scene in tqdm(dataset_file):
        for sentence in scene["sentences"].values():
            for word in nltk.word_tokenize(sentence[0]):
                if word2freq[word] < MAX_FREQ or word in word2index:
                    continue
                word2index[word] = index
                index += 1
    index2word = {v: k for k, v in word2index.items()}

    return word2freq, word2index, index2word


class Dataset:
    def __init__(
        self, json_path: str, word2freq: Dict, word2index: Dict, visual2index: Dict
    ):
        self.dataset_file = json.load(open(json_path))
        self.word2freq = word2freq
        self.word2index = word2index
        self.visual2index = visual2index

    def tokenize_sentence(self, sentence):
        return [
            self.word2index[word]
            if word in self.word2index
            else self.word2index["<unk>"]
            for word in nltk.word_tokenize(sentence)
        ]

    def __len__(self):
        return len(self.dataset_file)

    def __getitem__(self, idx: int):
        # Obtain elements
        scene = self.dataset_file[idx]
        # Prepare sentences
        nested_sentences = [sentences for sentences in scene["sentences"].values()]
        sentences = [sentence for sublist in nested_sentences for sentence in sublist]
        input_ids_sentence = torch.tensor(self.tokenize_sentence(" ".join(sentences)))
        # Prepare visuals
        input_ids_visuals = torch.tensor(
            [self.visual2index[element["visual_name"]] for element in scene["elements"]]
        )

        return input_ids_sentence, input_ids_visuals, scene["elements"]


class TrainDataset(Dataset, TorchDataset):
    def __init__(
        self, json_path: str, word2freq: Dict, word2index: Dict, visual2index: Dict
    ):
        super().__init__(json_path, word2freq, word2index, visual2index)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, scene_elements = super().__getitem__(idx)
        x_labels = torch.tensor(
            [
                0
                if element["x"] < 0
                else X_MASK - 1
                if element["x"] > (X_MASK - 1) * BUCKET_SIZE
                else np.floor(element["x"] / BUCKET_SIZE)
                for element in scene_elements
            ],
            dtype=torch.long,
        )
        # Obtain Y-indexes
        y_labels = torch.tensor(
            [
                0
                if element["y"] < 0
                else Y_MASK - 1
                if element["y"] > (Y_MASK - 1) * BUCKET_SIZE
                else np.floor(element["y"] / BUCKET_SIZE)
                for element in scene_elements
            ],
            dtype=torch.long,
        )
        # Obtain flips
        o_labels = torch.tensor([element["flip"] for element in scene_elements])

        return input_ids_sentence, input_ids_visuals, x_labels, y_labels, o_labels


class InferenceDataset(Dataset, TorchDataset):
    def __init__(
        self, json_path: str, word2freq: Dict, word2index: Dict, visual2index: Dict
    ):
        super().__init__(json_path, word2freq, word2index, visual2index)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, scene_elements = super().__getitem__(idx)
        x_labels = torch.tensor(
            [
                0
                if element["x"] < 0
                else (X_MASK - 1) * BUCKET_SIZE
                if element["x"] > (X_MASK - 1) * BUCKET_SIZE
                else element["x"]
                for element in scene_elements
            ],
            dtype=torch.long,
        )
        # Obtain Y-indexes
        y_labels = torch.tensor(
            [
                0
                if element["y"] < 0
                else (Y_MASK - 1) * BUCKET_SIZE
                if element["y"] > (Y_MASK - 1) * BUCKET_SIZE
                else element["y"]
                for element in scene_elements
            ],
            dtype=torch.long,
        )
        # Obtain flips
        o_labels = torch.tensor([element["flip"] for element in scene_elements])

        return input_ids_sentence, input_ids_visuals, x_labels, y_labels, o_labels


def collate_pad_batch(
    batch: Tuple[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
    ]
) -> Dict[str, torch.Tensor]:
    ids_text, ids_vis, x_labs, y_labs, o_labs = zip(*batch)
    # Pad the sentences
    ids_text = torch.nn.utils.rnn.pad_sequence(
        ids_text, batch_first=True, padding_value=0
    )
    # Pad the visuals
    ids_vis = torch.nn.utils.rnn.pad_sequence(
        ids_vis, batch_first=True, padding_value=0
    )
    # Pad the visual mappings and prepare final mappings
    x_labs = torch.nn.utils.rnn.pad_sequence(
        x_labs, batch_first=True, padding_value=-100
    )
    y_labs = torch.nn.utils.rnn.pad_sequence(
        y_labs, batch_first=True, padding_value=-100
    )
    o_labs = torch.nn.utils.rnn.pad_sequence(
        o_labs, batch_first=True, padding_value=-100
    )
    # Obtain the padding mask
    mask = ids_vis.clone()
    mask[torch.where(mask > 0)] = 1

    return {
        "ids_text": ids_text,
        "ids_vis": ids_vis,
        "x_labs": x_labs,
        "y_labs": y_labs,
        "o_labs": o_labs,
        "mask": mask,
    }
