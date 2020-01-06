from torch.utils.data import Dataset
import json
from transformers import BertTokenizer
import torch
import numpy as np

from constants import MAX_WIDTH, MAX_HEIGHT


class ScenesDataset(Dataset):
    def __init__(
        self, dataset_file_path: str, visual2index_path: str, mask_probability: float
    ):
        self.dataset_file = json.load(open(dataset_file_path))
        self.visual2index = json.load(open(visual2index_path))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.mask_token = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.sep_token = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.dataset_file)

    def __getitem__(self, idx: int):
        # Obtain elements
        scene = self.dataset_file[idx]
        tokenized_sentence = self.tokenizer.encode(
            scene["sentence"], add_special_tokens=True
        )
        tokenized_visuals = [
            self.visual2index[
                element["visual_name"].split(".")[0]
                + "_"
                + str(element["z"])
                + "_"
                + str(element["flip"])
                + "."
                + element["visual_name"].split(".")[-1]
            ]
            for element in scene["elements"]
        ]
        tokenized_visuals.append(self.sep_token)
        # Generate mask and skip SEP token
        mask = [
            np.random.choice(
                [0, 1], p=[1 - self.mask_probability, self.mask_probability]
            )
            if token != self.sep_token
            else 0
            for token in tokenized_visuals
        ]
        # Mask elements
        masked_visuals = [
            element if m == 0 else self.mask_token
            for m, element in zip(mask, tokenized_visuals)
        ]
        # Combine full sequence
        input_ids = torch.tensor(tokenized_sentence + masked_visuals)
        # Generate masked labels
        masked_lm_labels = [
            masked_visual.item() if input_id == self.mask_token else -1
            for input_id, masked_visual in zip(
                input_ids, torch.tensor(tokenized_sentence + tokenized_visuals)
            )
        ]
        # Obtain and normalize positions
        positions = [
            [element["x"] / MAX_WIDTH, element["y"] / MAX_HEIGHT]
            for element in scene["elements"]
        ]

        return input_ids, masked_lm_labels, positions
