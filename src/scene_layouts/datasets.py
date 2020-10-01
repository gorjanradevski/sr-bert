from torch.utils.data import Dataset as TorchDataset
import json
from transformers import BertTokenizer
import torch
import numpy as np
from typing import Tuple, Dict

BUCKET_SIZE = 20
X_MASK = 500 // BUCKET_SIZE + 1
X_PAD = 500 // BUCKET_SIZE + 2
Y_MASK = 400 // BUCKET_SIZE + 1
Y_PAD = 400 // BUCKET_SIZE + 2
O_MASK = 2
O_PAD = 3
SCENE_WIDTH_TRAIN = 500 // BUCKET_SIZE
SCENE_WIDTH_TEST = 500


class TrainDataset:
    def __init__(
        self, dataset_file_path: str, visual2index: Dict, mask_probability: float
    ):
        self.dataset_file = json.load(open(dataset_file_path))
        self.visual2index = visual2index
        self.mask_probability = mask_probability
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    @staticmethod
    def flip_scene(x_indexes: torch.Tensor, o_indexes: torch.Tensor):
        return torch.abs(SCENE_WIDTH_TRAIN - x_indexes), torch.abs(1 - o_indexes)

    @staticmethod
    def move_scene(x_indexes: torch.Tensor, y_indexes: torch.Tensor):
        # Left of right; True - left, False - right
        if torch.bernoulli(torch.tensor([0.5])).bool().item():
            # If left, find the minimum movement
            move_magnitude = torch.randint(0, x_indexes.min().item() + 1, size=(1,))
            x_indexes_moved = x_indexes.clone() - move_magnitude
        else:
            # If right, find the maximum movement
            move_magnitude = torch.randint(
                0, 500 // BUCKET_SIZE - x_indexes.max().item() + 1, size=(1,)
            )
            x_indexes_moved = x_indexes.clone() + move_magnitude

        # Up or down; True - up, False - down
        if torch.bernoulli(torch.tensor([0.5])).bool().item():
            # If top, find the minimum movement
            move_magnitude = torch.randint(0, y_indexes.min().item() + 1, size=(1,))
            y_indexes_moved = y_indexes.clone() - move_magnitude
        else:
            # If down, find the maximum movement
            move_magnitude = torch.randint(
                0, 400 // BUCKET_SIZE - y_indexes.max().item() + 1, size=(1,)
            )
            y_indexes_moved = y_indexes.clone() + move_magnitude

        return x_indexes_moved, y_indexes_moved

    def __len__(self):
        return len(self.dataset_file)

    def __getitem__(self, idx: int):
        # Obtain elements
        scene = self.dataset_file[idx]
        # Prepare sentences
        nested_sentences = [sentences for sentences in scene["sentences"].values()]
        sentences = np.random.permutation(
            np.array([sentence for sublist in nested_sentences for sentence in sublist])
        )

        input_ids_sentence = torch.tensor(
            self.tokenizer.encode(" ".join(sentences), add_special_tokens=True)
        )
        # Prepare visuals
        input_ids_visuals = torch.tensor(
            [self.visual2index[element["visual_name"]] for element in scene["elements"]]
        )
        # Obtain X-indexes
        x_indexes = torch.tensor(
            [
                0
                if element["x"] < 0
                else X_MASK - 1
                if element["x"] > (X_MASK - 1) * BUCKET_SIZE
                else np.floor(element["x"] / BUCKET_SIZE)
                for element in scene["elements"]
            ],
            dtype=torch.long,
        )
        # Obtain Y-indexes
        y_indexes = torch.tensor(
            [
                0
                if element["y"] < 0
                else Y_MASK - 1
                if element["y"] > (Y_MASK - 1) * BUCKET_SIZE
                else np.floor(element["y"] / BUCKET_SIZE)
                for element in scene["elements"]
            ],
            dtype=torch.long,
        )
        # Obtain flips
        o_indexes = torch.tensor([element["flip"] for element in scene["elements"]])
        # Flip scene with 50% prob
        if torch.bernoulli(torch.tensor([0.5])).bool().item():
            x_indexes, o_indexes = self.flip_scene(x_indexes, o_indexes)

        # Move scene with 50% prob
        if torch.bernoulli(torch.tensor([0.5])).bool().item():
            x_indexes, y_indexes = self.move_scene(x_indexes, y_indexes)

        return input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, o_indexes

    def masking(self, x_indexes, y_indexes, o_indexes):
        # https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py#L169
        # Create clones for everything
        x_labels = x_indexes.clone()
        y_labels = y_indexes.clone()
        o_labels = o_indexes.clone()
        # Get probability matrix
        probability_matrix = torch.full(x_indexes.shape, self.mask_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # We only compute loss on masked tokens
        x_labels[~masked_indices] = -100
        y_labels[~masked_indices] = -100
        o_labels[~masked_indices] = -100
        # 80% we replace with a mask token
        indices_replaced = (
            torch.bernoulli(torch.full(x_indexes.shape, 0.8)).bool() & masked_indices
        )
        x_indexes[indices_replaced] = X_MASK
        y_indexes[indices_replaced] = Y_MASK
        o_indexes[indices_replaced] = O_MASK
        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(x_labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_x = torch.randint(
            low=0, high=X_MASK - 1, size=x_labels.shape, dtype=torch.long
        )
        random_y = torch.randint(
            low=0, high=Y_MASK - 1, size=y_labels.shape, dtype=torch.long
        )
        random_f = torch.randint(
            low=0, high=O_MASK - 1, size=o_labels.shape, dtype=torch.long
        )
        x_indexes[indices_random] = random_x[indices_random]
        y_indexes[indices_random] = random_y[indices_random]
        o_indexes[indices_random] = random_f[indices_random]

        return x_indexes, y_indexes, o_indexes, x_labels, y_labels, o_labels


class InferenceDataset:
    def __init__(self, dataset_file_path: str, visual2index: Dict, without_text: bool):
        self.dataset_file = json.load(open(dataset_file_path))
        self.visual2index = visual2index
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.without_text = without_text

    def __len__(self):
        return len(self.dataset_file)

    def __getitem__(self, idx: int):
        # Obtain elements
        scene = self.dataset_file[idx]
        # Prepare sentences
        nested_sentences = [sentences for sentences in scene["sentences"].values()]
        sentences = np.array(
            [sentence for sublist in nested_sentences for sentence in sublist]
        )

        input_ids_sentence = torch.tensor(
            self.tokenizer.encode(" ".join(sentences), add_special_tokens=True)
        )

        if self.without_text:
            input_ids_sentence = torch.tensor(
                [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
            )
        # Prepare visuals
        input_ids_visuals = torch.tensor(
            [self.visual2index[element["visual_name"]] for element in scene["elements"]]
        )
        # Obtain X-indexes
        x_indexes = torch.tensor(
            [
                0
                if element["x"] < 0
                else X_MASK - 1
                if element["x"] > (X_MASK - 1) * BUCKET_SIZE
                else np.floor(element["x"] / BUCKET_SIZE)
                for element in scene["elements"]
            ],
            dtype=torch.long,
        )
        x_labels = torch.tensor(
            [
                0
                if element["x"] < 0
                else (X_MASK - 1) * BUCKET_SIZE
                if element["x"] > (X_MASK - 1) * BUCKET_SIZE
                else element["x"]
                for element in scene["elements"]
            ],
            dtype=torch.long,
        )
        # Obtain Y-indexes
        y_indexes = torch.tensor(
            [
                0
                if element["y"] < 0
                else Y_MASK - 1
                if element["y"] > (Y_MASK - 1) * BUCKET_SIZE
                else np.floor(element["y"] / BUCKET_SIZE)
                for element in scene["elements"]
            ],
            dtype=torch.long,
        )
        y_labels = torch.tensor(
            [
                0
                if element["y"] < 0
                else (Y_MASK - 1) * BUCKET_SIZE
                if element["y"] > (Y_MASK - 1) * BUCKET_SIZE
                else element["y"]
                for element in scene["elements"]
            ],
            dtype=torch.long,
        )
        # Obtain flips
        o_indexes = torch.tensor([element["flip"] for element in scene["elements"]])

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            o_indexes,
            x_labels,
            y_labels,
        )


class ContinuousTrainDataset(TrainDataset, TorchDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: Dict, mask_probability: float
    ):
        super().__init__(dataset_file_path, visual2index, mask_probability)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, o_indexes = super().__getitem__(
            idx
        )

        # Mask visual tokens
        x_indexes, y_indexes, o_indexes, x_labels, y_labels, o_labels = self.masking(
            x_indexes, y_indexes, o_indexes
        )

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            o_indexes,
            x_labels.float(),
            y_labels.float(),
            o_labels,
        )


class ContinuousInferenceDataset(InferenceDataset, TorchDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: Dict, without_text: bool = False
    ):
        super().__init__(dataset_file_path, visual2index, without_text)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, o_indexes, x_labels, y_labels = super().__getitem__(
            idx
        )

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            o_indexes,
            x_labels.float(),
            y_labels.float(),
            o_indexes,
        )


class DiscreteTrainDataset(TrainDataset, TorchDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: str, mask_probability: float
    ):
        super().__init__(dataset_file_path, visual2index, mask_probability)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, o_indexes = super().__getitem__(
            idx
        )

        # Mask visual tokens
        x_indexes, y_indexes, o_indexes, x_labels, y_labels, o_labels = self.masking(
            x_indexes, y_indexes, o_indexes
        )

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            o_indexes,
            x_labels,
            y_labels,
            o_labels,
        )


class DiscreteInferenceDataset(InferenceDataset, TorchDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: str, without_text: bool = False
    ):
        super().__init__(dataset_file_path, visual2index, without_text)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, o_indexes, x_labels, y_labels = super().__getitem__(
            idx
        )

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            o_indexes,
            x_labels,
            y_labels,
            o_indexes,
        )


index2pose_hb0 = {
    23: 0,
    34: 0,
    45: 0,
    51: 0,
    52: 0,
    53: 1,
    54: 1,
    55: 1,
    56: 1,
    57: 1,
    24: 2,
    25: 2,
    26: 2,
    27: 2,
    28: 2,
    29: 3,
    30: 3,
    31: 3,
    32: 3,
    33: 3,
    35: 4,
    36: 4,
    37: 4,
    38: 4,
    39: 4,
    40: 5,
    41: 5,
    42: 5,
    43: 5,
    44: 5,
    46: 6,
    47: 6,
    48: 6,
    49: 6,
    50: 6,
}
index2pose_hb1 = {
    58: 0,
    69: 0,
    80: 0,
    86: 0,
    87: 0,
    88: 1,
    89: 1,
    90: 1,
    91: 1,
    92: 1,
    59: 2,
    60: 2,
    61: 2,
    62: 2,
    63: 2,
    64: 3,
    65: 3,
    66: 3,
    67: 3,
    68: 3,
    70: 4,
    71: 4,
    72: 4,
    73: 4,
    74: 4,
    75: 5,
    76: 5,
    77: 5,
    78: 5,
    79: 5,
    81: 6,
    82: 6,
    83: 6,
    84: 6,
    85: 6,
}
index2expression_hb0 = {
    23: 0,
    53: 0,
    24: 0,
    29: 0,
    35: 0,
    40: 0,
    46: 0,
    34: 1,
    54: 1,
    25: 1,
    30: 1,
    36: 1,
    41: 1,
    47: 1,
    45: 2,
    55: 2,
    26: 2,
    31: 2,
    37: 2,
    42: 2,
    48: 2,
    51: 3,
    56: 3,
    27: 3,
    32: 3,
    38: 3,
    43: 3,
    49: 3,
    52: 4,
    57: 4,
    28: 4,
    33: 4,
    39: 4,
    44: 4,
    50: 4,
}
index2expression_hb1 = {
    58: 0,
    88: 0,
    59: 0,
    64: 0,
    70: 0,
    75: 0,
    81: 0,
    69: 1,
    89: 1,
    60: 1,
    65: 1,
    71: 1,
    76: 1,
    82: 1,
    80: 2,
    90: 2,
    61: 2,
    66: 2,
    72: 2,
    77: 2,
    83: 2,
    86: 3,
    91: 3,
    62: 3,
    67: 3,
    73: 3,
    78: 3,
    84: 3,
    87: 4,
    92: 4,
    63: 4,
    68: 4,
    74: 4,
    79: 4,
    85: 4,
}


class ClipartsPredictionDataset(TorchDataset):
    def __init__(self, dataset_file_path: str, visual2index: str, train: bool = False):
        self.dataset_file = json.load(open(dataset_file_path))
        self.visual2index = visual2index
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.train = train

    def __len__(self):
        return len(self.dataset_file)

    def __getitem__(self, idx: str):
        # Obtain elements
        scene = self.dataset_file[idx]
        # Prepare sentences
        nested_sentences = [sentences for sentences in scene["sentences"].values()]
        sentences = np.array(
            [sentence for sublist in nested_sentences for sentence in sublist]
        )
        if self.train:
            input_ids_sentence = torch.tensor(
                self.tokenizer.encode(
                    " ".join(np.random.permutation(sentences)), add_special_tokens=True
                )
            )
        else:
            input_ids_sentence = torch.tensor(
                self.tokenizer.encode(" ".join(sentences), add_special_tokens=True)
            )
        # Prepare visuals
        ids_visuals = [
            # Because they start from 1 we subtract 1
            self.visual2index[element["visual_name"]] - 1
            for element in scene["elements"]
        ]
        hb0_pose, hb0_expr, hb1_pose, hb1_expr = -100, -100, -100, -100
        object_visuals = []
        for visual in ids_visuals:
            if visual in index2pose_hb0 and visual in index2expression_hb0:
                hb0_pose = index2pose_hb0[visual]
                hb0_expr = index2expression_hb0[visual]
            elif visual in index2pose_hb1 and visual in index2expression_hb1:
                hb1_pose = index2pose_hb1[visual]
                hb1_expr = index2expression_hb1[visual]                
            elif visual > 92:
                object_visuals.append(visual - 70)
            else:
                object_visuals.append(visual)

        one_hot_objects = torch.zeros(23 + 33)
        one_hot_objects[object_visuals] = 1

        return (
            input_ids_sentence,
            one_hot_objects,
            torch.tensor(hb0_pose),
            torch.tensor(hb0_expr),
            torch.tensor(hb1_pose),
            torch.tensor(hb1_expr),
        )


def collate_pad_cliparts_prediction_batch(batch):
    ids_text, one_hot_objects, hb0_pose, hb0_expr, hb1_pose, hb1_expr = zip(*batch)
    ids_text = torch.nn.utils.rnn.pad_sequence(
        ids_text, batch_first=True, padding_value=0
    )
    attn_mask = ids_text.clone()
    attn_mask[torch.where(attn_mask > 0)] = 1

    return (
        ids_text,
        attn_mask,
        torch.stack([*one_hot_objects], dim=0),
        torch.stack([*hb0_pose], dim=0),
        torch.stack([*hb0_expr], dim=0),
        torch.stack([*hb1_pose], dim=0),
        torch.stack([*hb1_expr], dim=0),
    )


def collate_pad_batch(
    batch: Tuple[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
    ]
):
    ids_text, ids_vis, x_ind, y_ind, o_ind, x_lab, y_lab, o_lab = zip(*batch)
    # Get max text length to get the text positions
    max_text_length = max([element.size()[0] for element in ids_text])
    pos_text = torch.arange(max_text_length, dtype=torch.long)
    # Pad the sentences
    ids_text = torch.nn.utils.rnn.pad_sequence(
        ids_text, batch_first=True, padding_value=0
    )
    # Pad the visuals
    ids_vis = torch.nn.utils.rnn.pad_sequence(
        ids_vis, batch_first=True, padding_value=0
    )
    # Obtain the text positions
    pos_text = pos_text.unsqueeze(0).expand(ids_text.size())
    # Pad all visual inputs
    x_ind = torch.nn.utils.rnn.pad_sequence(
        x_ind, batch_first=True, padding_value=X_PAD
    )
    y_ind = torch.nn.utils.rnn.pad_sequence(
        y_ind, batch_first=True, padding_value=Y_PAD
    )
    o_ind = torch.nn.utils.rnn.pad_sequence(
        o_ind, batch_first=True, padding_value=O_PAD
    )
    # Pad the visual mappings and prepare final mappings
    text_labs = torch.ones_like(ids_text) * -100
    x_lab = torch.cat(
        [
            text_labs.float() if x_lab[0].dtype == torch.float32 else text_labs,
            torch.nn.utils.rnn.pad_sequence(
                x_lab, batch_first=True, padding_value=-100
            ),
        ],
        dim=1,
    )
    y_lab = torch.cat(
        [
            text_labs.float() if y_lab[0].dtype == torch.float32 else text_labs,
            torch.nn.utils.rnn.pad_sequence(
                y_lab, batch_first=True, padding_value=-100
            ),
        ],
        dim=1,
    )
    o_lab = torch.cat(
        [
            text_labs,
            torch.nn.utils.rnn.pad_sequence(
                o_lab, batch_first=True, padding_value=-100
            ),
        ],
        dim=1,
    )
    # Obtain token type ids
    t_types = torch.cat([torch.zeros_like(ids_text), torch.ones_like(ids_vis)], dim=1)
    # Obtain the attention mask
    attn_mask = torch.cat([ids_text, ids_vis], dim=1)
    attn_mask[torch.where(attn_mask > 0)] = 1

    return (
        ids_text,
        ids_vis,
        pos_text,
        x_ind,
        y_ind,
        o_ind,
        x_lab,
        y_lab,
        o_lab,
        t_types,
        attn_mask,
    )
