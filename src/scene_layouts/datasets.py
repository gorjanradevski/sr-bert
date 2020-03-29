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
F_MASK = 2
F_PAD = 3
SCENE_WIDTH_TRAIN = 500 // BUCKET_SIZE
SCENE_WIDTH_TEST = 500


class Text2VisualTrainDataset:
    def __init__(
        self, dataset_file_path: str, visual2index: Dict, mask_probability: float
    ):
        self.dataset_file = json.load(open(dataset_file_path))
        self.visual2index = visual2index
        self.mask_probability = mask_probability
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    @staticmethod
    def flip_scene(x_indexes: torch.Tensor, f_indexes: torch.Tensor):
        return torch.abs(SCENE_WIDTH_TRAIN - x_indexes), torch.abs(1 - f_indexes)

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
        f_indexes = torch.tensor([element["flip"] for element in scene["elements"]])
        # Flip scene with 50% prob
        if torch.bernoulli(torch.tensor([0.5])).bool().item():
            x_indexes, f_indexes = self.flip_scene(x_indexes, f_indexes)

        return input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, f_indexes

    def masking(self, x_indexes, y_indexes, f_indexes):
        # https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py#L169
        # Create clones for everything
        x_labels = x_indexes.clone()
        y_labels = y_indexes.clone()
        f_labels = f_indexes.clone()
        # Get probability matrix
        probability_matrix = torch.full(x_indexes.shape, self.mask_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # We only compute loss on masked tokens
        x_labels[~masked_indices] = -100
        y_labels[~masked_indices] = -100
        f_labels[~masked_indices] = -100
        # 80% we replace with a mask token
        indices_replaced = (
            torch.bernoulli(torch.full(x_indexes.shape, 0.8)).bool() & masked_indices
        )
        x_indexes[indices_replaced] = X_MASK
        y_indexes[indices_replaced] = Y_MASK
        f_indexes[indices_replaced] = F_MASK
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
            low=0, high=F_MASK - 1, size=f_labels.shape, dtype=torch.long
        )
        x_indexes[indices_random] = random_x[indices_random]
        y_indexes[indices_random] = random_y[indices_random]
        f_indexes[indices_random] = random_f[indices_random]

        return x_indexes, y_indexes, f_indexes, x_labels, y_labels, f_labels


class Text2VisualTestDataset:
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
        f_indexes = torch.tensor([element["flip"] for element in scene["elements"]])

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            f_indexes,
            x_labels,
            y_labels,
        )


class Text2VisualContinuousTrainDataset(Text2VisualTrainDataset, TorchDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: Dict, mask_probability: float
    ):
        super().__init__(dataset_file_path, visual2index, mask_probability)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, f_indexes = super().__getitem__(
            idx
        )

        # Mask visual tokens
        x_indexes, y_indexes, f_indexes, x_labels, y_labels, f_labels = self.masking(
            x_indexes, y_indexes, f_indexes
        )

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            f_indexes,
            x_labels.float(),
            y_labels.float(),
            f_labels,
        )


class Text2VisualContinuousTestDataset(Text2VisualTestDataset, TorchDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: Dict, without_text: bool = False
    ):
        super().__init__(dataset_file_path, visual2index, without_text)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, f_indexes, x_labels, y_labels = super().__getitem__(
            idx
        )

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            f_indexes,
            x_labels.float(),
            y_labels.float(),
            f_indexes,
        )


class Text2VisualDiscreteTrainDataset(Text2VisualTrainDataset, TorchDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: str, mask_probability: float
    ):
        super().__init__(dataset_file_path, visual2index, mask_probability)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, f_indexes = super().__getitem__(
            idx
        )

        # Mask visual tokens
        x_indexes, y_indexes, f_indexes, x_labels, y_labels, f_labels = self.masking(
            x_indexes, y_indexes, f_indexes
        )

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            f_indexes,
            x_labels,
            y_labels,
            f_labels,
        )


class Text2VisualDiscreteTestDataset(Text2VisualTestDataset, TorchDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: str, without_text: bool = False
    ):
        super().__init__(dataset_file_path, visual2index, without_text)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        input_ids_sentence, input_ids_visuals, x_indexes, y_indexes, f_indexes, x_labels, y_labels = super().__getitem__(
            idx
        )

        return (
            input_ids_sentence,
            input_ids_visuals,
            x_indexes,
            y_indexes,
            f_indexes,
            x_labels,
            y_labels,
            f_indexes,
        )


def collate_pad_discrete_text2visual_batch(
    batch: Tuple[
        Tuple[torch.Tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
    ]
):
    ids_text, ids_vis, x_ind, y_ind, f_ind, x_lab, y_lab, f_lab = zip(*batch)
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
    f_ind = torch.nn.utils.rnn.pad_sequence(
        f_ind, batch_first=True, padding_value=F_PAD
    )
    # Pad the visual mappings and prepare final mappings
    text_labs = torch.ones_like(ids_text) * -100
    x_lab = torch.cat(
        [
            text_labs,
            torch.nn.utils.rnn.pad_sequence(
                x_lab, batch_first=True, padding_value=-100
            ),
        ],
        dim=1,
    )
    y_lab = torch.cat(
        [
            text_labs,
            torch.nn.utils.rnn.pad_sequence(
                y_lab, batch_first=True, padding_value=-100
            ),
        ],
        dim=1,
    )
    f_lab = torch.cat(
        [
            text_labs,
            torch.nn.utils.rnn.pad_sequence(
                f_lab, batch_first=True, padding_value=-100
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
        f_ind,
        x_lab,
        y_lab,
        f_lab,
        t_types,
        attn_mask,
    )


def collate_pad_continuous_text2visual_batch(
    batch: Tuple[
        Tuple[torch.Tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
        Tuple[torch.tensor],
    ]
):
    ids_text, ids_vis, x_ind, y_ind, f_ind, x_lab, y_lab, f_lab = zip(*batch)
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
    f_ind = torch.nn.utils.rnn.pad_sequence(
        f_ind, batch_first=True, padding_value=F_PAD
    )
    # Pad the visual mappings and prepare final mappings
    text_labs = torch.ones_like(ids_text) * -100
    x_lab = torch.cat(
        [
            text_labs.float(),
            torch.nn.utils.rnn.pad_sequence(
                x_lab, batch_first=True, padding_value=-100
            ),
        ],
        dim=1,
    )
    y_lab = torch.cat(
        [
            text_labs.float(),
            torch.nn.utils.rnn.pad_sequence(
                y_lab, batch_first=True, padding_value=-100
            ),
        ],
        dim=1,
    )
    f_lab = torch.cat(
        [
            text_labs,
            torch.nn.utils.rnn.pad_sequence(
                f_lab, batch_first=True, padding_value=-100
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
        f_ind,
        x_lab,
        y_lab,
        f_lab,
        t_types,
        attn_mask,
    )