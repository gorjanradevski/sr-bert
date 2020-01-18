from torch.utils.data import Dataset
import json
from transformers import BertTokenizer
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Tuple, List

from constants import MAX_X, MAX_Y


class VisualScenesDataset(Dataset):
    def __init__(
        self, dataset_file_path: str, visual2index: str, mask_probability: float
    ):
        self.dataset_file = json.load(open(dataset_file_path))
        self.visual2index = visual2index
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.dataset_file)

    def __getitem__(self, idx: int):
        # Obtain elements
        scene = self.dataset_file[idx]
        # Prepare visuals
        tokenized_visuals = [
            self.visual2index[element["visual_name"]] for element in scene["elements"]
        ]
        # Mask visual tokens
        input_ids_visuals, masked_lm_labels_visuals = mask_tokens(
            torch.tensor(tokenized_visuals),
            self.tokenizer,
            self.mask_probability,
            self.visual2index,
        )
        # Obtain Z-indexes
        z_indexes = np.array([element["z"] for element in scene["elements"]])
        z_onehot = np.zeros((z_indexes.size, 3))
        z_onehot[np.arange(z_indexes.size), z_indexes] = 1
        # Obtain flips
        flips = np.array([element["flip"] for element in scene["elements"]])
        flips_onehot = np.zeros((flips.size, 2))
        flips_onehot[np.arange(flips.size), flips] = 1
        # Obtain and normalize visual positions
        visual_positions = [
            [element["x"] / MAX_X, element["y"] / MAX_Y] + z_index + flip
            for element, z_index, flip in zip(
                scene["elements"], z_onehot.tolist(), flips_onehot.tolist()
            )
        ]

        return (
            input_ids_visuals,
            masked_lm_labels_visuals,
            torch.tensor(visual_positions),
        )


class MultimodalScenesDataset(Dataset):
    def __init__(
        self, dataset_file_path: str, visual2index: str, mask_probability: float
    ):
        self.dataset_file = json.load(open(dataset_file_path))
        self.visual2index = visual2index
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.dataset_file)

    def __getitem__(self, idx: int):
        # Obtain elements
        scene = self.dataset_file[idx]
        # Prepare sentence
        tokenized_sentence = self.tokenizer.encode(
            scene["sentence"], add_special_tokens=True
        )
        # Prepare visuals
        tokenized_visuals = [
            self.visual2index[element["visual_name"]] for element in scene["elements"]
        ]
        tokenized_visuals.append(self.tokenizer.sep_token_id)
        # Masking sentence tokens
        input_ids_sentence, masked_lm_labels_sentence = mask_tokens(
            torch.tensor(tokenized_sentence),
            self.tokenizer,
            self.mask_probability,
            self.visual2index,
            masking_visuals=False,
        )
        # Mask visual tokens
        input_ids_visuals, masked_lm_labels_visuals = mask_tokens(
            torch.tensor(tokenized_visuals),
            self.tokenizer,
            self.mask_probability,
            self.visual2index,
            masking_visuals=True
        )
        # Obtain Z-indexes
        z_indexes = np.array([element["z"] for element in scene["elements"]])
        z_onehot = np.zeros((z_indexes.size, 3))
        z_onehot[np.arange(z_indexes.size), z_indexes] = 1
        # Obtain flips
        flips = np.array([element["flip"] for element in scene["elements"]])
        flips_onehot = np.zeros((flips.size, 2))
        flips_onehot[np.arange(flips.size), flips] = 1
        # Obtain and normalize visual positions
        visual_positions = [
            [element["x"] / MAX_X, element["y"] / MAX_Y] + z_index + flip
            for element, z_index, flip in zip(
                scene["elements"], z_onehot.tolist(), flips_onehot.tolist()
            )
        ]
        visual_positions.append([-1, -1, -1, -1, -1, -1, -1])

        return (
            input_ids_sentence,
            input_ids_visuals,
            masked_lm_labels_sentence,
            masked_lm_labels_visuals,
            torch.tensor(visual_positions),
        )


class ClipartsDataset(Dataset):
    def __init__(self, cliparts_path: str, visual2index_path: str):
        visual2index = json.load(open(visual2index_path))
        self.file_paths_indices = []
        for file_name, index in visual2index.items():
            file_path = os.path.join(cliparts_path, file_name)
            self.file_paths_indices.append((file_path, index))
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx: int):
        file_path, index = self.file_paths_indices[idx]
        image = Image.open(file_path).convert("RGB")
        image_transformed = self.transforms(image)

        return image_transformed, index

    def __len__(self):
        return len(self.file_paths_indices)


def mask_tokens(
    inputs: torch.Tensor,
    tokenizer,
    mask_prob,
    visual2index,
    masking_visuals: bool = False,
):
    # https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py#L169
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mask_prob)
    special_tokens_mask = tokenizer.get_special_tokens_mask(
        labels.tolist(), already_has_special_tokens=True
    )
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    # Let's check whether we are masking visuals
    if masking_visuals:
        low = len(tokenizer)
        high = len(tokenizer) + len(visual2index)
    else:
        low = 0
        high = len(tokenizer)
    random_words = torch.randint(
        low=low, high=high, size=labels.shape, dtype=torch.long
    )
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def collate_pad_visual_batch(
    batch: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
):
    input_ids_visuals, masked_lm_labels_visuals, visual_positions = zip(*batch)
    # Expand the position ids
    input_ids_visuals = torch.nn.utils.rnn.pad_sequence(
        input_ids_visuals, batch_first=True
    )
    masked_lm_labels_visuals = torch.nn.utils.rnn.pad_sequence(
        masked_lm_labels_visuals, batch_first=True, padding_value=-100
    )
    visual_positions = torch.nn.utils.rnn.pad_sequence(
        visual_positions, batch_first=True, padding_value=-1
    )
    # Obtain attention mask
    attention_mask = input_ids_visuals.clone()
    attention_mask[torch.where(attention_mask > 0)] = 1

    return (
        input_ids_visuals,
        masked_lm_labels_visuals,
        visual_positions,
        attention_mask,
    )


def collate_pad_multimodal_batch(
    batch: Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]
):
    input_ids_sentence, input_ids_visuals, masked_lm_labels_sentence, masked_lm_labels_visuals, visual_positions = zip(
        *batch
    )
    # Generate position ids for the text
    max_text_length = max([element.size()[0] for element in input_ids_sentence])
    text_positions = torch.arange(max_text_length, dtype=torch.long)
    # Pad text sequences
    input_ids_sentence = torch.nn.utils.rnn.pad_sequence(
        input_ids_sentence, batch_first=True
    )
    # Expand the position ids
    input_ids_visuals = torch.nn.utils.rnn.pad_sequence(
        input_ids_visuals, batch_first=True
    )
    input_ids = torch.cat([input_ids_sentence, input_ids_visuals], dim=1)
    text_positions = text_positions.unsqueeze(0).expand(input_ids_sentence.size())
    masked_lm_labels_sentence = torch.nn.utils.rnn.pad_sequence(
        masked_lm_labels_sentence, batch_first=True, padding_value=-100
    )
    masked_lm_labels_visuals = torch.nn.utils.rnn.pad_sequence(
        masked_lm_labels_visuals, batch_first=True, padding_value=-100
    )
    visual_positions = torch.nn.utils.rnn.pad_sequence(
        visual_positions, batch_first=True, padding_value=-1
    )
    # Obtain attention mask
    attention_mask = input_ids.clone()
    attention_mask[torch.where(attention_mask > 0)] = 1
    # Prepare masked labels
    masked_lm_labels = torch.cat(
        [masked_lm_labels_sentence, masked_lm_labels_visuals], dim=1
    )
    token_type_ids = torch.cat(
        [torch.zeros_like(input_ids_sentence), torch.ones_like(input_ids_visuals)],
        dim=1,
    )

    return (
        input_ids,
        masked_lm_labels,
        text_positions,
        visual_positions,
        token_type_ids,
        attention_mask,
    )
