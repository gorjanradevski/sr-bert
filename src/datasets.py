from torch.utils.data import Dataset
import json
from transformers import BertTokenizer
import torch
import os
from PIL import Image
from torchvision import transforms
from typing import Tuple, List

from constants import MAX_X, MAX_Y


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
        # Prepare sentence
        tokenized_sentence = self.tokenizer.encode(
            scene["sentence"], add_special_tokens=True
        )
        input_ids_sentence = torch.tensor(tokenized_sentence)
        # Prepare visuals
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
        # Mask visual tokens
        input_ids_visuals, masked_lm_labels_visuals = self.mask_tokens(
            torch.tensor(tokenized_visuals), self.tokenizer, self.mask_probability
        )
        # Obtain and normalize visual positions
        visual_positions = [
            [element["x"] / MAX_X, element["y"] / MAX_Y]
            for element in scene["elements"]
        ]
        visual_positions.append([-1, -1])

        return (
            input_ids_sentence,
            input_ids_visuals,
            masked_lm_labels_visuals,
            torch.tensor(visual_positions),
        )

    @staticmethod
    def mask_tokens(inputs: torch.Tensor, tokenizer, mask_prob):
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
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class ClipartsDataset(Dataset):
    def __init__(self, cliparts_path: str, visual2index_path: str):
        visual2index = json.load(open(visual2index_path))
        self.file_paths_indices = []
        for (file_path, index), index in visual2index.items():
            name, extension = file_path.split(".")
            file_name = name[:-4] + "." + extension
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


def collate_pad_batch(
    batch: Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]
):
    input_ids_sentence, input_ids_visuals, masked_lm_labels_visuals, visual_positions = zip(
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
    text_positions = text_positions.unsqueeze(0).expand(input_ids_sentence.size())
    masked_lm_labels_visuals = torch.nn.utils.rnn.pad_sequence(
        masked_lm_labels_visuals, batch_first=True, padding_value=-100
    )
    visual_positions = torch.nn.utils.rnn.pad_sequence(
        visual_positions, batch_first=True, padding_value=-1
    )
    # Concatenate stuff
    input_ids = torch.cat([input_ids_sentence, input_ids_visuals], dim=1)
    # Obtain attention mask
    attention_mask = input_ids.clone()
    attention_mask[torch.where(attention_mask > 0)] = 1
    # Prepare masked labels
    masked_lm_labels_text = torch.ones_like(input_ids_sentence) * -100
    masked_lm_labels = torch.cat(
        [masked_lm_labels_text, masked_lm_labels_visuals], dim=1
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
