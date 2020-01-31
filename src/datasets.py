from torch.utils.data import Dataset as TorchDataset
import json
from transformers import BertTokenizer
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Tuple, List

MAX_X = 598
MAX_Y = 704
VISUAL_MASK_TOKEN = 1


class ScenesDataset:
    def __init__(self, dataset_file_path: str, mask_probability: float = 0.0):
        self.dataset_file = json.load(open(dataset_file_path))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.dataset_file)

    def mask_tokens(
        self, inputs: torch.Tensor, mask_token_id: int, low: int, high: int
    ):
        # https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py#L169
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mask_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
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
        inputs[indices_replaced] = mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            low=low, high=high, size=labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class LinguisticScenesTrainDataset(TorchDataset, ScenesDataset):
    def __init__(self, dataset_file_path: str, mask_probability: float):
        super().__init__(dataset_file_path, mask_probability)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        scene = self.dataset_file[idx]
        # Collect sentences
        nested_sentences = [sentences for sentences in scene["sentences"].values()]
        sentences = [sentence for sublist in nested_sentences for sentence in sublist]
        # Obtain sentences
        tokenized_sentences = self.tokenizer.encode(
            " ".join(sentences), add_special_tokens=True
        )
        input_ids_sentence, masked_lm_labels_sentence = self.mask_tokens(
            torch.tensor(tokenized_sentences),
            self.tokenizer.mask_token_id,
            low=0,
            high=len(self.tokenizer),
        )
        return input_ids_sentence, masked_lm_labels_sentence


class LinguisticScenesInferenceDataset(TorchDataset, ScenesDataset):
    def __init__(self, dataset_file_path: str):
        super().__init__(dataset_file_path)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        scene = self.dataset_file[idx]
        input_ids_sentence = torch.tensor(
            self.tokenizer.encode(scene["sentence"], add_special_tokens=True)
        )
        input_id_label = torch.tensor(
            self.tokenizer.encode(scene["label"], add_special_tokens=False)
        )[0]
        return input_ids_sentence, input_id_label


class VisualScenesDataset(TorchDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: str, mask_probability: float
    ):
        super().__init__(dataset_file_path, mask_probability)
        self.visual2index = visual2index

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        # Obtain elements
        scene = self.dataset_file[idx]
        # Prepare visuals
        tokenized_visuals = [
            self.visual2index[element["visual_name"]] for element in scene["elements"]
        ]
        # Mask visual tokens
        input_ids_visuals, masked_lm_labels_visuals = self.mask_tokens(
            torch.tensor(tokenized_visuals),
            VISUAL_MASK_TOKEN,
            low=2,
            high=len(self.visual2index) + 2,
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


class MultimodalScenesTrainDataset(TorchDataset, ScenesDataset):
    def __init__(
        self, dataset_file_path: str, visual2index: str, mask_probability: float
    ):
        super().__init__(dataset_file_path, mask_probability)
        self.visual2index = visual2index

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        # Obtain elements
        scene = self.dataset_file[idx]
        # Obtain sentences
        nested_sentences = [sentences for sentences in scene["sentences"].values()]
        all_sentences = np.array(
            [sentence for sublist in nested_sentences for sentence in sublist]
        )
        sentences = np.random.choice(all_sentences, 3, replace=False)
        tokenized_sentence = self.tokenizer.encode(
            " ".join(sentences), add_special_tokens=True
        )
        # Prepare visuals
        tokenized_visuals = [
            self.visual2index[element["visual_name"]] for element in scene["elements"]
        ]
        tokenized_visuals.append(self.tokenizer.sep_token_id)
        # Masking sentence tokens
        input_ids_sentence, masked_lm_labels_sentence = self.mask_tokens(
            torch.tensor(tokenized_sentence),
            self.tokenizer.mask_token_id,
            low=0,
            high=len(self.tokenizer),
        )
        # Mask visual tokens
        input_ids_visuals, masked_lm_labels_visuals = self.mask_tokens(
            torch.tensor(tokenized_visuals),
            self.tokenizer.mask_token_id,
            low=len(self.tokenizer),
            high=len(self.tokenizer) + len(self.visual2index),
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


class MultimodalScenesInferenceLinguisticDataset(TorchDataset, ScenesDataset):
    def __init__(self, dataset_file_path: str, visual2index: str, use_visuals: bool):
        super().__init__(dataset_file_path)
        self.visual2index = visual2index
        self.use_visuals = use_visuals

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        scene = self.dataset_file[idx]
        input_ids_sentence = torch.tensor(
            self.tokenizer.encode(scene["sentence"], add_special_tokens=True)
        )
        input_id_label = torch.tensor(
            self.tokenizer.encode(scene["label"], add_special_tokens=False)
        )[0]
        # Prepare visuals
        if self.use_visuals:
            tokenized_visuals = [
                self.visual2index[element["visual_name"]]
                for element in scene["elements"]
            ]
        else:
            tokenized_visuals = []
        tokenized_visuals.append(self.tokenizer.sep_token_id)
        # Obtain Z-indexes
        if self.use_visuals:
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
        else:
            visual_positions = []
        visual_positions.append([-1.0, -1.0, -1, -1, -1, -1, -1])

        return (
            input_ids_sentence,
            torch.tensor(tokenized_visuals),
            torch.tensor(visual_positions),
            input_id_label,
        )


class MultimodalScenesInferenceVisualDataset(TorchDataset, ScenesDataset):
    def __init__(self, dataset_file_path: str, visual2index: str, use_linguistic: bool):
        super().__init__(dataset_file_path)
        self.visual2index = visual2index
        self.use_linguistic = use_linguistic

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx: int):
        # Obtain elements
        scene = self.dataset_file[idx]
        # Obtain sentences
        if self.use_linguistic:
            nested_sentences = [sentences for sentences in scene["sentences"].values()]
            all_sentences = [
                sentence for sublist in nested_sentences for sentence in sublist
            ]
            tokenized_sentence = self.tokenizer.encode(
                " ".join(all_sentences), add_special_tokens=True
            )
        else:
            tokenized_sentence = self.tokenizer.encode("[CLS] [SEP]")
        input_ids_sentence = torch.tensor(tokenized_sentence)
        # Prepare visuals
        tokenized_visuals = [
            self.visual2index[element["visual_name"]]
            if element["visual_name"] in self.visual2index
            else self.tokenizer.mask_token_id
            for element in scene["elements"]
        ]
        tokenized_visuals.append(self.tokenizer.sep_token_id)
        input_ids_visuals = torch.tensor(tokenized_visuals)
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

        label = torch.tensor(self.visual2index[scene["label"]["visual_name"]])

        return (
            input_ids_sentence,
            input_ids_visuals,
            torch.tensor(visual_positions),
            label,
        )


class ClipartsDataset(TorchDataset):
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


def collate_pad_linguistic_train_batch(
    batch: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
):
    input_ids_sentence, masked_lm_labels_sentence = zip(*batch)
    # Expand the position ids
    input_ids_sentence = torch.nn.utils.rnn.pad_sequence(
        input_ids_sentence, batch_first=True
    )
    masked_lm_labels_sentence = torch.nn.utils.rnn.pad_sequence(
        masked_lm_labels_sentence, batch_first=True, padding_value=-100
    )
    # Obtain attention mask
    attention_mask = input_ids_sentence.clone()
    attention_mask[torch.where(attention_mask > 0)] = 1

    return (input_ids_sentence, masked_lm_labels_sentence, attention_mask)


def collate_pad_linguistic_inference_batch(
    batch: Tuple[List[torch.Tensor], List[torch.Tensor]]
):
    input_ids_sentence, labels = zip(*batch)
    # Expand the position ids
    input_ids_sentence = torch.nn.utils.rnn.pad_sequence(
        input_ids_sentence, batch_first=True
    )
    labels = torch.tensor([*labels])
    # Obtain attention mask
    attention_mask = input_ids_sentence.clone()
    attention_mask[torch.where(attention_mask > 0)] = 1

    return (input_ids_sentence, labels, attention_mask)


def collate_pad_visual_train_batch(
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


def collate_pad_multimodal_train_batch(
    batch: Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.tensor],
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


def collate_pad_multimodal_inference_batch(
    batch: Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]
):
    input_ids_sentence, input_ids_visuals, visual_positions, labels = zip(*batch)
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
    visual_positions = torch.nn.utils.rnn.pad_sequence(
        visual_positions, batch_first=True, padding_value=-1
    )
    # Obtain attention mask
    attention_mask = input_ids.clone()
    attention_mask[torch.where(attention_mask > 0)] = 1
    # Prepare masked labels
    token_type_ids = torch.cat(
        [torch.zeros_like(input_ids_sentence), torch.ones_like(input_ids_visuals)],
        dim=1,
    )

    labels = torch.tensor([*labels])

    return (
        input_ids,
        labels,
        text_positions,
        visual_positions,
        token_type_ids,
        attention_mask,
    )
