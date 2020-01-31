import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import logging
import numpy as np
import json
from transformers import BertTokenizer, BertConfig
from modeling import MultiModalBert

from datasets import (
    MultimodalScenesInferenceVisualDataset,
    MultimodalScenesInferenceLinguisticDataset,
    collate_pad_multimodal_inference_batch,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_batch(
    input_ids: torch.tensor,
    predictions: torch.tensor,
    labels: torch.tensor,
    mask_id: int,
):
    mask_ids = np.where(input_ids.cpu().numpy() == mask_id)[1]
    output_ids = torch.argmax(predictions, dim=2)[
        torch.arange(predictions.size(0)), mask_ids
    ]

    return np.count_nonzero(output_ids == labels)


def dataset_factory(
    evaluation_type: str, test_dataset_path: str, visual2index, use_other_modality: bool
):
    if evaluation_type == "linguistic":
        return MultimodalScenesInferenceLinguisticDataset(
            test_dataset_path, visual2index, use_other_modality
        )
    elif evaluation_type == "visual":
        return MultimodalScenesInferenceVisualDataset(
            test_dataset_path, visual2index, use_other_modality
        )
    else:
        raise ValueError(f"Evaluation type {evaluation_type} not recognized!")


def inference(
    evaluation_type: str,
    checkpoint_path: str,
    test_dataset_path: str,
    visual2index_path: str,
    use_other_modality: bool,
    batch_size: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    test_dataset = dataset_factory(
        evaluation_type, test_dataset_path, visual2index, use_other_modality
    )
    logger.info(f"Testing on on {len(test_dataset)}")
    logger.warning(f"Evaluation type is: {evaluation_type}")
    logger.warning(f"The use of other modality is: {use_other_modality}")
    # Create samplers
    test_sampler = SequentialSampler(test_dataset)
    # Create loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_multimodal_inference_batch,
        sampler=test_sampler,
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Define training specifics
    config = BertConfig()
    config.vocab_size += len(visual2index)
    model = nn.DataParallel(MultiModalBert(config, device)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    total = 0
    correct = 0
    with torch.no_grad():
        # Reset current loss
        with tqdm(total=len(test_loader)) as pbar:
            for (
                input_ids,
                labels,
                text_positions,
                visual_positions,
                token_type_ids,
                attention_masks,
            ) in test_loader:
                input_ids, labels, text_positions, visual_positions, token_type_ids, attention_masks = (
                    input_ids.to(device),
                    labels.to(device),
                    text_positions.to(device),
                    visual_positions.to(device),
                    token_type_ids.to(device),
                    attention_masks.to(device),
                )
                predictions = model(
                    input_ids=input_ids,
                    text_positions=text_positions,
                    visual_positions=visual_positions,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_masks,
                )
                total += input_ids.size()[0]
                correct += evaluate_batch(
                    input_ids, predictions, labels, tokenizer.mask_token_id
                )
                # Update progress bar
                pbar.update(1)

        print(f"The accuracy of the model is: {correct / total}")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Performs inference with a multi-modal model."
    )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        default="linguistic",
        help="The evaluation type: linguistic or visual",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/multimodal_best_full_masking.pt",
        help="Checkpoint to a trained model.",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--visual2index_path",
        type=str,
        default="data/visual2index_full_masking.json",
        help="Path to the visual2index file.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument(
        "--use_other_modality",
        action="store_true",
        help="Whether to do inference using the linguistic input",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    inference(
        args.evaluation_type,
        args.checkpoint_path,
        args.test_dataset_path,
        args.visual2index_path,
        args.use_other_modality,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
