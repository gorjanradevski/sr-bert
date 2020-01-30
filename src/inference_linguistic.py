import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import logging
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer

from datasets import (
    LinguisticScenesInferenceDataset,
    collate_pad_linguistic_inference_batch,
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


def inference(checkpoint_path: str, test_dataset_path: str, batch_size: int):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"--- Using device {device}! ---")
    # Create datasets
    test_dataset = LinguisticScenesInferenceDataset(test_dataset_path)
    logger.info(f"Testing on on {len(test_dataset)}")
    # Create samplers
    test_sampler = SequentialSampler(test_dataset)
    # Create loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_linguistic_inference_batch,
        sampler=test_sampler,
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Define training specifics
    model = nn.DataParallel(BertForMaskedLM.from_pretrained("bert-base-uncased")).to(
        device
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # Set model in evaluation mode
    model.train(False)
    total = 0
    correct = 0
    with torch.no_grad():
        # Reset current loss
        with tqdm(total=len(test_loader)) as pbar:
            for (input_ids, labels, attention_masks) in test_loader:
                input_ids, labels, attention_masks = (
                    input_ids.to(device),
                    labels.to(device),
                    attention_masks.to(device),
                )
                predictions = model(
                    input_ids=input_ids, attention_mask=attention_masks
                )[0]
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
        description="Performs inference with a Linguistic model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/linguistic_best.pt",
        help="Checkpoint to a trained model.",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test dataset.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")

    return parser.parse_args()


def main():
    args = parse_args()
    inference(args.checkpoint_path, args.test_dataset_path, args.batch_size)


if __name__ == "__main__":
    main()
