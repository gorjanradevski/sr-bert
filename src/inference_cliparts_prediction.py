import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
from transformers import BertConfig

from scene_layouts.datasets import (
    ClipartsPredictionDataset,
    collate_pad_cliparts_prediction_batch,
)
from scene_layouts.modeling import ClipartsPredictionModel
from scene_layouts.evaluator import ClipartsPredictionEvaluator


def train(
    test_dataset_path: str,
    visual2index_path: str,
    bert_name: str,
    batch_size: int,
    checkpoint_path: str,
):
    logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    test_dataset = ClipartsPredictionDataset(test_dataset_path, visual2index)
    logging.info(f"Inference on {len(test_dataset)}")
    # Create loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_pad_cliparts_prediction_batch,
    )
    # Define training specifics
    config = BertConfig.from_pretrained(bert_name)
    config.vocab_size = len(visual2index)
    model = nn.DataParallel(ClipartsPredictionModel(config, bert_name)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)
    evaluator = ClipartsPredictionEvaluator(len(test_dataset), visual2index)
    with torch.no_grad():
        for ids_text, attn_mask, target_visuals in tqdm(test_loader):
            # forward
            ids_text, attn_mask, target_visuals = (
                ids_text.to(device),
                attn_mask.to(device),
                target_visuals.to(device),
            )
            # Get predictions
            probs = torch.sigmoid(model(ids_text, attn_mask))
            one_hot_pred = torch.zeros_like(probs)
            one_hot_pred[torch.where(probs > 0.5)] = 1
            # Aggregate predictions/targets
            evaluator.update_counters(
                one_hot_pred.cpu().numpy(), target_visuals.cpu().numpy()
            )

    precision, recall, f1_score = evaluator.per_object_pr()
    posses_expressions_accuracy = evaluator.posses_expressions_accuracy()
    logging.info("====================================================")
    logging.info(f"Precison is {precision}, recall is {recall}, F1 is {f1_score}")
    logging.info(f"Posess and expression accuracy is {posses_expressions_accuracy}")
    logging.info("====================================================")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Inference with Cliparts prediction model."
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the validation dataset.",
    )
    parser.add_argument(
        "--visual2index_path",
        type=str,
        default="data/visual2index.json",
        help="Path to the visual2index mapping json.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/best.pt",
        help="Path to a checkpoint.",
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The bert model name.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train(
        args.test_dataset_path,
        args.visual2index_path,
        args.bert_name,
        args.batch_size,
        args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
