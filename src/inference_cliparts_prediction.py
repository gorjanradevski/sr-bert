import argparse
import json
import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from scene_layouts.datasets import (
    ClipartsPredictionDataset,
    collate_pad_cliparts_prediction_batch,
)
from scene_layouts.evaluator import ClipartsPredictionEvaluator
from scene_layouts.modeling import ClipartsPredictionModel


def train(args):
    logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(
        open(os.path.join(args.visuals_dicts_path, "visual2index.json"))
    )
    index2pose_hb0 = json.load(
        open(os.path.join(args.visuals_dicts_path, "index2pose_hb0.json"))
    )
    index2pose_hb1 = json.load(
        open(os.path.join(args.visuals_dicts_path, "index2pose_hb1.json"))
    )
    index2expression_hb0 = json.load(
        open(os.path.join(args.visuals_dicts_path, "index2expression_hb0.json"))
    )
    index2expression_hb1 = json.load(
        open(os.path.join(args.visuals_dicts_path, "index2expression_hb1.json"))
    )
    test_dataset = ClipartsPredictionDataset(args.test_dataset_path, visual2index)
    print(f"Testing on {len(test_dataset)}")
    # Create loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pad_cliparts_prediction_batch,
    )
    # Define training specifics
    config = BertConfig.from_pretrained(args.bert_name)
    config.vocab_size = len(visual2index)
    model = nn.DataParallel(ClipartsPredictionModel(config, args.bert_name)).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.train(False)
    evaluator = ClipartsPredictionEvaluator(
        len(test_dataset),
        visual2index,
        index2pose_hb0,
        index2pose_hb1,
        index2expression_hb0,
        index2expression_hb1,
    )
    # Set model in evaluation mode
    model.train(False)
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # forward
            batch = {key: val.to(device) for key, val in batch.items()}
            # Get predictions
            probs = torch.sigmoid(model(batch["ids_text"], batch["attn_mask"]))
            one_hot_pred = torch.zeros_like(probs)
            # Regular objects
            one_hot_pred[:, :23][torch.where(probs[:, :23] > 0.35)] = 1
            one_hot_pred[:, 93:][torch.where(probs[:, 93:] > 0.35)] = 1
            # Mike and Jenny
            batch_indices = torch.arange(batch["ids_text"].size()[0])
            max_hb0 = torch.argmax(probs[:, 23:58], axis=-1)
            one_hot_pred[batch_indices, max_hb0 + 23] = 1
            max_hb1 = torch.argmax(probs[:, 58:93], axis=-1)
            one_hot_pred[batch_indices, max_hb1 + 58] = 1
            # Aggregate predictions/targets
            evaluator.update_counters(
                one_hot_pred.cpu().numpy(), batch["one_hot_visuals"].cpu().numpy()
            )

    precision, recall, f1_score = evaluator.per_object_prf()
    posses_acc, expr_acc = evaluator.posses_expressions_accuracy()
    logging.info("====================================================")
    logging.info(f"Precison is {precision}, recall is {recall}, F1 is {f1_score}")
    logging.info(
        f"Posess accuracy is {posses_acc}, and expression accuracy is {expr_acc}"
    )
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
        "--visuals_dicts_path",
        type=str,
        default="data/visuals_dicts/",
        help="Path to the directory with the visuals dictionaries.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
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
    train(args)


if __name__ == "__main__":
    main()
