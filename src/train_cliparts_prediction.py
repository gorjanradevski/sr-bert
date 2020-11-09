import argparse
import json
import logging
import os

import torch
import torch.optim as optim
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
    # Set up logging
    if args.log_filepath:
        logging.basicConfig(
            level=logging.INFO, filename=args.log_filepath, filemode="w"
        )
    else:
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
    train_dataset = ClipartsPredictionDataset(
        args.train_dataset_path, visual2index, train=True
    )
    val_dataset = ClipartsPredictionDataset(args.val_dataset_path, visual2index)
    logging.info(f"Training on {len(train_dataset)}")
    logging.info(f"Validating on {len(val_dataset)}")
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pad_cliparts_prediction_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pad_cliparts_prediction_batch,
    )
    # Define training specifics
    config = BertConfig.from_pretrained(args.bert_name)
    config.vocab_size = len(visual2index)
    model = nn.DataParallel(ClipartsPredictionModel(config, args.bert_name)).to(device)
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score = -1.0
    evaluator = ClipartsPredictionEvaluator(
        len(val_dataset),
        visual2index,
        index2pose_hb0,
        index2pose_hb1,
        index2expression_hb0,
        index2expression_hb1,
    )
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        evaluator.reset_counters()
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                batch = {key: val.to(device) for key, val in batch.items()}
                probs = model(batch["ids_text"], batch["attn_mask"])
                # Loss and backward
                loss = criterion(probs, batch["target_visuals"])
                loss.backward()
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                # Update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        with torch.no_grad():
            for batch in tqdm(val_loader):
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
                    one_hot_pred.cpu().numpy(), batch["target_visuals"].cpu().numpy()
                )

        precision, recall, f1_score = evaluator.per_object_prf()
        posses_acc, expr_acc = evaluator.posses_expressions_accuracy()
        total_score = f1_score + posses_acc + expr_acc
        if total_score > best_score:
            best_score = total_score
            logging.info("====================================================")
            logging.info(f"Found best on epoch {epoch+1}. Saving model!")
            logging.info(
                f"Precison is {precision}, recall is {recall}, f1 score is {f1_score}"
            )
            logging.info(
                f"Posess accuracy is {posses_acc}, and expressions accuracy is {expr_acc}"
            )
            logging.info("====================================================")
            torch.save(model.state_dict(), args.save_model_path)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains a Cliparts prediction model.")
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="data/train_dataset.json",
        help="Path to the train dataset.",
    )
    parser.add_argument(
        "--val_dataset_path",
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
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="The number of epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="The learning rate."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="The weight decay."
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/best.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The bert model name.",
    )
    parser.add_argument(
        "--log_filepath", type=str, default=None, help="The logging file."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
