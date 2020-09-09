import argparse
import torch
import torch.optim as optim
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
    train_dataset_path: str,
    val_dataset_path: str,
    visual2index_path: str,
    bert_name: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    clip_val: float,
    save_model_path: str,
    log_filepath: str,
):
    # Set up logging
    if log_filepath:
        logging.basicConfig(level=logging.INFO, filename=log_filepath, filemode="w")
    else:
        logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    train_dataset = ClipartsPredictionDataset(
        train_dataset_path, visual2index, train=True
    )
    val_dataset = ClipartsPredictionDataset(val_dataset_path, visual2index)
    logging.info(f"Training on {len(train_dataset)}")
    logging.info(f"Validating on {len(val_dataset)}")
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pad_cliparts_prediction_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_pad_cliparts_prediction_batch,
    )
    # Define training specifics
    config = BertConfig.from_pretrained(bert_name)
    config.vocab_size = len(visual2index)
    model = nn.DataParallel(ClipartsPredictionModel(config, bert_name)).to(device)
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_f_score = -1.0
    evaluator = ClipartsPredictionEvaluator(len(val_dataset), visual2index)
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        evaluator.reset_counters()
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for ids_text, attn_mask, target_visuals in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                ids_text, attn_mask, target_visuals = (
                    ids_text.to(device),
                    attn_mask.to(device),
                    target_visuals.to(device),
                )
                probs = model(ids_text, attn_mask)
                # Loss and backward
                loss = criterion(probs, target_visuals)
                loss.backward()
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                # Update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        with torch.no_grad():
            for ids_text, attn_mask, target_visuals in tqdm(val_loader):
                # forward
                ids_text, attn_mask, target_visuals = (
                    ids_text.to(device),
                    attn_mask.to(device),
                    target_visuals.to(device),
                )
                # Get predictions
                probs = torch.sigmoid(model(ids_text, attn_mask))
                one_hot_pred = torch.zeros_like(probs)
                # Regular objects
                one_hot_pred[:, :23][torch.where(probs[:, :23] > 0.5)] = 1
                one_hot_pred[:, 93:][torch.where(probs[:, 93:] > 0.5)] = 1
                # Mike and Jenny
                batch_indices = torch.arange(ids_text.size()[0])
                max_hb0 = torch.argmax(probs[:, 23:58], axis=-1)
                one_hot_pred[batch_indices, max_hb0 + 23] = 1
                max_hb1 = torch.argmax(probs[:, 58:93], axis=-1)
                one_hot_pred[batch_indices, max_hb1 + 58] = 1
                # Aggregate predictions/targets
                evaluator.update_counters(
                    one_hot_pred.cpu().numpy(), target_visuals.cpu().numpy()
                )

        cur_f_score = evaluator.f1_score()
        if cur_f_score > best_f_score:
            best_f_score = cur_f_score
            logging.info("====================================================")
            logging.info(
                f"Found best with F1 {best_f_score} on epoch {epoch+1}. Saving model!"
            )
            torch.save(model.state_dict(), save_model_path)
            logging.info("====================================================")


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
        "--visual2index_path",
        type=str,
        default="data/visual2index.json",
        help="Path to the visual2index mapping json.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="The number of epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="The learning rate."
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
    train(
        args.train_dataset_path,
        args.val_dataset_path,
        args.visual2index_path,
        args.bert_name,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.epochs,
        args.clip_val,
        args.save_model_path,
        args.log_filepath,
    )


if __name__ == "__main__":
    main()
