import argparse
import json
import logging
import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnn_baseline.datasets import (
    InferenceDataset,
    TrainDataset,
    build_vocab,
    collate_pad_batch,
)
from rnn_baseline.modeling import baseline_factory
from scene_layouts.datasets import BUCKET_SIZE, O_PAD
from scene_layouts.evaluator import Evaluator
from scene_layouts.utils import abs_distance, relative_distance


def train(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(
        open(os.path.join(args.visuals_dicts_path, "visual2index.json"))
    )
    word2freq, word2index, _ = build_vocab(json.load(open(args.train_dataset_path)))
    train_dataset = TrainDataset(
        args.train_dataset_path, word2freq, word2index, visual2index
    )
    val_dataset = InferenceDataset(
        args.val_dataset_path, word2freq, word2index, visual2index
    )
    logging.info(f"Training on {len(train_dataset)}")
    logging.info(f"Validating on {len(val_dataset)}")
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_pad_batch,
    )
    # Define training specifics
    num_cliparts = len(visual2index) + 1
    vocab_size = len(word2index)
    model = nn.DataParallel(
        baseline_factory(args.baseline_name, num_cliparts, vocab_size, 256, device)
    ).to(device)
    # Loss and optimizer
    criterion_f = nn.NLLLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_avg_metrics = -1.0
    evaluator = Evaluator(len(val_dataset))
    for epoch in range(args.epochs):
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                batch = {key: val.to(device) for key, val in batch.items()}
                x_scores, y_scores, o_scores = model(
                    batch["ids_text"], batch["ids_vis"]
                )
                # Get losses for the absolute and relative similarities
                abs_loss = (
                    abs_distance(
                        x_scores,
                        batch["x_lab"],
                        y_scores,
                        batch["y_lab"],
                        batch["attn_mask"],
                    ).sum()
                    / batch["ids_text"].size()[0]
                )
                relative_loss = (
                    relative_distance(
                        x_scores,
                        batch["x_lab"],
                        y_scores,
                        batch["y_lab"],
                        batch["attn_mask"],
                    ).sum()
                    / batch["ids_text"].size()[0]
                )
                o_loss = (
                    criterion_f(o_scores.view(-1, O_PAD - 1), batch["o_lab"].view(-1))
                    * 10
                )
                # Backward
                # Minus because the loss are according to the similarity
                loss = abs_loss + relative_loss + o_loss
                loss.backward()
                # clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                # update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        # Reset counters
        evaluator.reset_metrics()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                # forward
                batch = {key: val.to(device) for key, val in batch.items()}
                x_scores, y_scores, o_scores = model(
                    batch["ids_text"], batch["ids_vis"]
                )
                x_out, y_out = (
                    x_scores * BUCKET_SIZE + BUCKET_SIZE / 2,
                    y_scores * BUCKET_SIZE + BUCKET_SIZE / 2,
                )
                o_out = torch.argmax(o_scores, dim=-1)
                evaluator.update_metrics(
                    x_out,
                    batch["x_lab"],
                    y_out,
                    batch["y_lab"],
                    o_out,
                    batch["o_lab"],
                    batch["attn_mask"],
                )
        abs_sim = evaluator.get_abs_sim()
        rel_sim = evaluator.get_rel_sim()
        o_acc = evaluator.get_o_acc() / 100  # All normalized
        cur_avg_metrics = (abs_sim + rel_sim + o_acc) / 3
        if cur_avg_metrics > best_avg_metrics:
            best_avg_metrics = cur_avg_metrics
            logging.info("====================================================")
            logging.info("Found new best with average metrics per scene:")
            logging.info(f"- Absolute similarity: {abs_sim}")
            logging.info(f"- Relative similarity: {rel_sim}")
            logging.info(f"- Orientation accuracy: {o_acc}")
            logging.info(f"on epoch {epoch+1}. Saving model!!!")
            torch.save(model.state_dict(), args.save_model_path)
            logging.info("====================================================")
        else:
            logging.info(f"Avg metrics on epoch {epoch+1} is: {cur_avg_metrics}. ")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Trains a Continuous RNN baseline model."
    )
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
        "--learning_rate", type=float, default=2e-5, help="The learning rate."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/best.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="The weight decay."
    )
    parser.add_argument(
        "--baseline_name",
        type=str,
        default="attn_continuous",
        help="Type of the continuous baseline",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
