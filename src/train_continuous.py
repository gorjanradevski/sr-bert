import argparse
import json
import logging
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from scene_layouts.datasets import (
    BUCKET_SIZE,
    O_PAD,
    ContinuousInferenceDataset,
    ContinuousTrainDataset,
    collate_pad_batch,
)
from scene_layouts.evaluator import Evaluator
from scene_layouts.generation_strategies import train_cond
from scene_layouts.modeling import SpatialContinuousBert
from scene_layouts.utils import abs_distance, relative_distance


def train(args):
    # Set up logging
    if args.log_filepath:
        logging.basicConfig(
            level=logging.INFO, filename=args.log_filepath, filemode="w"
        )
    else:
        logging.basicConfig(level=logging.INFO)
    assert args.gen_strategy in [
        "one_step_all_continuous",
        "left_to_right_continuous",
        "one_step_all_left_to_right_continuous",
    ]
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(
        open(os.path.join(args.visuals_dicts_path, "visual2index.json"))
    )
    train_dataset = ContinuousTrainDataset(
        args.train_dataset_path, visual2index, mask_probability=args.mask_probability
    )
    val_dataset = ContinuousInferenceDataset(args.val_dataset_path, visual2index)
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
    config = BertConfig.from_pretrained(args.bert_name)
    config.vocab_size = len(visual2index) + 1  # Because of the padding token
    model = nn.DataParallel(SpatialContinuousBert(config, args.bert_name)).to(device)
    # Loss and optimizer
    criterion_o = nn.NLLLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    cur_epoch = 0
    best_avg_metrics = -1.0
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        best_avg_metrics = checkpoint["metrics"]
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint

        logging.warning(
            f"Starting training from checkpoint {args.checkpoint_path} with starting epoch {cur_epoch}!"
        )
        logging.warning(f"The previous best avg similarity was {best_avg_metrics}!")

    evaluator = Evaluator(len(val_dataset))
    for epoch in range(cur_epoch, args.epochs):
        start_time = datetime.now()
        logging.info(f"Starting epoch {epoch + 1} at {start_time}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                batch = {key: val.to(device) for key, val in batch.items()}
                x_scores, y_scores, o_scores = model(
                    batch["ids_text"],
                    batch["ids_vis"],
                    batch["pos_text"],
                    batch["x_ind"],
                    batch["y_ind"],
                    batch["o_ind"],
                    batch["t_types"],
                    batch["attn_mask"],
                )
                # Get loss for absolute and relative similarity
                batch_size, max_ids_text = batch["ids_text"].size()
                abs_loss = (
                    abs_distance(
                        x_scores,
                        batch["x_lab"],
                        y_scores,
                        batch["y_lab"],
                        batch["attn_mask"][:, max_ids_text:],
                    ).sum()
                    / batch["ids_text"].size()[0]
                )
                relative_loss = (
                    relative_distance(
                        x_scores,
                        batch["x_lab"],
                        y_scores,
                        batch["y_lab"],
                        batch["attn_mask"][:, max_ids_text:],
                    ).sum()
                    / batch["ids_text"].size()[0]
                )
                o_loss = criterion_o(
                    o_scores.view(-1, O_PAD + 1), batch["o_lab"].view(-1)
                )
                # Backward
                # Minus because the loss is computed according to the similarity
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
                x_out, y_out, o_out = train_cond(
                    "continuous",
                    batch["ids_text"],
                    batch["ids_vis"],
                    batch["pos_text"],
                    batch["x_ind"],
                    batch["y_ind"],
                    batch["o_ind"],
                    batch["t_types"],
                    batch["attn_mask"],
                    model,
                )
                x_out, y_out = (
                    x_out * BUCKET_SIZE + BUCKET_SIZE / 2,
                    y_out * BUCKET_SIZE + BUCKET_SIZE / 2,
                )
                evaluator.update_metrics(
                    x_out,
                    batch["x_lab"],
                    y_out,
                    batch["y_lab"],
                    o_out,
                    batch["o_lab"],
                    batch["attn_mask"][:, batch["ids_text"].size()[1] :],
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
        logging.info("Saving intermediate checkpoint...")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "similarity": best_avg_metrics,
            },
            args.intermediate_save_checkpoint_path,
        )
        logging.info(f"Finished epoch {epoch+1} in {datetime.now() - start_time}.")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Trains a Continuous Spatial-BERT model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to a pretrained model.",
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
        default="data/val_dataset.json",
        help="Path to the validation dataset.",
    )
    parser.add_argument(
        "--visuals_dicts_path",
        type=str,
        default="data/visuals_dicts/",
        help="Path to the directory with the visuals dictionaries.",
    )
    parser.add_argument(
        "--mask_probability", type=float, default=0.15, help="The mask probability."
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
        "--intermediate_save_checkpoint_path",
        type=str,
        default="models/intermediate.pt",
        help="Where to save the intermediate checkpoint.",
    )
    parser.add_argument(
        "--gen_strategy",
        type=str,
        default="left_to_right_continuous",
        help="How to generate the positions during inference",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="The weight decay."
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
