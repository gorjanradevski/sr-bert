import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import sys
import logging
from datetime import datetime
import json
from transformers import BertConfig

from scene_layouts.datasets import (
    DiscreteTrainDataset,
    DiscreteInferenceDataset,
    collate_pad_discrete_batch,
    X_PAD,
    Y_PAD,
    F_PAD,
    BUCKET_SIZE,
)
from scene_layouts.evaluator import Evaluator
from scene_layouts.generation_strategies import train_cond_discrete
from scene_layouts.modeling import SpatialDiscreteBert


def train(
    checkpoint_path: str,
    train_dataset_path: str,
    val_dataset_path: str,
    visual2index_path: str,
    mask_probability: float,
    bert_name: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    clip_val: float,
    save_model_path: str,
    intermediate_save_checkpoint_path: str,
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
    train_dataset = DiscreteTrainDataset(
        train_dataset_path, visual2index, mask_probability=mask_probability
    )
    val_dataset = DiscreteInferenceDataset(val_dataset_path, visual2index)
    logging.info(f"Training on {len(train_dataset)}")
    logging.info(f"Validating on {len(val_dataset)}")
    # Create samplers
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_pad_discrete_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_discrete_batch,
        sampler=val_sampler,
    )
    # Define training specifics
    config = BertConfig.from_pretrained(bert_name)
    config.vocab_size = len(visual2index) + 1
    model = nn.DataParallel(SpatialDiscreteBert(config, bert_name)).to(device)
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    cur_epoch = 0
    best_avg_metrics = sys.maxsize
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        best_avg_metrics = checkpoint["best_avg_metrics"]
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint

        logging.warning(
            f"Starting training from checkpoint {checkpoint_path} with starting epoch {cur_epoch}!"
        )
        logging.warning(f"The previous best avg distance was {best_avg_metrics}!")

    evaluator = Evaluator(len(val_dataset))
    for epoch in range(cur_epoch, epochs):
        start_time = datetime.now()
        logging.info(f"Starting epoch {epoch + 1} at {start_time}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for (
                ids_text,
                ids_vis,
                pos_text,
                x_ind,
                y_ind,
                f_ind,
                x_lab,
                y_lab,
                f_lab,
                t_types,
                attn_mask,
            ) in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, x_lab, y_lab, f_lab, t_types, attn_mask = (
                    ids_text.to(device),
                    ids_vis.to(device),
                    pos_text.to(device),
                    x_ind.to(device),
                    y_ind.to(device),
                    f_ind.to(device),
                    x_lab.to(device),
                    y_lab.to(device),
                    f_lab.to(device),
                    t_types.to(device),
                    attn_mask.to(device),
                )
                x_scores, y_scores, f_scores = model(
                    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
                )
                # Get losses as classification losses
                x_loss = criterion(x_scores.view(-1, X_PAD + 1), x_lab.view(-1))
                y_loss = criterion(y_scores.view(-1, Y_PAD + 1), y_lab.view(-1))
                f_loss = criterion(f_scores.view(-1, F_PAD + 1), f_lab.view(-1))
                # Comibine losses and backward
                loss = x_loss + y_loss + f_loss
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
        # Reset counters
        evaluator.reset_metrics()
        with torch.no_grad():
            for (
                ids_text,
                ids_vis,
                pos_text,
                x_ind,
                y_ind,
                f_ind,
                x_lab,
                y_lab,
                f_lab,
                t_types,
                attn_mask,
            ) in tqdm(val_loader):
                # forward
                ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, x_lab, y_lab, f_lab, t_types, attn_mask = (
                    ids_text.to(device),
                    ids_vis.to(device),
                    pos_text.to(device),
                    x_ind.to(device),
                    y_ind.to(device),
                    f_ind.to(device),
                    x_lab.to(device),
                    y_lab.to(device),
                    f_lab.to(device),
                    t_types.to(device),
                    attn_mask.to(device),
                )
                max_ids_text = ids_text.size()[1]
                x_out, y_out, f_out = train_cond_discrete(
                    ids_text,
                    ids_vis,
                    pos_text,
                    x_ind,
                    y_ind,
                    f_ind,
                    t_types,
                    attn_mask,
                    model,
                )
                x_out, y_out = (
                    x_out * BUCKET_SIZE + BUCKET_SIZE / 2,
                    y_out * BUCKET_SIZE + BUCKET_SIZE / 2,
                )
                evaluator.update_metrics(
                    x_out,
                    x_lab[:, max_ids_text:],
                    y_out,
                    y_lab[:, max_ids_text:],
                    f_out,
                    f_lab[:, max_ids_text:],
                    attn_mask[:, max_ids_text:],
                )

        abs_dist = evaluator.get_abs_dist()
        rel_dist = evaluator.get_rel_dist()
        f_acc = evaluator.get_f_acc()
        cur_avg_metrics = (abs_dist + rel_dist - f_acc) / 3
        if cur_avg_metrics < best_avg_metrics:
            best_avg_metrics = cur_avg_metrics
            logging.info("====================================================")
            logging.info("Found new best with average metrics per scene:")
            logging.info(f"- Absolute distance: {abs_dist}")
            logging.info(f"- Relative distance: {rel_dist}")
            logging.info(f"- Flip accuracy: {f_acc}")
            logging.info(f"on epoch {epoch+1}. Saving model!!!")
            torch.save(model.state_dict(), save_model_path)
            logging.info("====================================================")
        else:
            logging.info(f"Avg metrics on epoch {epoch+1} is: {cur_avg_metrics}. ")
        logging.info("Saving intermediate checkpoint...")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "distance": best_avg_metrics,
            },
            intermediate_save_checkpoint_path,
        )
        logging.info(f"Finished epoch {epoch+1} in {datetime.now() - start_time}.")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains a SpatialBERT model.")
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
        default="data/test_dataset.json",
        help="Path to the validation dataset.",
    )
    parser.add_argument(
        "--visual2index_path",
        type=str,
        default="data/visual2index.json",
        help="Path to the visual2index mapping json.",
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
        "--weight_decay", type=float, default=0.0, help="The weight decay."
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
        args.checkpoint_path,
        args.train_dataset_path,
        args.val_dataset_path,
        args.visual2index_path,
        args.mask_probability,
        args.bert_name,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.epochs,
        args.clip_val,
        args.save_model_path,
        args.intermediate_save_checkpoint_path,
        args.log_filepath,
    )


if __name__ == "__main__":
    main()
