import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import sys
import logging
import json
from transformers import BertConfig
from utils import relative_distance, real_distance

from datasets import (
    Text2VisualContinuousDataset,
    collate_pad_continuous_text2visual_batch,
    F_PAD,
    X_MASK,
    Y_MASK,
    F_MASK,
)
from modeling import Text2VisualContinuousBert


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    checkpoint_path: str,
    train_dataset_path: str,
    val_dataset_path: str,
    visual2index_path: str,
    mask_probability: float,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    clip_val: float,
    save_model_path: str,
    intermediate_save_checkpoint_path: str,
):
    # https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    train_dataset = Text2VisualContinuousDataset(
        train_dataset_path, visual2index, mask_probability=mask_probability, train=True
    )
    val_dataset = Text2VisualContinuousDataset(
        val_dataset_path, visual2index, mask_probability=1.0, train=False
    )
    logger.info(f"Training on {len(train_dataset)}")
    logger.info(f"Validating on {len(val_dataset)}")
    # Create samplers
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_pad_continuous_text2visual_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_continuous_text2visual_batch,
        sampler=val_sampler,
    )
    # Define training specifics
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.vocab_size = len(visual2index) + 3
    model = nn.DataParallel(Text2VisualContinuousBert(config, device)).to(device)
    # Loss and optimizer
    criterion_f = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cur_epoch = 0
    best_avg_distance = sys.maxsize
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        best_avg_distance = checkpoint["distance"]
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint

        logger.warning(
            f"Starting training from checkpoint {checkpoint_path} with starting epoch {cur_epoch}!"
        )
        logger.warning(f"The previous best avg distance was {best_avg_distance}!")

    for epoch in range(cur_epoch, epochs):
        logger.info(f"Starting epoch {epoch + 1}...")
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
                # Get losses for the real distances
                max_ids_text = ids_text.size()[1]
                x_real_loss = (
                    real_distance(
                        x_scores[:, max_ids_text:],
                        x_lab[:, max_ids_text:],
                        attn_mask[:, max_ids_text:],
                    )
                    / ids_text.size()[0]
                ) * 0.35
                y_real_loss = (
                    real_distance(
                        y_scores[:, max_ids_text:],
                        y_lab[:, max_ids_text:],
                        attn_mask[:, max_ids_text:],
                    )
                    / ids_text.size()[0]
                ) * 0.35
                x_relative_loss = (
                    relative_distance(
                        x_scores[:, max_ids_text:],
                        x_lab[:, max_ids_text:],
                        attn_mask[:, max_ids_text:],
                    )
                    / ids_text.size()[0]
                )
                y_relative_loss = (
                    relative_distance(
                        y_scores[:, max_ids_text:],
                        y_lab[:, max_ids_text:],
                        attn_mask[:, max_ids_text:],
                    )
                    / ids_text.size()[0]
                )
                f_loss = criterion_f(f_scores.view(-1, F_PAD + 1), f_lab.view(-1))
                # Comibine losses and backward
                loss = (
                    x_real_loss
                    + y_real_loss
                    + f_loss
                    + x_relative_loss
                    + y_relative_loss
                )
                loss.backward()
                # clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                # update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        # Reset counters
        total_dist_x_relative = 0
        total_dist_y_relative = 0
        total_dist_x_real = 0
        total_dist_y_real = 0
        total_acc_f = 0
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
                # Set all indices to MASK tokens
                x_ind[:, :] = X_MASK
                y_ind[:, :] = Y_MASK
                f_ind[:, :] = F_MASK
                for iteration in range(2):
                    first = torch.cat([x_ind, y_ind, f_ind], dim=1).cpu()
                    for i in range(ids_vis.size()[1]):
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
                        x_ind[:, i] = X_MASK
                        y_ind[:, i] = Y_MASK
                        f_ind[:, i] = F_MASK
                        max_ids_text = ids_text.size()[1]
                        x_scores, y_scores, f_scores = model(
                            ids_text,
                            ids_vis,
                            pos_text,
                            x_ind,
                            y_ind,
                            f_ind,
                            t_types,
                            attn_mask,
                        )
                        x_ind[:, i] = torch.ceil(x_scores[:, max_ids_text:][:, i])
                        y_ind[:, i] = torch.ceil(y_scores[:, max_ids_text:][:, i])
                        f_ind[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text:][
                            :, i
                        ]

                    # Check for termination
                    last = torch.cat([x_ind, y_ind, f_ind], dim=1).cpu()
                    if torch.all(torch.eq(first, last)):
                        break
                total_dist_x_relative += relative_distance(
                    x_ind, x_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
                ).item()
                total_dist_y_relative += relative_distance(
                    y_ind, y_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
                ).item()
                total_dist_x_real += real_distance(
                    x_ind, x_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
                ).item()
                total_dist_y_real += real_distance(
                    y_ind, y_lab[:, max_ids_text:], attn_mask[:, max_ids_text:]
                ).item()
                total_acc_f += (
                    f_ind == f_lab[:, max_ids_text:]
                ).sum().item() / f_ind.size()[1]

            total_dist_x_relative /= len(val_dataset)
            total_dist_y_relative /= len(val_dataset)
            total_dist_x_real /= len(val_dataset)
            total_dist_y_real /= len(val_dataset)
            cur_avg_distance = round(
                (
                    total_dist_x_relative
                    + total_dist_y_relative
                    + total_dist_x_real
                    + total_dist_y_real
                )
                / 4
            )
            if cur_avg_distance < best_avg_distance:
                best_avg_distance = cur_avg_distance
                print("====================================================")
                print("Found new best with average distances per scene:")
                print(f"- X relative distance: {round(total_dist_x_relative, 2)}")
                print(f"- Y relative distance: {round(total_dist_y_relative, 2)}")
                print(f"- X real distance: {round(total_dist_x_real, 2)}")
                print(f"- Y real distance: {round(total_dist_y_real, 2)}")
                print(f"on epoch {epoch+1}. Saving model!!!")
                torch.save(model.state_dict(), save_model_path)
                print("====================================================")
            else:
                print(f"Avg distance on epoch {epoch+1} is: {cur_avg_distance}. ")
            print("Saving intermediate checkpoint...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "distance": best_avg_distance,
                },
                intermediate_save_checkpoint_path,
            )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains a Text2Position model.")
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

    return parser.parse_args()


def main():
    args = parse_args()
    train(
        args.checkpoint_path,
        args.train_dataset_path,
        args.val_dataset_path,
        args.visual2index_path,
        args.mask_probability,
        args.batch_size,
        args.learning_rate,
        args.epochs,
        args.clip_val,
        args.save_model_path,
        args.intermediate_save_checkpoint_path,
    )


if __name__ == "__main__":
    main()
