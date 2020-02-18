import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from tqdm import tqdm
import sys
import logging
import json
from transformers import BertConfig

from datasets import (
    Text2VisualDataset,
    collate_pad_text2visual_batch,
    X_PAD,
    Y_PAD,
    F_PAD,
)
from modeling import Text2VisualBert


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    embeddings_path: str,
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
    train_dataset = Subset(
        Text2VisualDataset(
            train_dataset_path,
            visual2index,
            mask_probability=mask_probability,
            train=False,
        ),
        [0, 1],
    )
    val_dataset = Text2VisualDataset(
        val_dataset_path, visual2index, mask_probability=mask_probability, train=False
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
        collate_fn=collate_pad_text2visual_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_text2visual_batch,
        sampler=val_sampler,
    )
    # Define training specifics
    model = nn.DataParallel(Text2VisualBert(BertConfig(), device, embeddings_path)).to(
        device
    )
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cur_epoch = 0
    best_val_loss = sys.maxsize
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cur_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["loss"]
        # https://discuss.pytorch.org/t/cuda-out-of-memory-after-loading-model/50681
        del checkpoint

        logger.warning(
            f"Starting training from checkpoint {checkpoint_path} with starting epoch {cur_epoch}!"
        )
        logger.warning(f"The previous best loss was {best_val_loss}!")

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
                    t_types,
                    attn_mask,
                )
                x_scores, y_scores, f_scores = model(
                    ids_text, ids_vis, pos_text, x_ind, y_ind, f_ind, t_types, attn_mask
                )
                # Get losses
                x_loss = criterion(x_scores.view(-1, X_PAD + 1), x_lab.view(-1))
                y_loss = criterion(y_scores.view(-1, Y_PAD + 1), y_lab.view(-1))
                f_loss = criterion(f_scores.view(-1, F_PAD + 1), f_lab.view(-1))
                # Comibine losses and backward
                loss = x_loss + y_loss + f_loss
                loss.backward()
                # clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                # update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})
        """
        # Set model in evaluation mode
        model.train(False)
        with torch.no_grad():
            # Reset current loss
            cur_val_loss = 0
            for _ in tqdm(range(10)):
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
                ) in val_loader:
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
                        t_types,
                        attn_mask,
                    )
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
                    # Get losses
                    x_loss = criterion(x_scores.view(-1, X_PAD + 1), x_lab.view(-1))
                    y_loss = criterion(y_scores.view(-1, Y_PAD + 1), y_lab.view(-1))
                    f_loss = criterion(f_scores.view(-1, F_PAD + 1), f_lab.view(-1))
                    # Comibine losses
                    loss = x_loss + y_loss + f_loss
                    cur_val_loss += loss.item()

            cur_val_loss /= 10
            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                print("======================")
                print(
                    f"Found new best with loss {best_val_loss} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                torch.save(model.state_dict(), save_model_path)
                print("======================")
            else:
                print(f"Loss on epoch {epoch+1} is: {cur_val_loss}. ")
            print("Saving intermediate checkpoint...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                intermediate_save_checkpoint_path,
            )
            """


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains a Text2Position model.")
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="models/cliparts_embeddings.pt",
        help="Path to an embedding matrix",
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
        args.embeddings_path,
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
