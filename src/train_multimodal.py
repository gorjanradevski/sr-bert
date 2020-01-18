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

from datasets import MultimodalScenesDataset, collate_pad_multimodal_batch
from modeling import MultiModalBert


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    checkpoint_path: str,
    val_size: int,
    dataset_path: str,
    visual2index_path: str,
    mask_probability: float,
    batch_size: int,
    load_embeddings_path: str,
    learning_rate: float,
    epochs: int,
    clip_val: float,
    save_model_path: str,
    intermediate_save_checkpoint_path: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    dataset = MultimodalScenesDataset(
        dataset_path, visual2index, mask_probability=mask_probability
    )
    # train_size = len(dataset) - val_size
    train_dataset = Subset(dataset, list(range(0, len(dataset), 4000)))
    # val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    logger.info(f"Training on {len(train_dataset)}")
    logger.info(f"Validating on {len(train_dataset)}")
    # Create samplers
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(train_dataset)
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_pad_multimodal_batch,
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_multimodal_batch,
        sampler=val_sampler,
    )
    # Define training specifics
    config = BertConfig()
    config.vocab_size += len(visual2index)
    model = nn.DataParallel(MultiModalBert(load_embeddings_path, config, device)).to(
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
        logger.warning(f"Starting training from checkpoint {checkpoint_path}")

    for epoch in range(cur_epoch, epochs):
        logger.info(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for (
                input_ids,
                masked_lm_labels,
                text_positions,
                visual_positions,
                token_type_ids,
                attention_masks,
            ) in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                input_ids, masked_lm_labels, text_positions, visual_positions, token_type_ids, attention_masks = (
                    input_ids.to(device),
                    masked_lm_labels.to(device),
                    text_positions.to(device),
                    visual_positions.to(device),
                    token_type_ids.to(device),
                    attention_masks.to(device),
                )
                predictions = model(
                    input_ids,
                    text_positions,
                    visual_positions,
                    token_type_ids,
                    attention_masks,
                )
                predictions = predictions.view(-1, config.vocab_size)
                masked_lm_labels = masked_lm_labels.view(-1)
                loss = criterion(predictions, masked_lm_labels)
                # backward
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
        with torch.no_grad():
            # Reset current loss
            cur_val_loss = 0
            for (
                input_ids,
                masked_lm_labels,
                text_positions,
                visual_positions,
                token_type_ids,
                attention_masks,
            ) in val_loader:
                input_ids, masked_lm_labels, text_positions, visual_positions, token_type_ids, attention_masks = (
                    input_ids.to(device),
                    masked_lm_labels.to(device),
                    text_positions.to(device),
                    visual_positions.to(device),
                    token_type_ids.to(device),
                    attention_masks.to(device),
                )
                predictions = model(
                    input_ids,
                    text_positions,
                    visual_positions,
                    token_type_ids,
                    attention_masks,
                )
                predictions = predictions.view(-1, config.vocab_size)
                masked_lm_labels = masked_lm_labels.view(-1)
                loss = criterion(predictions, masked_lm_labels)
                cur_val_loss += loss.item()
                print(f"Inputs: {input_ids}")
                print(
                    f"Predictions: {torch.argmax(predictions, dim=1).view(input_ids.size())}"
                )
                print(f"Labels: {masked_lm_labels.view(input_ids.size())}")
            """
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
                save_model_path,
            )
            """


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains a MultimodalBert model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to a trained model.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/dataset.json",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--visual2index_path",
        type=str,
        default="data/visual2index.json",
        help="Path to the visual2index mapping json.",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
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
        "--load_embeddings_path",
        type=str,
        default="models/clipart_embeddings.pt",
        help="From where to load the embeddings.",
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
        default="models/multimodal_best.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--intermediate_save_checkpoint_path",
        type=str,
        default="models/multimodal_intermediate.pt",
        help="Where to save the model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train(
        args.checkpoint_path,
        args.val_size,
        args.dataset_path,
        args.visual2index_path,
        args.mask_probability,
        args.batch_size,
        args.load_embeddings_path,
        args.learning_rate,
        args.epochs,
        args.clip_val,
        args.save_model_path,
        args.intermediate_save_checkpoint_path,
    )


if __name__ == "__main__":
    main()
