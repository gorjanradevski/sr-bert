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

from datasets import VisualScenesDataset, collate_pad_visual_batch
from modeling import VisualBert


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    use_cuda: bool,
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
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    logger.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    train_dataset = VisualScenesDataset(
        train_dataset_path, visual2index, mask_probability=mask_probability
    )
    val_dataset = VisualScenesDataset(
        val_dataset_path, visual2index, mask_probability=mask_probability
    )
    # Create samplers
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_pad_visual_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_visual_batch,
        sampler=val_sampler,
    )
    # Define training specifics
    config = BertConfig(vocab_size=len(visual2index) + 3)
    model = VisualBert(config)
    if checkpoint_path is not None:
        logger.warning(f"Starting training from checkpoint {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if use_cuda:
        model = nn.DataParallel(model).to(device)
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = sys.maxsize
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for (
                input_ids_visuals,
                masked_lm_labels,
                visual_positions,
                attention_masks,
            ) in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                input_ids_visuals, masked_lm_labels, visual_positions, attention_masks = (
                    input_ids_visuals.to(device),
                    masked_lm_labels.to(device),
                    visual_positions.to(device),
                    attention_masks.to(device),
                )
                predictions = model(
                    input_ids_visuals, visual_positions, attention_masks
                )
                predictions = predictions.view(-1, len(visual2index) + 3)
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
            for _ in tqdm(range(10)):
                for (
                    input_ids_visuals,
                    masked_lm_labels,
                    visual_positions,
                    attention_masks,
                ) in val_loader:
                    input_ids_visuals, masked_lm_labels, visual_positions, attention_masks = (
                        input_ids_visuals.to(device),
                        masked_lm_labels.to(device),
                        visual_positions.to(device),
                        attention_masks.to(device),
                    )
                    predictions = model(
                        input_ids_visuals, visual_positions, attention_masks
                    )
                    predictions = predictions.view(-1, len(visual2index) + 3)
                    masked_lm_labels = masked_lm_labels.view(-1)
                    loss = criterion(predictions, masked_lm_labels)
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
            torch.save(model.state_dict(), intermediate_save_checkpoint_path)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains a Scene model.")
    parser.add_argument("--use_cuda", action="store_true", help="Whether to use cuda.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to a pretrained model.",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="data/dataset_v2_train.json",
        help="Path to the train dataset.",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default="data/dataset_v2_val.json",
        help="Path to the val dataset.",
    )
    parser.add_argument(
        "--visual2index_path",
        type=str,
        default="data/visual2index.json",
        help="Path to the visual2index mapping json.",
    )
    parser.add_argument(
        "--mask_probability", type=float, default=0.3, help="The mask probability."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs.")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="The learning rate."
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
        help="Where to save the model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train(
        args.use_cuda,
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
