import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import sys
import logging
from transformers import BertForMaskedLM

from datasets import LinguisticScenesDataset, collate_pad_linguistic_batch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    checkpoint_path: str,
    train_dataset_path: str,
    test_dataset_path: str,
    mask_probability: float,
    batch_size: int,
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
    train_dataset = LinguisticScenesDataset(
        train_dataset_path, mask_probability=mask_probability
    )
    test_dataset = LinguisticScenesDataset(
        test_dataset_path, mask_probability=mask_probability
    )
    logger.info(f"Training on {len(train_dataset)}")
    logger.info(f"Validating on {len(test_dataset)}")
    # Create samplers
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(test_dataset)
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_pad_linguistic_batch,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_linguistic_batch,
        sampler=val_sampler,
    )
    # Define training specifics
    model = nn.DataParallel(BertForMaskedLM.from_pretrained("bert-base-uncased")).to(
        device
    )
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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
            for (input_ids, masked_lm_labels, attention_masks) in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                input_ids, masked_lm_labels, attention_masks = (
                    input_ids.to(device),
                    masked_lm_labels.to(device),
                    attention_masks.to(device),
                )
                predictions = model(
                    input_ids=input_ids, attention_mask=attention_masks
                )[0]
                predictions = predictions.view(
                    -1, model.module.bert.embeddings.word_embeddings.num_embeddings
                )
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
                for (input_ids, masked_lm_labels, attention_masks) in val_loader:
                    input_ids, masked_lm_labels, attention_masks = (
                        input_ids.to(device),
                        masked_lm_labels.to(device),
                        attention_masks.to(device),
                    )
                    predictions = model(
                        input_ids=input_ids, attention_mask=attention_masks
                    )[0]
                    predictions = predictions.view(
                        -1, model.module.bert.embeddings.word_embeddings.num_embeddings
                    )
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                intermediate_save_checkpoint_path,
            )


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
        "--train_dataset_path",
        type=str,
        default="data/train_dataset.json",
        help="Path to the train dataset.",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test dataset.",
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
        default="models/linguistic_best.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--intermediate_save_checkpoint_path",
        type=str,
        default="models/linguistic_intermediate.pt",
        help="Where to save the model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train(
        args.checkpoint_path,
        args.train_dataset_path,
        args.test_dataset_path,
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
