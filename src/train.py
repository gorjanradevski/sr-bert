import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import logging

from datasets import ScenesDatasetTrain, ScenesDatasetVal, collate_pad_batch
from modeling import SceneModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    train_dataset_path: str,
    val_dataset_path: str,
    visual2index_path: str,
    mask_probability: float,
    batch_size: int,
    load_embeddings_path: str,
    learning_rate: float,
    epochs: int,
    clip_val: float,
    save_model_path: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ScenesDatasetTrain(
        train_dataset_path, visual2index_path, mask_probability=0.3
    )
    val_dataset = ScenesDatasetVal(
        val_dataset_path, visual2index_path, mask_probability=0.3
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_pad_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_pad_batch
    )
    model = nn.DataParallel(SceneModel(load_embeddings_path)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = sys.maxsize
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}...")
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for (
                input_ids,
                masked_lm_labels,
                text_positions,
                visual_positions,
                token_type_ids,
            ) in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                input_ids, masked_lm_labels, text_positions, visual_positions, token_type_ids = (
                    input_ids.to(device),
                    masked_lm_labels.to(device),
                    text_positions.to(device),
                    visual_positions.to(device),
                    token_type_ids.to(device),
                )
                predictions = model(
                    input_ids, text_positions, visual_positions, token_type_ids
                )
                # TODO: Change to a variable instead of hardcoding the value
                predictions = predictions.view(-1, 31277)
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
            ) in tqdm(val_loader):
                input_ids, masked_lm_labels, text_positions, visual_positions, token_type_ids = (
                    input_ids.to(device),
                    masked_lm_labels.to(device),
                    text_positions.to(device),
                    visual_positions.to(device),
                    token_type_ids.to(device),
                )
                predictions = model(
                    input_ids, text_positions, visual_positions, token_type_ids
                )
                # TODO: Change to a variable instead of hardcoding the value
                predictions = predictions.view(-1, 31277)
                masked_lm_labels = masked_lm_labels.view(-1)
                loss = criterion(predictions, masked_lm_labels)
                cur_val_loss += loss.item()

            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                print("======================")
                print(
                    f"Found new best with loss{best_val_loss} on epoch "
                    f"{epoch+1}. Saving model!!!"
                )
                torch.save(model.state_dict(), save_model_path)
                print("======================")
            else:
                print(f"Loss on epoch {epoch+1} is: {cur_val_loss}")


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains a Scene model.")
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
        "--load_embeddings_path",
        type=str,
        default="models/updated_embeddings.pt",
        help="From where to load the embeddings.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="The learning rate."
    )
    parser.add_argument(
        "--clip_val", type=float, default=2.0, help="The clipping threshold."
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/best_model.pt",
        help="Where to save the model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train(
        args.train_dataset_path,
        args.val_dataset_path,
        args.visual2index_path,
        args.mask_probability,
        args.batch_size,
        args.load_embeddings_path,
        args.learning_rate,
        args.epochs,
        args.clip_val,
        args.save_model_path,
    )


if __name__ == "__main__":
    main()
