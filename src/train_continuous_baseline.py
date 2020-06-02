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
from scene_layouts.evaluator import abs_distance, relative_distance, Evaluator
from rnn_baseline.modeling import ArrangementsContinuousDecoder
from rnn_baseline.datasets import (
    TrainDataset,
    InferenceDataset,
    collate_pad_batch,
    build_vocab,
)
from scene_layouts.datasets import BUCKET_SIZE, O_PAD


def train(
    checkpoint_path: str,
    train_dataset_path: str,
    val_dataset_path: str,
    visual2index_path: str,
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
    word2freq, word2index, index2word = build_vocab(json.load(open(train_dataset_path)))
    train_dataset = TrainDataset(
        train_dataset_path, word2freq, word2index, visual2index
    )
    val_dataset = InferenceDataset(
        val_dataset_path, word2freq, word2index, visual2index
    )
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
        collate_fn=collate_pad_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_batch,
        sampler=val_sampler,
    )
    # Define training specifics
    num_cliparts = len(visual2index) + 1
    vocab_size = len(word2index)
    model = nn.DataParallel(
        ArrangementsContinuousDecoder(num_cliparts, vocab_size, 256, device)
    ).to(device)
    # Loss and optimizer
    criterion_f = nn.NLLLoss()
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
        best_avg_metrics = checkpoint["metrics"]
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
            for ids_text, ids_vis, x_lab, y_lab, o_lab, attn_mask in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                ids_text, ids_vis, x_lab, y_lab, o_lab, attn_mask = (
                    ids_text.to(device),
                    ids_vis.to(device),
                    x_lab.to(device),
                    y_lab.to(device),
                    o_lab.to(device),
                    attn_mask.to(device),
                )
                x_scores, y_scores, o_scores = model(ids_text, ids_vis)
                # Get losses for the absolute and relative distances
                abs_loss = (
                    abs_distance(x_scores, x_lab, y_scores, y_lab, attn_mask).sum()
                    / ids_text.size()[0]
                )
                relative_loss = (
                    relative_distance(x_scores, x_lab, y_scores, y_lab, attn_mask).sum()
                    / ids_text.size()[0]
                )
                o_loss = criterion_f(o_scores.view(-1, O_PAD - 1), o_lab.view(-1)) * 10
                # Backward
                loss = abs_loss + relative_loss + o_loss
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
        evaluator.reset_metrics()
        with torch.no_grad():
            for ids_text, ids_vis, x_lab, y_lab, o_lab, attn_mask in tqdm(val_loader):
                # forward
                ids_text, ids_vis, x_lab, y_lab, o_lab, attn_mask = (
                    ids_text.to(device),
                    ids_vis.to(device),
                    x_lab.to(device),
                    y_lab.to(device),
                    o_lab.to(device),
                    attn_mask.to(device),
                )
                x_scores, y_scores, o_scores = model(ids_text, ids_vis)
                x_out, y_out = (
                    x_scores * BUCKET_SIZE + BUCKET_SIZE / 2,
                    y_scores * BUCKET_SIZE + BUCKET_SIZE / 2,
                )
                o_out = torch.argmax(o_scores, dim=-1)
                evaluator.update_metrics(
                    x_out, x_lab, y_out, y_lab, o_out, o_lab, attn_mask
                )
        abs_dist = evaluator.get_abs_dist()
        rel_dist = evaluator.get_rel_dist()
        o_acc = evaluator.get_o_acc()
        cur_avg_metrics = (abs_dist + rel_dist - o_acc) / 3
        if cur_avg_metrics < best_avg_metrics:
            best_avg_metrics = cur_avg_metrics
            logging.info("====================================================")
            logging.info("Found new best with average metrics per scene:")
            logging.info(f"- Absolute distance: {abs_dist}")
            logging.info(f"- Relative distance: {rel_dist}")
            logging.info(f"- Flip accuracy: {o_acc}")
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
    parser = argparse.ArgumentParser(description="Trains a RNN baseline model.")
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
        "--weight_decay", type=float, default=0.0, help="The weight decay."
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
