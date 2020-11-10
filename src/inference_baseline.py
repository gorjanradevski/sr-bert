import argparse
import json
import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnn_baseline.datasets import InferenceDataset, build_vocab, collate_pad_batch
from rnn_baseline.modeling import baseline_factory
from scene_layouts.datasets import BUCKET_SIZE
from scene_layouts.evaluator import Evaluator


@torch.no_grad()
def inference(args):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(
        open(os.path.join(args.visuals_dicts_path, "visual2index.json"))
    )
    word2freq, word2index, _ = build_vocab(json.load(open(args.train_dataset_path)))
    test_dataset = InferenceDataset(
        args.test_dataset_path, word2freq, word2index, visual2index
    )
    logging.info(f"Performing inference on {len(test_dataset)}")
    # Create loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_pad_batch,
    )
    num_cliparts = len(visual2index) + 1
    vocab_size = len(word2index)
    model = nn.DataParallel(
        baseline_factory(args.baseline_name, num_cliparts, vocab_size, 256, device)
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.train(False)
    print(f"Starting inference from checkpoint {args.checkpoint_path}!")
    evaluator = Evaluator(len(test_dataset))
    for batch in tqdm(test_loader):
        # forward
        batch = {key: val.to(device) for key, val in batch.items()}
        x_scores, y_scores, o_scores = model(batch["ids_text"], batch["ids_vis"])
        if "discrete" in args.baseline_name:
            x_out, y_out = (
                torch.argmax(x_scores, dim=-1) * BUCKET_SIZE + BUCKET_SIZE / 2,
                torch.argmax(y_scores, dim=-1) * BUCKET_SIZE + BUCKET_SIZE / 2,
            )
        elif "continuous" in args.baseline_name:
            x_out, y_out = (
                x_scores * BUCKET_SIZE + BUCKET_SIZE / 2,
                y_scores * BUCKET_SIZE + BUCKET_SIZE / 2,
            )
        else:
            raise ValueError("Invalid model type!")
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

    print(
        f"The avg ABSOLUTE sim per scene is: {evaluator.get_abs_sim()} +/- {evaluator.get_abs_error_bar()}"
    )
    print(
        f"The avg RELATIVE sim per scene is: {evaluator.get_rel_sim()} +/- {evaluator.get_rel_error_bar()}"
    )
    print(f"The avg ACCURACY for the orientation is: {evaluator.get_o_acc()}")
    if args.abs_dump_path is not None and args.rel_dump_path is not None:
        evaluator.dump_results(args.abs_dump_path, args.rel_dump_path)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Inference with an RNN baseline model."
    )
    parser.add_argument(
        "--baseline_name", type=str, default=None, help="The baseline name."
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
        "--test_dataset_path",
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
        "--abs_dump_path",
        type=str,
        default=None,
        help="Location of the absolute similarity results.",
    )
    parser.add_argument(
        "--rel_dump_path",
        type=str,
        default=None,
        help="Location of the relative similarity results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    inference(args)


if __name__ == "__main__":
    main()
