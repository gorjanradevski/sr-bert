import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
import os
from scene_layouts.evaluator import Evaluator
from rnn_baseline.modeling import baseline_factory
from rnn_baseline.datasets import InferenceDataset, collate_pad_batch, build_vocab
from scene_layouts.datasets import BUCKET_SIZE


def inference(
    baseline_name: str,
    checkpoint_path: str,
    train_dataset_path: str,
    test_dataset_path: str,
    visuals_dicts_path: str,
    batch_size: int,
    abs_dump_path: str,
    rel_dump_path: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(
        open(os.path.join(visuals_dicts_path, "visual2index.json"))
    )
    word2freq, word2index, index2word = build_vocab(json.load(open(train_dataset_path)))
    test_dataset = InferenceDataset(
        test_dataset_path, word2freq, word2index, visual2index
    )
    logging.info(f"Performing inference on {len(test_dataset)}")
    # Create loaders
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_pad_batch
    )
    num_cliparts = len(visual2index) + 1
    vocab_size = len(word2index)
    model = nn.DataParallel(
        baseline_factory(baseline_name, num_cliparts, vocab_size, 256, device)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)
    print(f"Starting inference from checkpoint {checkpoint_path}!")
    evaluator = Evaluator(len(test_dataset))
    with torch.no_grad():
        for ids_text, ids_vis, x_lab, y_lab, o_lab, attn_mask in tqdm(test_loader):
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
            if "discrete" in baseline_name:
                x_out, y_out = (
                    torch.argmax(x_scores, dim=-1) * BUCKET_SIZE + BUCKET_SIZE / 2,
                    torch.argmax(y_scores, dim=-1) * BUCKET_SIZE + BUCKET_SIZE / 2,
                )
            elif "continuous" in baseline_name:
                x_out, y_out = (
                    x_scores * BUCKET_SIZE + BUCKET_SIZE / 2,
                    y_scores * BUCKET_SIZE + BUCKET_SIZE / 2,
                )
            else:
                raise ValueError("Invalid model type!")
            o_out = torch.argmax(o_scores, dim=-1)
            evaluator.update_metrics(
                x_out, x_lab, y_out, y_lab, o_out, o_lab, attn_mask
            )

        print(
            f"The avg ABSOLUTE dst per scene is: {evaluator.get_abs_sim()} +/- {evaluator.get_abs_error_bar()}"
        )
        print(
            f"The avg RELATIVE dst per scene is: {evaluator.get_rel_sim()} +/- {evaluator.get_rel_error_bar()}"
        )
        print(f"The avg ACCURACY for the orientation is: {evaluator.get_o_acc()}")
        if abs_dump_path is not None and rel_dump_path is not None:
            evaluator.dump_results(abs_dump_path, rel_dump_path)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Inference with a RNN baseline model.")
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
    inference(
        args.baseline_name,
        args.checkpoint_path,
        args.train_dataset_path,
        args.test_dataset_path,
        args.visuals_dicts_path,
        args.batch_size,
        args.abs_dump_path,
        args.rel_dump_path,
    )


if __name__ == "__main__":
    main()
