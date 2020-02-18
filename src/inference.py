import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import logging
import json
from transformers import BertConfig

from datasets import (
    Text2VisualDataset,
    collate_pad_text2visual_batch,
    X_MASK,
    Y_MASK,
    F_MASK,
)
from modeling import Text2VisualBert


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inference(
    embeddings_path: str,
    checkpoint_path: str,
    test_dataset_path: str,
    visual2index_path: str,
    num_iter: int,
):
    # https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"--- Using device {device}! ---")
    # Create datasets
    visual2index = json.load(open(visual2index_path))
    test_dataset = Text2VisualDataset(
        test_dataset_path, visual2index, mask_probability=1.0, train=False
    )
    logger.info(f"Testing on {len(test_dataset)}")
    # Create samplers
    test_sampler = SequentialSampler(test_dataset)
    # Create loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_pad_text2visual_batch,
        sampler=test_sampler,
    )
    # Prepare model
    model = nn.DataParallel(Text2VisualBert(BertConfig(), device, embeddings_path)).to(
        device
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.train(False)
    # Criterion
    total_dist_x = 0
    total_dist_y = 0
    total_acc_f = 0
    logger.warning(f"Starting inference from checkpoint {checkpoint_path}!")
    # Set model in evaluation mode
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
        ) in tqdm(test_loader):
            # Set all indices to MASK tokens
            x_ind[:, :] = X_MASK
            y_ind[:, :] = Y_MASK
            f_ind[:, :] = F_MASK
            for iteration in range(num_iter):
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
                    x_ind[:, i] = torch.argmax(x_scores, dim=-1)[:, max_ids_text:][:, i]
                    y_ind[:, i] = torch.argmax(y_scores, dim=-1)[:, max_ids_text:][:, i]
                    f_ind[:, i] = torch.argmax(f_scores, dim=-1)[:, max_ids_text:][:, i]
                # Check for termination
                last = torch.cat([x_ind, y_ind, f_ind], dim=1).cpu()
                if torch.all(torch.eq(first, last)):
                    break

            total_dist_x += torch.sum(
                torch.abs(x_ind - x_lab[:, max_ids_text:]).float()
                * attn_mask[:, max_ids_text:]
            ).item()
            total_dist_y += torch.sum(
                torch.abs(y_ind - y_lab[:, max_ids_text:]).float()
                * attn_mask[:, max_ids_text:]
            ).item()
            total_acc_f += (
                f_ind == f_lab[:, max_ids_text:]
            ).sum().item() / f_ind.size()[1]

        total_dist_x /= len(test_dataset)
        total_dist_y /= len(test_dataset)
        print(f"The average distance per scene for X is: {round(total_dist_x, 2)}")
        print(f"The average distance per scene for Y is: {round(total_dist_y, 2)}")
        print(
            f"The average accuracy per scene for F is: {total_acc_f/len(test_dataset)}"
        )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Performs inference with a Text2Position model."
    )
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
        "--test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--visual2index_path",
        type=str,
        default="data/visual2index.json",
        help="Path to the visual2index mapping json.",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=5,
        help="Number of iterations for the inference.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    inference(
        args.embeddings_path,
        args.checkpoint_path,
        args.test_dataset_path,
        args.visual2index_path,
        args.num_iter,
    )


if __name__ == "__main__":
    main()
