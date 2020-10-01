import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
from transformers import BertConfig

from scene_layouts.datasets import (
    ClipartsPredictionDataset,
    collate_pad_cliparts_prediction_batch,
)
from scene_layouts.modeling import ClipartsPredictionModel
from scene_layouts.evaluator import ClipartsPredictionEvaluator


def train(
    train_dataset_path: str,
    val_dataset_path: str,
    visual2index_path: str,
    bert_name: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    clip_val: float,
    save_model_path: str,
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
    train_dataset = ClipartsPredictionDataset(
        train_dataset_path, visual2index, train=True
    )
    val_dataset = ClipartsPredictionDataset(val_dataset_path, visual2index)
    logging.info(f"Training on {len(train_dataset)}")
    logging.info(f"Validating on {len(val_dataset)}")
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pad_cliparts_prediction_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_pad_cliparts_prediction_batch,
    )
    # Define training specifics
    config = BertConfig.from_pretrained(bert_name)
    config.vocab_size = len(visual2index)
    model = nn.DataParallel(ClipartsPredictionModel(config, bert_name)).to(device)
    # Loss and optimizer
    object_criterion = nn.BCEWithLogitsLoss()
    pose_expr_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_score = -1.0
    evaluator = ClipartsPredictionEvaluator(len(val_dataset), visual2index)
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        evaluator.reset_counters()
        # Set model in train mode
        model.train(True)
        with tqdm(total=len(train_loader)) as pbar:
            for (
                ids_text,
                attn_mask,
                one_hot_objects_targets,
                hb0_poses_targets,
                hb0_exprs_targets,
                hb1_poses_targets,
                hb1_exprs_targets,
            ) in train_loader:
                # remove past gradients
                optimizer.zero_grad()
                # forward
                ids_text, attn_mask, one_hot_objects_targets, hb0_poses_targets, hb0_exprs_targets, hb1_poses_targets, hb1_exprs_targets = (
                    ids_text.to(device),
                    attn_mask.to(device),
                    one_hot_objects_targets.to(device),
                    hb0_poses_targets.to(device),
                    hb0_exprs_targets.to(device),
                    hb1_poses_targets.to(device),
                    hb1_exprs_targets.to(device),
                )
                object_probs, hb0_pose_probs, hb0_expr_probs, hb1_pose_probs, hb1_expr_probs = model(
                    ids_text, attn_mask
                )
                # Loss and backward
                object_loss = object_criterion(object_probs, one_hot_objects_targets)
                hb0_pose_loss = pose_expr_criterion(hb0_pose_probs, hb0_poses_targets)
                hb0_expr_loss = pose_expr_criterion(hb0_expr_probs, hb0_exprs_targets)
                hb1_pose_loss = pose_expr_criterion(hb1_pose_probs, hb1_poses_targets)
                hb1_expr_loss = pose_expr_criterion(hb1_expr_probs, hb1_exprs_targets)
                loss = (
                    object_loss
                    + hb0_pose_loss
                    + hb0_expr_loss
                    + hb1_pose_loss
                    + hb1_expr_loss
                )
                loss.backward()
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                # Update weights
                optimizer.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss.item()})

        # Set model in evaluation mode
        model.train(False)
        with torch.no_grad():
            for (
                ids_text,
                attn_mask,
                one_hot_objects_targets,
                hb0_poses_targets,
                hb0_exprs_targets,
                hb1_poses_targets,
                hb1_exprs_targets,
            ) in tqdm(val_loader):
                # forward
                ids_text, attn_mask, one_hot_objects_targets, hb0_poses_targets, hb0_exprs_targets, hb1_poses_targets, hb1_exprs_targets = (
                    ids_text.to(device),
                    attn_mask.to(device),
                    one_hot_objects_targets.to(device),
                    hb0_poses_targets.to(device),
                    hb0_exprs_targets.to(device),
                    hb1_poses_targets.to(device),
                    hb1_exprs_targets.to(device),
                )
                # Get predictions
                object_probs, hb0_pose_probs, hb0_expr_probs, hb1_pose_probs, hb1_expr_probs = model(
                    ids_text, attn_mask
                )
                one_hot_objects_preds = torch.zeros_like(object_probs)
                # Regular objects
                one_hot_objects_preds[torch.where(object_probs > 0.5)] = 1
                # Mike and Jenny predictions
                hb0_hb1_poses_preds = torch.cat(
                    [
                        torch.argmax(hb0_pose_probs, axis=-1),
                        torch.argmax(hb1_pose_probs, axis=-1),
                    ],
                    dim=0,
                )
                hb0_hb1_exprs_preds = torch.cat(
                    [
                        torch.argmax(hb0_expr_probs, axis=-1),
                        torch.argmax(hb1_expr_probs, axis=-1),
                    ],
                    dim=0,
                )
                # Mike and Jenny targets
                hb0_hb1_poses_targets = torch.cat(
                    [hb0_poses_targets, hb1_poses_targets], dim=0
                )
                hb0_hb1_exprs_targets = torch.cat(
                    [hb0_exprs_targets, hb1_exprs_targets], dim=0
                )
                # Aggregate predictions/targets
                evaluator.update_counters(
                    one_hot_objects_targets.cpu().numpy(),
                    one_hot_objects_preds.cpu().numpy(),
                    hb0_hb1_poses_targets.cpu().numpy(),
                    hb0_hb1_poses_preds.cpu().numpy(),
                    hb0_hb1_exprs_targets.cpu().numpy(),
                    hb0_hb1_exprs_preds.cpu().numpy(),
                )

        precision, recall, _ = evaluator.per_object_pr()
        posses_acc, expr_acc = evaluator.posses_expressions_accuracy()
        total_score = precision + recall + posses_acc + expr_acc
        if total_score > best_score:
            best_score = total_score
            logging.info("====================================================")
            logging.info(f"Found best on epoch {epoch+1}. Saving model!")
            logging.info(f"Precison is {precision}, recall is {recall}")
            logging.info(
                f"Posess accuracy is {posses_acc}, and expressions accuracy is {expr_acc}"
            )
            logging.info("====================================================")
            torch.save(model.state_dict(), save_model_path)


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Trains a Cliparts prediction model.")
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
        "--weight_decay", type=float, default=0.01, help="The weight decay."
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="models/best.pt",
        help="Where to save the model.",
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The bert model name.",
    )
    parser.add_argument(
        "--log_filepath", type=str, default=None, help="The logging file."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train(
        args.train_dataset_path,
        args.val_dataset_path,
        args.visual2index_path,
        args.bert_name,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.epochs,
        args.clip_val,
        args.save_model_path,
        args.log_filepath,
    )


if __name__ == "__main__":
    main()
