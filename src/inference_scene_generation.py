import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from scene_layouts.datasets import (
    BUCKET_SIZE,
    ContinuousInferenceDataset,
    DiscreteInferenceDataset,
    collate_pad_batch,
)
from scene_layouts.evaluator import Evaluator
from scene_layouts.generation_strategies import generation_strategy_factory
from scene_layouts.modeling import (
    ClipartsPredictionModel,
    SpatialContinuousBert,
    SpatialDiscreteBert,
)


def inference(args):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device {device}! ---")
    # Create dataset
    visual2index = json.load(
        open(os.path.join(args.visuals_dicts_path, "visual2index.json"))
    )
    test_dataset = (
        DiscreteInferenceDataset(args.test_dataset_path, visual2index)
        if args.model_type == "discrete"
        else ContinuousInferenceDataset(args.test_dataset_path, visual2index)
    )
    print(f"Testing on {len(test_dataset)}")
    # Create loader
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=4, collate_fn=collate_pad_batch
    )
    # Prepare SR-BERT model
    assert args.model_type in ["discrete", "continuous"]
    config = BertConfig.from_pretrained(args.bert_name)
    config.vocab_size = len(visual2index) + 1
    sr_bert_model = nn.DataParallel(
        SpatialDiscreteBert(config, args.bert_name)
        if args.model_type == "discrete"
        else SpatialContinuousBert(config, args.bert_name)
    ).to(device)
    sr_bert_model.load_state_dict(
        torch.load(args.sr_bert_checkpoint_path, map_location=device)
    )
    sr_bert_model.train(False)
    # Prepare Cliparts prediction model
    config = BertConfig.from_pretrained(args.bert_name)
    config.vocab_size = len(visual2index)
    clip_pred_model = nn.DataParallel(
        ClipartsPredictionModel(config, args.bert_name)
    ).to(device)
    clip_pred_model.load_state_dict(
        torch.load(args.clip_pred_checkpoint_path, map_location=device)
    )
    clip_pred_model.train(False)
    print(f"Starting inference from checkpoint {args.sr_bert_checkpoint_path}!")
    print(f"Using {args.gen_strategy}!")
    evaluator = Evaluator(len(test_dataset))
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # forward
            batch = {key: val.to(device) for key, val in batch.items()}
            # Get cliparts predictions
            ids_text_attn_mask = torch.ones_like(batch["ids_text"]).to(device)
            probs = torch.sigmoid(
                clip_pred_model(batch["ids_text"], ids_text_attn_mask)
            )
            clip_art_preds = torch.zeros_like(probs).to(device)
            # Regular objects
            clip_art_preds[:, :23][torch.where(probs[:, :23] > 0.35)] = 1
            clip_art_preds[:, 93:][torch.where(probs[:, 93:] > 0.35)] = 1
            # Mike and Jenny
            batch_indices = torch.arange(batch["ids_text"].size()[0]).to(device)
            max_hb0 = torch.argmax(probs[:, 23:58], axis=-1)
            clip_art_preds[batch_indices, max_hb0 + 23] = 1
            max_hb1 = torch.argmax(probs[:, 58:93], axis=-1)
            clip_art_preds[batch_indices, max_hb1 + 58] = 1
            pred_vis = (
                torch.tensor(
                    [
                        i + 1
                        for i in range(clip_art_preds.size()[1])
                        if clip_art_preds[0, i] == 1
                    ]
                )
                .unsqueeze(0)
                .to(device)
            )
            # Get spatial arrangements
            # Create an attention mask and token types tensor
            attn_mask = torch.ones(
                1, batch["ids_text"].size()[1] + pred_vis.size()[1]
            ).to(device)
            t_types = torch.cat(
                [torch.zeros_like(batch["ids_text"]), torch.ones_like(pred_vis)], dim=1
            ).to(device)
            x_out, y_out, o_out = generation_strategy_factory(
                args.gen_strategy,
                args.model_type,
                batch["ids_text"],
                pred_vis,
                batch["pos_text"],
                t_types,
                attn_mask,
                sr_bert_model,
                device,
            )
            x_out, y_out = (
                x_out * BUCKET_SIZE + BUCKET_SIZE / 2,
                y_out * BUCKET_SIZE + BUCKET_SIZE / 2,
            )
            common_pred_x, common_pred_y, common_pred_o, common_gts_x, common_gts_y, common_gts_o = evaluator.find_common_cliparts(
                pred_vis[0].tolist(),
                batch["ids_vis"][0].tolist(),
                x_out[0].tolist(),
                batch["x_lab"][0].tolist(),
                y_out[0].tolist(),
                batch["y_lab"][0].tolist(),
                o_out[0].tolist(),
                batch["o_lab"][0].tolist(),
            )
            common_attn_mask = torch.ones_like(common_pred_x)
            evaluator.update_metrics(
                common_pred_x,
                common_gts_x,
                common_pred_y,
                common_gts_y,
                common_pred_o,
                common_gts_o,
                common_attn_mask,
            )

        print(
            f"The absolute similarity per scene is: {evaluator.get_abs_sim()} +/- {evaluator.get_abs_error_bar()}"
        )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(description="Performs scene generation inference.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="discrete",
        help="The type of the model used to perform inference",
    )
    parser.add_argument(
        "--clip_pred_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to a pretrained cliparts prediction model.",
    )
    parser.add_argument(
        "--sr_bert_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to a pretrained SR-BERT model.",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/test_dataset.json",
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--visuals_dicts_path",
        type=str,
        default="data/visuals_dicts/",
        help="Path to the directory with the visuals dictionaries.",
    )
    parser.add_argument(
        "--gen_strategy",
        type=str,
        default="highest_confidence",
        help="How to generate the positions during inference",
    )
    parser.add_argument(
        "--bert_name",
        type=str,
        default="bert-base-uncased",
        help="The bert model name.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    inference(args)


if __name__ == "__main__":
    main()
