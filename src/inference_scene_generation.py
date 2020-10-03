import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from transformers import BertConfig
from scene_layouts.generation_strategies import generation_strategy_factory

from scene_layouts.datasets import (
    DiscreteInferenceDataset,
    ContinuousInferenceDataset,
    collate_pad_batch,
    BUCKET_SIZE,
)
from scene_layouts.evaluator import Evaluator
from scene_layouts.modeling import (
    SpatialDiscreteBert,
    SpatialContinuousBert,
    ClipartsPredictionModel,
)


def inference(
    sr_bert_checkpoint_path: str,
    clip_pred_checkpoint_path: str,
    model_type: str,
    test_dataset_path: str,
    visuals_dicts_path: str,
    gen_strategy: str,
    bert_name: str,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device {device}! ---")
    # Create dataset
    visual2index = json.load(
        open(os.path.join(visuals_dicts_path, "visual2index.json"))
    )
    test_dataset = (
        DiscreteInferenceDataset(test_dataset_path, visual2index)
        if model_type == "discrete"
        else ContinuousInferenceDataset(test_dataset_path, visual2index),
    )
    print(f"Testing on {len(test_dataset)}")
    # Create loader
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=4, collate_fn=collate_pad_batch
    )
    # Prepare SR-BERT model
    assert model_type in ["discrete", "continuous"]
    config = BertConfig.from_pretrained(bert_name)
    config.vocab_size = len(visual2index) + 1
    sr_bert_model = nn.DataParallel(
        SpatialDiscreteBert(config, bert_name)
        if model_type == "discrete"
        else SpatialContinuousBert(config, bert_name)
    ).to(device)
    sr_bert_model.load_state_dict(
        torch.load(sr_bert_checkpoint_path, map_location=device)
    )
    sr_bert_model.train(False)
    # Prepare Cliparts prediction model
    config = BertConfig.from_pretrained(bert_name)
    config.vocab_size = len(visual2index)
    clip_pred_model = nn.DataParallel(ClipartsPredictionModel(config, bert_name)).to(
        device
    )
    clip_pred_model.load_state_dict(
        torch.load(clip_pred_checkpoint_path, map_location=device)
    )
    clip_pred_model.train(False)
    print(f"Starting inference from checkpoint {sr_bert_checkpoint_path}!")
    print(f"Using {gen_strategy}!")
    evaluator = Evaluator(len(test_dataset))
    with torch.no_grad():
        for (ids_text, gt_vis, pos_text, _, _, _, x_lab, y_lab, o_lab, _, _) in tqdm(
            test_loader
        ):
            # forward
            ids_text, gt_vis, pos_text, x_lab, y_lab, o_lab = (
                ids_text.to(device),
                gt_vis.to(device),
                pos_text.to(device),
                x_lab.to(device),
                y_lab.to(device),
                o_lab.to(device),
            )
            # Get cliparts predictions
            ids_text_attn_mask = torch.ones_like(ids_text)
            probs = torch.sigmoid(clip_pred_model(ids_text, ids_text_attn_mask))
            clip_art_preds = torch.zeros_like(probs)
            # Regular objects
            clip_art_preds[:, :23][torch.where(probs[:, :23] > 0.35)] = 1
            clip_art_preds[:, 93:][torch.where(probs[:, 93:] > 0.35)] = 1
            # Mike and Jenny
            batch_indices = torch.arange(ids_text.size()[0])
            max_hb0 = torch.argmax(probs[:, 23:58], axis=-1)
            clip_art_preds[batch_indices, max_hb0 + 23] = 1
            max_hb1 = torch.argmax(probs[:, 58:93], axis=-1)
            clip_art_preds[batch_indices, max_hb1 + 58] = 1
            pred_vis = torch.tensor(
                [
                    i + 1
                    for i in range(clip_art_preds.size()[1])
                    if clip_art_preds[0, i] == 1
                ]
            ).unsqueeze(0)
            # Get spatial arrangements
            # Create an attention mask and token types tensor
            attn_mask = torch.ones(1, ids_text.size()[1] + pred_vis.size()[1])
            t_types = torch.cat(
                [torch.zeros_like(ids_text), torch.ones_like(pred_vis)], dim=1
            )
            x_out, y_out, o_out = generation_strategy_factory(
                gen_strategy,
                model_type,
                ids_text,
                pred_vis,
                pos_text,
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
                gt_vis[0].tolist(),
                x_out[0].tolist(),
                x_lab[0].tolist(),
                y_out[0].tolist(),
                y_lab[0].tolist(),
                o_out[0].tolist(),
                o_lab[0].tolist(),
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
    inference(
        args.sr_bert_checkpoint_path,
        args.clip_pred_checkpoint_path,
        args.model_type,
        args.test_dataset_path,
        args.visuals_dicts_path,
        args.gen_strategy,
        args.bert_name,
    )


if __name__ == "__main__":
    main()
