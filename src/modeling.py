from torch import nn
import torch
from torchvision.models import resnet152
from transformers import BertConfig, BertModel, BertForMaskedLM
import math
import logging

# https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MlmHead(nn.Module):
    def __init__(self, config: BertConfig, num_cliparts: int):
        super(MlmHead, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.layer_norm = torch.nn.LayerNorm(768, eps=config.layer_norm_eps)
        self.predictions = nn.Linear(768, num_cliparts, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_cliparts))

    def forward(self, sequence_output):
        hidden_states = self.dense(sequence_output)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        prediction_scores = self.predictions(hidden_states) + self.bias
        return prediction_scores

    @staticmethod
    def gelu(x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MultiModalBert(nn.Module):
    def __init__(
        self, embeddings_path: str, config: BertConfig, finetune: bool, device
    ):
        super(MultiModalBert, self).__init__()
        self.cliparts_embeddings = nn.Embedding.from_pretrained(
            torch.load(embeddings_path)
        )
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        logger.info("Embeddings and BERT loaded")
        self.visual_position_projector = nn.Linear(7, 768)
        self.mlm_head = MlmHead(config, self.cliparts_embeddings.num_embeddings)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.finetune = finetune
        self.device = device

        # Disable BERT fine-tuning
        for param in self.bert.parameters():
            param.requires_grad = finetune
        # Disable cliparts_embeddings fine-tuning
        self.cliparts_embeddings.weight.requires_grad = finetune

    def forward(
        self,
        word_input_ids,
        vis_input_ids,
        text_positions,
        visual_positions,
        token_type_ids,
        attention_mask=None,
    ):
        word_embeddings = self.bert.embeddings.word_embeddings(word_input_ids)
        vis_embeddings = self.cliparts_embeddings(vis_input_ids)
        input_embeddings = torch.cat([word_embeddings, vis_embeddings], dim=1)
        text_pos_embeddings = self.bert.embeddings.position_embeddings(text_positions)
        vis_pos_embeddings = self.visual_position_projector(visual_positions)
        position_embeddings = torch.cat(
            [text_pos_embeddings, vis_pos_embeddings], dim=1
        ).to(self.device)

        sequence_output = self.bert(
            inputs_embeds=input_embeddings,
            position_embeddings=position_embeddings,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )[0]
        prediction_scores = self.mlm_head(sequence_output)

        return self.log_softmax(prediction_scores)

    def train(self, mode: bool):
        if self.finetune and mode:
            self.bert.train(mode)
            self.mlm_head.train(mode)
        elif mode:
            self.bert.train(mode)
            self.mlm_head.train(mode)
        else:
            self.bert.train(mode)
            self.mlm_head.train(mode)


class VisualBert(nn.Module):
    def __init__(self, embeddings_path: str, config: BertConfig, finetune):
        super(VisualBert, self).__init__()
        self.cliparts_embeddings = nn.Embedding.from_pretrained(
            torch.load(embeddings_path)
        )
        logger.info(f"Building BERT from {config}")
        self.bert = BertForMaskedLM(config)
        logger.info("Embeddings and BERT loaded")
        self.visual_position_projector = nn.Linear(7, 768)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.finetune = finetune
        # Disable cliparts_embeddings fine-tuning
        self.cliparts_embeddings.weight.requires_grad = finetune

    def forward(self, vis_input_ids, visual_positions, attention_mask=None):
        vis_embeddings = self.cliparts_embeddings(vis_input_ids)
        vis_pos_embeddings = self.visual_position_projector(visual_positions)

        outputs = self.bert(
            inputs_embeds=vis_embeddings,
            position_embeddings=vis_pos_embeddings,
            attention_mask=attention_mask,
        )[0]

        return self.log_softmax(outputs)


class ImageEmbeddingsGenerator(nn.Module):
    def __init__(self):
        super(ImageEmbeddingsGenerator, self).__init__()
        self.resnet = torch.nn.Sequential(
            *(list(resnet152(pretrained=True).children())[:-1])
        )
        self.pooler = nn.AdaptiveAvgPool1d(768)

    def forward(self, x):
        output = torch.flatten(self.resnet(x), start_dim=1).unsqueeze(1)

        return self.pooler(output)
