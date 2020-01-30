from torch import nn
import torch
from torchvision.models import resnet152
from transformers import BertConfig, BertModel, BertOnlyMLMHead, BertForMaskedLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalBert(nn.Module):
    def __init__(self, config: BertConfig, device, embeddings_path: str = None):
        super(MultiModalBert, self).__init__()
        self.embeddings = (
            nn.Embedding.from_pretrained(
                torch.load(embeddings_path, map_location=device),
                freeze=False,
                padding_idx=0,
            )
            if embeddings_path
            else nn.Embedding(config.vocab_size, config.hidden_size)
        )
        self.visual_position_projector = nn.Linear(7, 768)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.mlm_head = BertOnlyMLMHead(config)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.device = device

    def forward(
        self,
        input_ids,
        text_positions,
        visual_positions,
        token_type_ids,
        attention_mask=None,
    ):
        input_embeddings = self.embeddings(input_ids)
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


class VisualBert(nn.Module):
    def __init__(self, config: BertConfig):
        super(VisualBert, self).__init__()
        self.visual_position_projector = nn.Linear(7, 768)
        self.bert = BertForMaskedLM(config)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, vis_input_ids, visual_positions, attention_mask=None):
        vis_pos_embeddings = self.visual_position_projector(visual_positions)
        outputs = self.bert(
            input_ids=vis_input_ids,
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
