from transformers import BertModel
from torch import nn
import torch
from torchvision.models import resnet152
from transformers import BertConfig
import math


class MlmHead(nn.Module):
    def __init__(self, config: BertConfig, vocab_size: int):
        super(MlmHead, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.layer_norm = torch.nn.LayerNorm(768, eps=config.layer_norm_eps)
        self.predictions = nn.Linear(768, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, sequence_output):
        hidden_states = self.dense(sequence_output)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        prediction_scores = self.predictions(hidden_states) + self.bias
        return prediction_scores

    @staticmethod
    def gelu(x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SceneModel(nn.Module):
    def __init__(self, embeddings_path: str, config: BertConfig):
        super(SceneModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        embeddings = torch.load(embeddings_path)
        self.bert.embeddings.word_embeddings = self.bert.embeddings.word_embeddings.from_pretrained(
            embeddings["weight"]
        )
        self.visual_position_projector = nn.Linear(2, 768)
        self.text_position_projector = self.bert.embeddings.position_embeddings
        self.mlm_head = MlmHead(
            config, self.bert.embeddings.word_embeddings.num_embeddings
        )

    def forward(self, input_ids, text_positions, visual_positions, token_type_ids):
        text_pos_embeddings = self.text_position_projector(text_positions)
        vis_pos_embeddings = self.visual_position_projector(visual_positions)
        position_embeddings = torch.cat(
            [text_pos_embeddings, vis_pos_embeddings], dim=1
        )

        sequence_output = self.bert(
            input_ids=input_ids,
            position_embeddings=position_embeddings,
            token_type_ids=token_type_ids,
        )[0]
        prediction_scores = self.mlm_head(sequence_output)

        return prediction_scores


class ImageEmbeddingsGenerator(nn.Module):
    def __init__(self):
        super(ImageEmbeddingsGenerator, self).__init__()
        self.resnet = torch.nn.Sequential(
            *(list(resnet152(pretrained=True).children())[:-1])
        )
        self.resnet.eval()
        self.pooler = nn.AdaptiveAvgPool1d(768)

    def forward(self, x):
        output = torch.flatten(self.resnet(x), start_dim=1).unsqueeze(1)

        return self.pooler(output)
