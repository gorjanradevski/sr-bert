from torch import nn
import torch
from transformers import BertConfig, BertModel, BertOnlyMLMHead
import logging

from datasets import X_PAD, Y_PAD, F_PAD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Text2VisualDiscreteBert(nn.Module):
    def __init__(self, config: BertConfig, device):
        super(Text2VisualDiscreteBert, self).__init__()
        self.cliparts_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=0,
        )
        self.x_embeddings = nn.Embedding(
            num_embeddings=X_PAD + 1,
            embedding_dim=config.hidden_size,
            padding_idx=X_PAD,
        )
        self.y_embeddings = nn.Embedding(
            num_embeddings=Y_PAD + 1,
            embedding_dim=config.hidden_size,
            padding_idx=Y_PAD,
        )
        self.f_embeddings = nn.Embedding(
            num_embeddings=F_PAD + 1,
            embedding_dim=config.hidden_size,
            padding_idx=F_PAD,
        )
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Change config for the positions
        config.vocab_size = X_PAD + 1
        self.x_head = BertOnlyMLMHead(config)
        config.vocab_size = Y_PAD + 1
        self.y_head = BertOnlyMLMHead(config)
        config.vocab_size = F_PAD + 1
        self.f_head = BertOnlyMLMHead(config)
        # Change config for the depth
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.device = device

    def forward(
        self,
        input_ids_sen,
        input_ids_vis,
        text_positions,
        x_ind,
        y_ind,
        f_ind,
        token_type_ids,
        attention_mask=None,
    ):
        # Word and clipart embeddings
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids_sen)
        vis_embeddings = self.cliparts_embeddings(input_ids_vis)
        input_embeddings = torch.cat([word_embeddings, vis_embeddings], dim=1).to(
            self.device
        )
        # Text positions
        text_embed = self.bert.embeddings.position_embeddings(text_positions)
        # Visual positions
        vis_embed = (
            self.x_embeddings(x_ind)
            + self.y_embeddings(y_ind)
            + self.f_embeddings(f_ind)
        )
        position_embeddings = torch.cat([text_embed, vis_embed], dim=1).to(self.device)
        sequence_output = self.bert(
            inputs_embeds=input_embeddings,
            position_embeddings=position_embeddings,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )[0]
        x_scores = self.x_head(sequence_output)
        y_scores = self.y_head(sequence_output)
        f_scores = self.f_head(sequence_output)

        return (
            self.log_softmax(x_scores),
            self.log_softmax(y_scores),
            self.log_softmax(f_scores),
        )


class Text2VisualContinuousBert(nn.Module):
    def __init__(self, config: BertConfig, device):
        super(Text2VisualContinuousBert, self).__init__()
        self.cliparts_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=0,
        )
        self.x_embeddings = nn.Embedding(
            num_embeddings=X_PAD + 1,
            embedding_dim=config.hidden_size,
            padding_idx=X_PAD,
        )
        self.y_embeddings = nn.Embedding(
            num_embeddings=Y_PAD + 1,
            embedding_dim=config.hidden_size,
            padding_idx=Y_PAD,
        )
        self.f_embeddings = nn.Embedding(
            num_embeddings=F_PAD + 1,
            embedding_dim=config.hidden_size,
            padding_idx=F_PAD,
        )
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Change config for the positions
        config.vocab_size = 1
        self.x_head = BertOnlyMLMHead(config)
        self.y_head = BertOnlyMLMHead(config)
        config.vocab_size = F_PAD + 1
        self.f_head = BertOnlyMLMHead(config)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.device = device

    def forward(
        self,
        input_ids_sen,
        input_ids_vis,
        text_positions,
        x_ind,
        y_ind,
        f_ind,
        token_type_ids,
        attention_mask=None,
    ):
        # Word and clipart embeddings
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids_sen)
        vis_embeddings = self.cliparts_embeddings(input_ids_vis)
        input_embeddings = torch.cat([word_embeddings, vis_embeddings], dim=1).to(
            self.device
        )
        # Text positions
        text_embed = self.bert.embeddings.position_embeddings(text_positions)
        # Visual positions
        vis_embed = (
            self.x_embeddings(x_ind)
            + self.y_embeddings(y_ind)
            + self.f_embeddings(f_ind)
        )
        position_embeddings = torch.cat([text_embed, vis_embed], dim=1).to(self.device)
        sequence_output = self.bert(
            inputs_embeds=input_embeddings,
            position_embeddings=position_embeddings,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )[0]

        return (
            torch.sigmoid(self.x_head(sequence_output)) * (X_PAD - 2),
            torch.sigmoid(self.y_head(sequence_output)) * (Y_PAD - 2),
            self.log_softmax(self.f_head(sequence_output)),
        )
