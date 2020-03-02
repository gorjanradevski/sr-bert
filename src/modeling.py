from torch import nn
import torch
from transformers import BertConfig, BertModel, BertOnlyMLMHead
import logging

from datasets import X_PAD, Y_PAD, F_PAD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Text2VisualDiscreteBert(nn.Module):
    def __init__(self, config: BertConfig, device, finetune: bool = False):
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
        self.pos_layer_norm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)
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
        self.finetune = finetune

        for param in self.bert.parameters():
            param.requires_grad = finetune

    def train(self, mode: bool):
        if self.finetune and mode:
            self.bert.train(True)
        if mode:
            self.cliparts_embeddings.train(True)
            self.x_embeddings.train(True)
            self.y_embeddings.train(True)
            self.f_embeddings.train(True)
            self.pos_dropout.train(True)
            self.pos_layer_norm.train(True)
            self.x_head.train(True)
            self.y_head.train(True)
            self.f_head.train(True)
        else:
            self.bert.train(False)
            self.cliparts_embeddings.train(False)
            self.x_embeddings.train(False)
            self.y_embeddings.train(False)
            self.f_embeddings.train(False)
            self.pos_dropout.train(False)
            self.pos_layer_norm.train(False)
            self.x_head.train(False)
            self.y_head.train(False)
            self.f_head.train(False)

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
        vis_embed = self.pos_layer_norm(vis_embed)
        vis_embed = self.pos_dropout(vis_embed)
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
    def __init__(self, config: BertConfig, device, finetune: bool = False):
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
        self.pos_layer_norm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Change config for the positions
        config.vocab_size = 2
        self.xy_head = BertOnlyMLMHead(config)
        config.vocab_size = F_PAD + 1
        self.f_head = BertOnlyMLMHead(config)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.device = device
        self.finetune = finetune

    def train(self, mode: bool):
        if self.finetune and mode:
            self.bert.train(True)
        if mode:
            self.cliparts_embeddings.train(True)
            self.x_embeddings.train(True)
            self.y_embeddings.train(True)
            self.f_embeddings.train(True)
            self.pos_dropout.train(True)
            self.pos_layer_norm.train(True)
            self.xy_head.train(True)
            self.f_head.train(True)
        else:
            self.bert.train(False)
            self.cliparts_embeddings.train(False)
            self.x_embeddings.train(False)
            self.y_embeddings.train(False)
            self.f_embeddings.train(False)
            self.pos_dropout.train(False)
            self.pos_layer_norm.train(False)
            self.xy_head.train(False)
            self.f_head.train(False)

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
        vis_embed = self.pos_layer_norm(vis_embed)
        vis_embed = self.pos_dropout(vis_embed)
        position_embeddings = torch.cat([text_embed, vis_embed], dim=1).to(self.device)
        sequence_output = self.bert(
            inputs_embeds=input_embeddings,
            position_embeddings=position_embeddings,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )[0]

        return (
            torch.sigmoid(self.xy_head(sequence_output))[:, :, 0] * (X_PAD - 2),
            torch.sigmoid(self.xy_head(sequence_output))[:, :, 1] * (Y_PAD - 2),
            self.log_softmax(self.f_head(sequence_output)),
        )
