from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, functional
from transformers import RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPooler,
    RobertaPreTrainedModel,
)


class MishActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))


class NFQAClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dims: Sequence[int] = (768, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.linear_layers = nn.Sequential(
            *(nn.Linear(input_dim, dim) for dim in hidden_dims)
        )
        self.classification_layer = torch.nn.Linear(hidden_dims[-1], num_labels)
        self.activations = [MishActivation()] * len(hidden_dims)
        self.dropouts = [torch.nn.Dropout(p=dropout)] * len(hidden_dims)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer, activation, dropout in zip(
            self.linear_layers, self.activations, self.dropouts
        ):
            output = dropout(activation(layer(output)))
        return self.classification_layer(output)


class RobertaNFQAClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _DROPOUT = 0.0

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.embedder = RobertaModel(config, add_pooling_layer=True)
        self.pooler = RobertaPooler(config)
        self.feedforward = NFQAClassificationHead(config.hidden_size, config.num_labels)
        self.dropout = torch.nn.Dropout(self._DROPOUT)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.embedder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        logits = self.feedforward(self.dropout(self.pooler(sequence_output)))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
