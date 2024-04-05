import pytorch_lightning as pl
import numpy as np
import torch
import hydra
from torch import nn
import os
import sys
import inspect
from omegaconf import OmegaConf
from functools import partial
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class ProtoNet(pl.LightningModule):
    def __init__(self, cfg):
        super(ProtoNet, self).__init__()

        # Config
        self.cfg = cfg

        # Loss functions
        self.lossFunction = torch.nn.BCELoss()
        
        # Hyperparameters
        self.save_hyperparameters(cfg)

        # Encoder
        self.encoder = EncoderBlock(cfg)
        
        # Layernormalizing-block
        self.layerNormBlock = LayerNormalizingBlock(cfg)

        # Output function
        self.sigmoid = torch.nn.Sigmoid()
        self.prediction_scaling = cfg.model.prediction_scaling

    def forward(
        self,
        query_molecules: torch.Tensor,
        support_molecules_active: torch.Tensor,
        support_molecules_inactive: torch.Tensor,
        support_set_actives_size: torch.Tensor,
        support_set_inactives_size: torch.Tensor,
    ) -> torch.Tensor:
        # Get embeddings from molecule encoder
        query_embedding = self.encoder(query_molecules)
        support_actives_embedding = self.encoder(support_molecules_active)
        support_inactives_embedding = self.encoder(support_molecules_inactive)

        # Retrieve updated representations from the context module
        # - Layernorm
        (
            query_embedding,
            support_actives_embedding,
            support_inactives_embedding,
        ) = self.layerNormBlock(
            query_embedding, support_actives_embedding, support_inactives_embedding
        )
        
        
        masked_active_embeddings = support_actives_embedding
        active_embeddings_sum = torch.sum(masked_active_embeddings, dim=1)
        
        nbr = support_set_actives_size.reshape(-1,1).expand(-1,
                active_embeddings_sum.shape[1])
        
        active_prototypes = 1/nbr * active_embeddings_sum # shape [bs, 512]
        active_prototypes = active_prototypes.reshape(active_prototypes.shape[0], 1,
                                                        active_prototypes.shape[1])
        
        # -inactive prototypes

        masked_inactive_embeddings = support_inactives_embedding
        inactive_embeddings_sum = torch.sum(masked_inactive_embeddings, dim=1)

        nbr = support_set_inactives_size.reshape(-1,1).expand(-1,
                inactive_embeddings_sum.shape[1])

        inactive_prototypes = 1/nbr * inactive_embeddings_sum # shape [bs, 512]
        inactive_prototypes = inactive_prototypes.reshape(inactive_prototypes.shape[0],
                                                          1,
                                                          inactive_prototypes.shape[1])
        
        #query_embedding
        similarities_a = torch.squeeze(
            query_embedding @ torch.transpose(active_prototypes, 1, 2)
            )
        similarities_i = torch.squeeze(
            query_embedding @ torch.transpose(inactive_prototypes, 1, 2)
        )

        predictions = similarities_a - similarities_i

        predictions = self.sigmoid(self.prediction_scaling * predictions)
        
        return predictions
    
# ------------------------------------------------------------------------------------
# Initialization

def init_lecun(m):
    nn.init.normal_(
        m.weight,
        mean=0.0,
        std=torch.sqrt(torch.tensor([1.0]) / m.in_features).numpy()[0],
    )
    nn.init.zeros_(m.bias)


def init_kaiming(m, nonlinearity):
    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
    nn.init.zeros_(m.bias)


@torch.no_grad()
def init_weights(m, activation_function="linear"):
    if activation_function == "relu":
        if type(m) == nn.Linear:
            init_kaiming(m, nonlinearity="relu")
    elif activation_function == "selu":
        if type(m) == nn.Linear:
            init_lecun(m)
    elif activation_function == "linear":
        if type(m) == nn.Linear:
            init_lecun(m)
#-------------------------------------------------------------------------------------
# Modules

activation_function_mapping = {
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
}

dropout_mapping = {"relu": nn.Dropout, "selu": nn.AlphaDropout}

class EncoderBlock(nn.Module):
    """
    Fully connected molecule encoder block.
    - Takes molecular descriptors, e.g., ECFPs and RDKit fps as inputs
    - returns a molecular representation
    """

    def __init__(self, cfg: OmegaConf):
        super(EncoderBlock, self).__init__()

        # Input layer
        self.dropout = dropout_mapping[cfg.model.encoder.activation](
            cfg.model.encoder.regularization.input_dropout
        )
        self.fc = nn.Linear(
            cfg.model.encoder.input_dim, cfg.model.encoder.number_hidden_neurons
        )
        self.act = activation_function_mapping[cfg.model.encoder.activation]

        # Hidden layer
        self.hidden_linear_layers = nn.ModuleList([])
        self.hidden_dropout_layers = nn.ModuleList([])
        self.hidden_activations = nn.ModuleList([])

        for _ in range(cfg.model.encoder.number_hidden_layers):
            self.hidden_dropout_layers.append(
                dropout_mapping[cfg.model.encoder.activation](
                    cfg.model.encoder.regularization.dropout
                )
            )
            self.hidden_linear_layers.append(
                nn.Linear(
                    cfg.model.encoder.number_hidden_neurons,
                    cfg.model.encoder.number_hidden_neurons,
                )
            )
            self.hidden_activations.append(
                activation_function_mapping[cfg.model.encoder.activation]
            )

        # Output layer
        self.dropout_o = dropout_mapping[cfg.model.encoder.activation](
            cfg.model.encoder.regularization.dropout
        )
        self.fc_o = nn.Linear(
            cfg.model.encoder.number_hidden_neurons,
            cfg.model.associationSpace_dim,
        )
        self.act_o = activation_function_mapping[cfg.model.encoder.activation]

        # Initialization
        encoder_initialization = partial(init_weights, cfg.model.encoder.activation)
        self.apply(encoder_initialization)

    def forward(self, molecule_representation: torch.Tensor) -> torch.Tensor:
        # Input layer
        x = self.dropout(molecule_representation)
        x = self.fc(x)
        x = self.act(x)

        # Hidden layer
        for hidden_dropout, hidden_layer, hidden_activation_function in zip(
            self.hidden_dropout_layers,
            self.hidden_linear_layers,
            self.hidden_activations,
        ):
            x = hidden_dropout(x)
            x = hidden_layer(x)
            x = hidden_activation_function(x)

        # Output layer
        x = self.dropout_o(x)
        x = self.fc_o(x)
        x = self.act_o(x)

        return x
    

class LayerNormalizingBlock(nn.Module):
    """
    Layernorm-block which scales/transforms the representations for query, ac-
    tive, and inactive support set molecules.
    """

    def __init__(self, cfg: OmegaConf):
        super(LayerNormalizingBlock, self).__init__()

        self.cfg = cfg

        if cfg.model.layerNormBlock.usage:
            self.layernorm_query = nn.LayerNorm(
                cfg.model.associationSpace_dim,
                elementwise_affine=cfg.model.layerNormBlock.affine,
            )
            self.layernorm_support_actives = nn.LayerNorm(
                cfg.model.associationSpace_dim,
                elementwise_affine=cfg.model.layerNormBlock.affine,
            )
            self.layernorm_support_inactives = nn.LayerNorm(
                cfg.model.associationSpace_dim,
                elementwise_affine=cfg.model.layerNormBlock.affine,
            )

    def forward(
        self,
        query_embedding: torch.Tensor,
        support_actives_embedding: torch.Tensor,
        support_inactives_embedding: torch.Tensor,
    ) -> tuple:
        """
        inputs:
        - query; torch.tensor;
          dim: [batch-size, 1, embedding-dim]
            * e.g.: [512, 1, 1024]
        - active support set molecules; torch.tensor;
          dim: [batch-size, active-padding-dim, embedding-dim]
          * e.g.: [512, 9, 1024]
        - inactive support set molecules; torch.tensor;
          dim: [batch-size, inactive-padding-dim, initial-embedding-dim]
          * e.g.: [512, 11, 1024]

        return:
        tuple which includes the updated representations for query, active, and inactive
        support set molecules:
        (query, active support set molecules, inactive support set molecules)
        """

        # Layer normalization
        # Since the layernorm operations are optional the module just updates represen-
        # tations if the the referring option is set in the config.
        if self.cfg.model.layerNormBlock.usage:
            query_embedding = self.layernorm_query(query_embedding)
            support_actives_embedding = self.layernorm_support_actives(
                support_actives_embedding
            )
            if support_inactives_embedding is not None:
                support_inactives_embedding = self.layernorm_support_inactives(
                    support_inactives_embedding
                )
        return query_embedding, support_actives_embedding, support_inactives_embedding
#---------------------------------------------------------------------------------------
# Performance metrics


def compute_auc_score(predictions, labels, target_ids):
    aucs = list()
    target_id_list = list()

    for target_idx in torch.unique(target_ids):
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows]

        if torch.unique(y).shape[0] == 2:
            auc = roc_auc_score(y, preds)
            aucs.append(auc)
            target_id_list.append(target_idx.item())
        else:
            aucs.append(np.nan)
            target_id_list.append(target_idx.item())
    return np.nanmean(aucs), aucs, target_id_list


def compute_dauprc_score(predictions, labels, target_ids):
    dauprcs = list()
    target_id_list = list()

    for target_idx in torch.unique(target_ids):
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows].int()

        if torch.unique(y).shape[0] == 2:
            number_actives = y[y == 1].shape[0]
            number_inactives = y[y == 0].shape[0]
            number_total = number_actives + number_inactives

            random_clf_auprc = number_actives / number_total
            auprc = average_precision_score(
                y.numpy().flatten(), preds.numpy().flatten()
            )

            dauprc = auprc - random_clf_auprc
            dauprcs.append(dauprc)
            target_id_list.append(target_idx.item())
        else:
            dauprcs.append(np.nan)
            target_id_list.append(target_idx.item())

    return np.nanmean(dauprcs), dauprcs, target_id_list


    """
    For a given metric, this class tracks the difference between its values on the
    training and validation.
    Precisely, the absolute difference between training and validation value is
    returned.
    """
    def __init__(self):
        self.train_value = None
        self.val_value = None
    
    def set_train_value(self, new_value:float):
        """
        With this function the training value can be set. Having set the training value,
        the training value cannot be set again before the absolute-delta value is
        returned. This is to ensure that compared values really belong to the same epoch
        since training and validation values can just set once per epoch.
        """
        #assert self.train_value is None, "train value already set"
        self.train_value = new_value
    
    def set_val_value(self, new_value:float):
        """
        With this function the val. value can be set. Having set the val. value,
        the val. value cannot be set again before the absolute-delta value is
        returned. This is to ensure that compared values really belong to the same epoch
        since training and validation values can just set once per epoch.
        """
        #assert self.val_value is None, "val value already set"
        self.val_value = new_value
    
    @property
    def absolute_delta(self):
        """
        This property returns the absolute difference between previously set training
        and validation values.
        Querying this property resets the set training and validation values.
        """
        
        if self.train_value is not None: # this has to be done for pl sanity check
            assert self.train_value is not None, "train value not set"
            assert self.val_value is not None, "val value not set"
            
            delta = self.train_value - self.val_value
            abs_delta = np.abs(delta)
            
            # Reset values
            self.train_value = None
            self.val_value = None
            
            return abs_delta
        else:
            return 1 # max delta is 1
        
#---------------------------------------------------------------------------------------
# optimizer

def define_opimizer(config, parameters):
    if config.model.training.optimizer == "AdamW":
        base_optimizer = torch.optim.AdamW
    elif config.model.training.optimizer == "SGD":
        base_optimizer = torch.optim.SGD
    else:
        base_optimizer = torch.optim.Adam

    optimizer = base_optimizer(
        parameters,
        lr=config.model.training.lr,
        weight_decay=config.model.training.weightDecay,
    )

    if config.model.training.lrScheduler.usage:
        lrs_1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=config.model.training.lr, total_iters=5
        )
        lrs_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.994)

        lrs = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[lrs_1, lrs_2], milestones=[40]
        )

        lr_dict = {"scheduler": lrs, "monitor": "loss_val"}

        return [optimizer], [lr_dict]
    else:
        return optimizer