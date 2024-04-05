"""
This files includes an experiment manager module.
It takes the pretrained model, and the data module as inputs and performs the
autoregressive inference experiment.
"""

#---------------------------------------------------------------------------------------
# Dependencies
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
import pickle
from cfg import Config
from dataloader import (DataModule_for_autregressive_inference_on_FSMol_testset)
current_loc = __file__.rsplit("/",3)[0]
import sys
sys.path.append(current_loc)
from src.protonet.load_trained_pn_model import Protonet_inference_module
from src.mhnfs.load_trained_model import MHNfs_inference_module

#---------------------------------------------------------------------------------------
# Experiment Manager

class ExperimentManager_AutoregressiveInference():
    def __init__(self, cfg=Config()):
        #self.model = Protonet_inference_module(device=cfg.device)
        self.model = MHNfs_inference_module(device=cfg.device)
        self.cfg = cfg
        
        self.task_ids = list(range(157))
        self.experiment_rerun_seeds = [8915, 1318, 7221, 7540,  664, 6137, 6833, 8471,
                                       9449, 7322]
        
    def perform_experiment(self):
        auc_dict = dict()
        dauc_pr_dict = dict()
        
        # Loop over tasks
        for task_id in self.task_ids:
            print(f'Processing task {task_id} ...')
            data_module = DataModule_for_autregressive_inference_on_FSMol_testset(task_id,
                                                                                  self.cfg)
            
            # Loop over reruns
            aucs_rerun_dict = dict()
            dauc_prs_rerun_dict = dict()
            for seed in self.experiment_rerun_seeds:
                print(f'... rerun seed {seed} ...')
                data_module.sample_initial_support_set_and_candidate_pool(seed)
                
                # Iteratively add members to support set with predicting pseudo labels
                
                aucs = list()
                dauc_prs = list()
                for i in range(self.cfg.nbr_support_set_candidates):
                    support_actives_input = data_module.support_set_actives_inputs
                    support_inactives_input = data_module.support_set_inactives_inputs
                    
                    # Prediction evaluation set
                    evaluation_predictions = []
                    evaluation_labels = []
                    for batch in data_module.eval_dataloader:
                        query_inputs = torch.unsqueeze(batch['inputs'],1)
                        query_labels = batch['labels']
                        
                        predictions = self.model.predict(
                            query_inputs,
                            support_actives_input.expand(query_inputs.shape[0], -1, -1),
                            support_inactives_input.expand(query_inputs.shape[0], -1, -1)
                        )
                        if len(predictions.numpy().shape) == 0:
                            evaluation_predictions = (evaluation_predictions + 
                                                    list(predictions.numpy().flatten())
                                                    )
                            evaluation_labels = (evaluation_labels + 
                                                list(query_labels.numpy().flatten())
                                                )
                        else:
                            evaluation_predictions = (evaluation_predictions + 
                                                    list(predictions.numpy())
                                                    )
                            evaluation_labels = (evaluation_labels + 
                                                list(query_labels.numpy())
                                                )
                    
                    # Performance on evaluation set
                    auc = roc_auc_score(evaluation_labels, evaluation_predictions)
                    auc_pr = average_precision_score(evaluation_labels,
                                                     evaluation_predictions)
                    nbr_inactives, nbr_actives = np.unique(evaluation_labels,
                                                           return_counts=True)[1]
                    random_clf = nbr_actives / (nbr_actives + nbr_inactives)
                    dauc_pr = auc_pr - random_clf
                    
                    aucs.append(auc)
                    dauc_prs.append(dauc_pr)
                    
                    # Prediction support set candidates
                    if i == (self.cfg.nbr_support_set_candidates-1):
                        break
                    
                    candidate_predictions = []
                    for batch in data_module.support_candidate_pool_dataloader:
                        query_inputs = torch.unsqueeze(batch['inputs'], 1)
                        predictions = self.model.predict(
                            query_inputs,
                            support_actives_input.expand(query_inputs.shape[0], -1, -1),
                            support_inactives_input.expand(query_inputs.shape[0], -1, -1)
                        )
                        candidate_predictions = (candidate_predictions + 
                                                  list(predictions.numpy())
                                                  )
                            
                    # Update support set
                    raw_active_cand_id = np.argmax(candidate_predictions)
                    raw_inactive_cand_id = np.argmin(candidate_predictions)
                    
                    data_module.add_candidate_to_support_set_and_remove_from_pool(
                        raw_active_cand_id,
                        raw_inactive_cand_id
                    )
            
                aucs_rerun_dict[seed] = aucs
                dauc_prs_rerun_dict[seed] = dauc_prs
            
            auc_dict[task_id] = aucs_rerun_dict
            dauc_pr_dict[task_id] = dauc_prs_rerun_dict
        
        # Save results
        current_loc = __file__.rsplit("/",3)[0]
        with open(current_loc + self.cfg.results_path_protonet, 'wb') as f:
            pickle.dump([auc_dict, dauc_pr_dict], f)
    def __init__(self, cfg=Config()):
        self.model = Protonet_inference_module(device=cfg.device)
        self.cfg = cfg
        
        self.task_ids = list(range(157))
        self.experiment_rerun_seeds = [8915, 1318, 7221, 7540,  664, 6137, 6833, 8471,
                                       9449, 7322]
        
    def perform_experiment(self):
        auc_dict = dict()
        dauc_pr_dict = dict()
        
        # Loop over tasks
        for task_id in self.task_ids:
            print(f'Processing task {task_id} ...')
            data_module = DataModule_for_transductive_inference_on_FSMol_testset(task_id,
                                                                                  self.cfg)
            
            # Loop over reruns
            aucs_rerun_dict = dict()
            dauc_prs_rerun_dict = dict()
            for seed in self.experiment_rerun_seeds:
                print(f'... rerun seed {seed} ...')
                data_module.sample_initial_support_set_and_create_query_set(seed)
                
                # Iteratively add members to support set with predicting pseudo labels
                
                aucs = list()
                dauc_prs = list()
                query_pseudo_labeled__label_buffer = list()
                query_pseudo_labeled_prediction_buffer = list()
                for i in range(self.cfg.nbr_support_set_candidates):
                    support_actives_input = data_module.support_set_actives_inputs
                    support_inactives_input = data_module.support_set_inactives_inputs
                    
                    # Prediction query set
                    query_predictions = []
                    query_labels = []
                    for batch in data_module.query_dataloader:
                        query_inputs = torch.unsqueeze(batch['inputs'],1)
                        labels = batch['labels']
                        
                        predictions = self.model.predict(
                            query_inputs,
                            support_actives_input.expand(query_inputs.shape[0], -1, -1),
                            support_inactives_input.expand(query_inputs.shape[0], -1, -1)
                        )
                        query_predictions = (query_predictions + 
                                                  list(predictions.numpy().flatten())
                                                  )
                        query_labels = (query_labels + 
                                               list(labels.numpy())
                                               )
                    # Performance on evaluation set
                    evaluation_labels = (list(query_labels) +
                                         query_pseudo_labeled__label_buffer)
                    evaluation_predictions = (list(query_predictions) +
                                              query_pseudo_labeled_prediction_buffer)
                    auc = roc_auc_score(np.array(evaluation_labels),
                                        np.array(evaluation_predictions))
                    auc_pr = average_precision_score(np.array(evaluation_labels),
                                                     np.array(evaluation_predictions))
                    nbr_inactives, nbr_actives = np.unique(evaluation_labels,
                                                           return_counts=True)[1]
                    random_clf = nbr_actives / (nbr_actives + nbr_inactives)
                    dauc_pr = auc_pr - random_clf
                    
                    aucs.append(auc)
                    dauc_prs.append(dauc_pr)
                        
                    # Update support set
                    raw_active_cand_id = np.argmax(query_predictions)
                    active_pred = np.max(query_predictions)
                    active_pred_label = query_labels[raw_active_cand_id]
                    raw_inactive_cand_id = np.argmin(query_predictions)
                    inactive_pred = np.min(query_predictions)
                    inactive_pred_label = query_labels[raw_inactive_cand_id]
                    
                    query_pseudo_labeled__label_buffer = (
                        query_pseudo_labeled__label_buffer + 
                        [active_pred_label, inactive_pred_label]
                    )
                    query_pseudo_labeled_prediction_buffer = (
                        query_pseudo_labeled_prediction_buffer +
                        [active_pred, inactive_pred]
                    )
                    
                    data_module.add_candidate_to_support_set_and_remove_from_query_set(
                        raw_active_cand_id,
                        raw_inactive_cand_id
                    )
                    
                aucs_rerun_dict[seed] = aucs
                dauc_prs_rerun_dict[seed] = dauc_prs
            
            auc_dict[task_id] = aucs_rerun_dict
            dauc_pr_dict[task_id] = dauc_prs_rerun_dict
        
        # Save results
        current_loc = __file__.rsplit("/",3)[0]
        with open(current_loc + self.cfg.results_path_protonet_transductive, 'wb') as f:
            pickle.dump([auc_dict, dauc_pr_dict], f)


#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    experiment_manager = ExperimentManager_AutoregressiveInference()
    experiment_manager.perform_experiment()
        
            
                