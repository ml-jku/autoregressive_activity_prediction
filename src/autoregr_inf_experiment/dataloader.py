"""
This file includes the dataloader which provides the data for the autoregressive
inference experiment.
The tasks are the ones included in the FS-Mol test set.

For each task, the available  available datapoints, i.e. a molecular structure with
associated labels, are split into a) a (initial) support set, b) a support set
candidates set, and c) an evaluation set:

    a. Initial support set
    * consists of 1 active and 1 inactive molecule
    * randomly drawn from all available data points for this task
    * based on **real labels**

    b. Support set candidates set
    * consists of both 32 active and 32 inactive molecules
    * randomly drawn
    * molecules from [a.] not included
    * MHNfs predicts pseudo-labels for these molecules, and chooses, for every
      iteartion, one molecule with active pseudo-label and one molecule with inactive
      pseudo-label to add to the support set.

    c. Evaluation set
    * Includes all molecules which are not in [a.] and [b.]
    * For every iteration, MHNfs predicts the activity for all molecules in the
      evaluation set
    * Performance is reported in terms of AUC and ΔAUC-PR.

# Preprocessed FS-Mol data
The pre-processed FS-Mol test data is stored in triplets (molecule-id, task-id, label).
Also the molecular inputs for the MHNfs model, i.e. feature vectors, are available.
We load this data, split the data points as described above, and build a dataloader from
it.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader
#from src.autoregr_inf_experiment.cfg import Config
from cfg import Config

#---------------------------------------------------------------------------------------
# Data-Module

class DataModule_for_autregressive_inference_on_FSMol_testset:
    
    def __init__(self, task_id, cfg=Config()):
      
      #-------------------------------------------------------------------------------
      # Helper functions
      def _split_data(mol_ids, labels, seed, nbr_support_set_candidates):
        """
        This function takes the mol-ids and labels (from the tripets) and creates the
        support set candidates, and evaluation set.
        (The initial support set is created later).
        """
        
        # set seed
        np.random.seed(seed)
        
        # Get indices of active and inactive molecules
        mol_ids_active = mol_ids[labels == 1].flatten()
        mol_ids_inactive = mol_ids[labels == 0].flatten()
        
        # Randomly shuffle the indices
        np.random.shuffle(mol_ids_active)
        np.random.shuffle(mol_ids_inactive)
        
        # Split the indices into support set candidates and evaluation set
        mol_ids_active_ss_candidates = mol_ids_active[:nbr_support_set_candidates]
        mol_ids_inactive_ss_candidates = mol_ids_inactive[:nbr_support_set_candidates]
        
        mol_ids_active_eval = mol_ids_active[nbr_support_set_candidates:]
        mol_ids_inactive_eval = mol_ids_inactive[nbr_support_set_candidates:]
        
        labels_eval = np.array([1] * mol_ids_active_eval.shape[0] +
                                [0] * mol_ids_inactive_eval.shape[0])
        mol_ids_eval = np.array(list(mol_ids_active_eval) + 
                                list(mol_ids_inactive_eval))
        
        return (mol_ids_active_ss_candidates, mol_ids_inactive_ss_candidates,
                mol_ids_eval, labels_eval)
      
      class EvalDataset():
        def __init__(self, mol_ids, labels, mol_inputs):
          self.mol_ids = mol_ids
          self.labels = labels
          self.mol_inputs = mol_inputs
          self.len = mol_ids.shape[0]
        def __getitem__(self, index):
          sample_labels = self.labels[index]
          feature_mtx_ids = self.mol_ids[index]
          sample_inputs = self.mol_inputs[feature_mtx_ids]
          
          sample = {'inputs': sample_inputs, 'labels': sample_labels}
          return sample
        def __len__(self):
            return self.len
        
      
      #-------------------------------------------------------------------------------
      # Main part of the function
      self.cfg = cfg
      self.task_id = task_id
      
      # Initialize task specific data storage
      self.task_statistics = dict()
      
      # Load FS-Mol triplets and features
      mol_ids = np.load(cfg.data_path + "mol_ids.npy")
      task_ids = np.load(cfg.data_path + "task_ids.npy")
      labels = np.load(cfg.data_path + "labels.npy").astype('float32')
      self.mol_inputs = np.load(cfg.data_path + "mol_inputs.npy").astype('float32')
      
      # We filter the datapoints relevant for the requested task
      bool_filter = (task_ids == task_id)
      mol_ids = mol_ids[bool_filter]
      labels = labels[bool_filter]
      
      # Compute nbr of active and inactive molecules and store info in task_statistics
      nbr_inactives, nbr_actives = np.unique(labels, return_counts=True)[1]
      self.task_statistics['nbr_inactives_all'] = nbr_inactives
      self.task_statistics['nbr_actives_all'] = nbr_actives
      
      # Split data into evaluation set and support set pool
      (self.mol_ids_active_ss_candidates, self.mol_ids_inactive_ss_candidates,
        self.mol_ids_eval, self.labels_eval
        ) = _split_data(mol_ids,
                        labels,
                        cfg.seed,
                        cfg.nbr_support_set_candidates)
        
      # Create Evaluation Dataset
      self.eval_dataset = EvalDataset(self.mol_ids_eval,
                                      self.labels_eval,
                                      self.mol_inputs)
      self.eval_dataloader = DataLoader(self.eval_dataset,
                                          batch_size=cfg.inference_batch_size,
                                          shuffle=False,)
    
    def sample_initial_support_set_and_candidate_pool(self, seed):
      """
      This function samples one active and one inactive molecule from the support set
      canidate pool to create an initial support set.
      
      seed: "experiment rerun seed"
      """
      
      np.random.seed(seed)
      np.random.shuffle(self.mol_ids_active_ss_candidates)
      np.random.shuffle(self.mol_ids_inactive_ss_candidates)
      
      self.support_set_actives_ids = self.mol_ids_active_ss_candidates[:1]
      self.support_set_inactives_ids = self.mol_ids_inactive_ss_candidates[:1]
      
      # Create a support set candidate pool which does not include the drawn indices
      # We omit label information here
      self.support_candidate_pool = np.concatenate(
        [self.mol_ids_active_ss_candidates[1:],
         self.mol_ids_inactive_ss_candidates[1:]],
        0)
      
      # Create support set and candite pool inputs
      self.support_set_actives_inputs = torch.from_numpy(
        self.mol_inputs[self.support_set_actives_ids]).unsqueeze(0)
      self.support_set_inactives_inputs = torch.from_numpy(
        self.mol_inputs[self.support_set_inactives_ids]).unsqueeze(0)
      
      candidate_dataset = self.CandidateDataSet(self.support_candidate_pool,
                                               self.mol_inputs)
      self.support_candidate_pool_dataloader = DataLoader(
        candidate_dataset,
        batch_size=self.cfg.inference_batch_size,
        shuffle=False,
        )
    
    def add_candidate_to_support_set_and_remove_from_pool(self,
                                                          raw_active_candidate_ids,
                                                          raw_inactive_candidate_ids):
      """
      This function adds a member from the candidate pool to the support set and removes
      it from the pool.
      """
      assert type(raw_active_candidate_ids)==np.int64
      assert type(raw_inactive_candidate_ids)==np.int64
      
      raw_active_candidate_ids = list([raw_active_candidate_ids])
      raw_inactive_candidate_ids = list([raw_inactive_candidate_ids])
      
      active_cand_ids = self.support_candidate_pool[raw_active_candidate_ids]
      inactive_cand_ids = self.support_candidate_pool[raw_inactive_candidate_ids]
      
      # add members to support set
      self.support_set_actives_ids = np.concatenate([self.support_set_actives_ids,
                                                     active_cand_ids], 0)
      self.support_set_inactives_ids = np.concatenate([self.support_set_inactives_ids,
                                                       inactive_cand_ids], 0)
      
      # remove members from candidate pool      
      ids_to_remove = (list(raw_active_candidate_ids) +
                       list(raw_inactive_candidate_ids))
      self.support_candidate_pool = np.delete(self.support_candidate_pool,
                                              ids_to_remove)
      
      # update inputs
      self.support_set_actives_inputs = torch.from_numpy(
        self.mol_inputs[self.support_set_actives_ids]).unsqueeze(0)
      self.support_set_inactives_inputs = torch.from_numpy(
        self.mol_inputs[self.support_set_inactives_ids]).unsqueeze(0)
      
      candidate_dataset = self.CandidateDataSet(self.support_candidate_pool,
                                               self.mol_inputs)
      self.support_candidate_pool_dataloader = DataLoader(
        candidate_dataset,
        batch_size=self.cfg.inference_batch_size,
        shuffle=False,
        )
    
    class CandidateDataSet():
      def __init__(self, mol_ids, mol_inputs):
        self.mol_ids = mol_ids
        self.mol_inputs = mol_inputs
        self.len = mol_ids.shape[0]
      def __getitem__(self, index):
        feature_mtx_ids = self.mol_ids[index]
        sample_inputs = self.mol_inputs[feature_mtx_ids]
        sample = {'inputs': sample_inputs}
        return sample
      def __len__(self):
        return self.len

class DataModule_for_transductive_inference_on_FSMol_testset:
  """
  This module adapts the DataModule_for_autregressive_inference_on_FSMol_testset:
  Instead of providing different sets for evaluation and support set candidates, here
  The query set is both the evaluation set and the support set candidates set.
  """
    
  def __init__(self, task_id, cfg=Config()):

    # Main part of the function
    self.cfg = cfg
    self.task_id = task_id
    
    # Load FS-Mol triplets and features
    mol_ids = np.load(cfg.data_path + "mol_ids.npy")
    task_ids = np.load(cfg.data_path + "task_ids.npy")
    labels = np.load(cfg.data_path + "labels.npy").astype('float32')
    self.mol_inputs = np.load(cfg.data_path + "mol_inputs.npy").astype('float32')
    
    # We filter the datapoints relevant for the requested task
    bool_filter = (task_ids == task_id)
    mol_ids = mol_ids[bool_filter]
    labels = labels[bool_filter]
    
    # Get ids of active and inactive molecules
    self.mol_ids_active = mol_ids[labels == 1].flatten()
    self.mol_ids_inactive = mol_ids[labels == 0].flatten()
    
  
  def sample_initial_support_set_and_create_query_set(self, seed):
    """
    This function samples one active and one inactive molecule from the support set
    canidate pool to create an initial support set.
    
    seed: "experiment rerun seed"
    """
    
    np.random.seed(seed)
    np.random.shuffle(self.mol_ids_active)
    np.random.shuffle(self.mol_ids_inactive)
    
    self.support_set_actives_ids = self.mol_ids_active[:1]
    self.support_set_inactives_ids = self.mol_ids_inactive[:1]
    
    # Query set
    query_set_actives_ids = self.mol_ids_active[1:]
    query_set_inactives_ids = self.mol_ids_inactive[1:]
    self.query_set_ids = np.concatenate([query_set_actives_ids,
                                         query_set_inactives_ids], 0)
    self.query_set_labels = np.array([1] * query_set_actives_ids.shape[0] +
                                     [0] * query_set_inactives_ids.shape[0])
    
    # Create support set and candite pool inputs
    self.support_set_actives_inputs = torch.from_numpy(
      self.mol_inputs[self.support_set_actives_ids]).unsqueeze(0)
    self.support_set_inactives_inputs = torch.from_numpy(
      self.mol_inputs[self.support_set_inactives_ids]).unsqueeze(0)
    
    query_dataset = self.QueryDataset(self.query_set_ids,
                                      self.query_set_labels,
                                      self.mol_inputs)
    self.query_dataloader = DataLoader(query_dataset,
                                        batch_size=self.cfg.inference_batch_size,
                                        shuffle=False,)
  
  def add_candidate_to_support_set_and_remove_from_query_set(self,
                                                        raw_active_candidate_ids,
                                                        raw_inactive_candidate_ids):
    """
    This function adds a member from the candidate pool to the support set and removes
    it from the pool.
    """
    assert type(raw_active_candidate_ids)==np.int64
    assert type(raw_inactive_candidate_ids)==np.int64
    
    raw_active_candidate_ids = list([raw_active_candidate_ids])
    raw_inactive_candidate_ids = list([raw_inactive_candidate_ids])
    
    active_cand_ids = self.query_set_ids[raw_active_candidate_ids]
    inactive_cand_ids = self.query_set_ids[raw_inactive_candidate_ids]
    
    # add members to support set
    self.support_set_actives_ids = np.concatenate([self.support_set_actives_ids,
                                                    active_cand_ids], 0)
    self.support_set_inactives_ids = np.concatenate([self.support_set_inactives_ids,
                                                      inactive_cand_ids], 0)
    
    # remove members from candidate pool      
    ids_to_remove = (list(raw_active_candidate_ids) +
                      list(raw_inactive_candidate_ids))
    self.query_set_ids = np.delete(self.query_set_ids, ids_to_remove)
    self.query_set_labels = np.delete(self.query_set_labels, ids_to_remove)
    
    # update inputs
    self.support_set_actives_inputs = torch.from_numpy(
      self.mol_inputs[self.support_set_actives_ids]).unsqueeze(0)
    self.support_set_inactives_inputs = torch.from_numpy(
      self.mol_inputs[self.support_set_inactives_ids]).unsqueeze(0)
    
    query_dataset = self.QueryDataset(self.query_set_ids,
                                      self.query_set_labels,
                                      self.mol_inputs)
    self.query_dataloader = DataLoader(query_dataset,
                                        batch_size=self.cfg.inference_batch_size,
                                        shuffle=False,)
  
  class QueryDataset():
      def __init__(self, mol_ids, labels, mol_inputs):
        self.mol_ids = mol_ids
        self.labels = labels
        self.mol_inputs = mol_inputs
        self.len = mol_ids.shape[0]
      def __getitem__(self, index):
        sample_labels = self.labels[index]
        feature_mtx_ids = self.mol_ids[index]
        sample_inputs = self.mol_inputs[feature_mtx_ids]
        
        sample = {'inputs': sample_inputs, 'labels': sample_labels}
        return sample
      def __len__(self):
          return self.len

  #----------------------------------------------------------------------------------
       

#---------------------------------------------------------------------------------------
# For debugging

if __name__ == "__main__":
    #data_module = DataModule_for_autregressive_inference_on_FSMol_testset(task_id=1)   
    
    data_module = DataModule_for_transductive_inference_on_FSMol_testset(task_id=1)
    data_module.sample_initial_support_set_and_create_query_set(seed=1234)
    data_module.add_candidate_to_support_set_and_remove_from_query_set(np.int64(12),
                                                                       np.int64(14))
    data_module.add_candidate_to_support_set_and_remove_from_query_set(np.int64(7),
                                                                       np.int64(3))
    