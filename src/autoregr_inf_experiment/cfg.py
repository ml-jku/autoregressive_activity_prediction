"""
This files includes the config for the autoregressive inference experiment,
e.g. path to the data ...
(Model hyperparameters are not included (see assets/mhnfs_data))
"""
from dataclasses import dataclass

@dataclass
class Config:
    
    #-----------------------------------------------------------------------------------
    # Base settings
    seed: int = 1234
    
    # Data
    data_path: str = "" #TODO set path
    nbr_support_set_candidates: int = 32
    inference_batch_size: int = 64
    
    # Experiment
    device='gpu'
    
    # Results
    results_path: str = "" #TODO set path
    
    #-----------------------------------------------------------------------------------
    # baseline
    results_path_baseline: str = "" #TODO set path
    
    #-----------------------------------------------------------------------------------
    # Transductive experiment
    results_path_transductive: str = "" #TODO set path
    
    #-----------------------------------------------------------------------------------
    # Protonet experiment
    results_path_protonet: str = "" #TODO set path
