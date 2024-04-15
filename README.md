# Autoregressive activity prediction for low-data drug discovery

## 💻 Run the experiments

### Clone repo and download data
```bash
# Clone repo
git clone https://github.com/ml-jku/autoregressive_activity_prediction.git

# Move into dir
cd ./autoregressive_activity_prediction

# Download and unzip assets folder (~600 MB zipped, ~4 GB unzipped)
pip install gdown
gdown https://drive.google.com/uc?id=1ZW1zzNEjrFmhCb4L0z2J2RWBOB9d3pAe
unzip assets.zip

# Download and unzip preprocessed fsmol data (~400 MB zipped, ~5 GB unzipped)
# Move to location at which data should be stored
cd path_to_preprocessed_fsmol_data_dir
gdown https://drive.google.com/uc?id=1SEi8dkkdXudWzRFAYABBckk12tNWfGtX
unzip preprocessed_data
```
### Update paths in config
```hydra
# config location: .src/autoregr_inf_experiment/cfg.py

# Base settings
    seed: int = 1234
    
    # Data
    data_path: str = "path_to_preprocessed_fsmol_data_dir" #TODO set path
    nbr_support_set_candidates: int = 32
    inference_batch_size: int = 64
    
    # Experiment
    device='gpu'
    
    # Results
    results_path: str = "" #TODO set path
...
```


### Conda environment
```bash
# Create conda environment
conda env create -f requirements.yml -n your_env_name

# Activate conda env
conda activate
```
### Update experiment config
Add suitable output paths for the experiment here: ```.src/autoregr_inf_experiment/cfg.py```.

### Run experiment
```bash
# Navigate into directory
cd .src/autoregr_inf_experiment/

# Run autoregressive inference experiment
python experiment_manager.py

# Create results by running the evaluation script
python evaluation.py
```

For different experiment variants see the experiment_manager.py file.
