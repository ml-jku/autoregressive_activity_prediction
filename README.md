# Autoregressive activity prediction for low-data drug discovery

## 💻 Run the experiments

### Get started:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Clone repo
git clone https://github.com/ml-jku/autoregressive_activity_prediction.git
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