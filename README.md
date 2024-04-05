# Autoregressive activity prediction for low-data drug discovery

## 💻 Run the experiments

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
