# BNEM: A Boltzmann Sampler Based on Bootstrapped Noised Energy Matching


## Installation


```bash

# create micromamba environment
conda env create -f environment.yaml
conda activate bnem

# install requirements
pip install -r requirements.txt

```

## Run experiments
To run an experiment and log it to WandB, e.g., GMM with NEM, you can run on the command line

```bash
export WANDB_ENTITY=<your_wandb_entity>
python dem/train.py experiment=gmm_nem model=nem
```
you could modify `configs/logger/wandb.yaml` to customize your wandb.

Here's the list of available models:

    - dem: iDEM
    - dem_en: iDEM with an energy parameterisation
    - nem: NEM
    - bnem: BNEM
