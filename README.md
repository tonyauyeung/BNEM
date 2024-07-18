## Installation


```bash

# create micromamba environment
conda env create -f environment.yaml
conda activate dem

# install requirements
pip install -r requirements.txt

```

## Run experiments
To run an experiment, e.g., GMM with ENDEM, you can run on the command line

```bash
#model=dem/dem_en/endem/endem_bootstrap
#export WANDB_ENTITY=Energy-basedDEM
python dem/train.py experiment=gmm_idem model=endem
```
you could modify `configs/logger/wandb.yaml` to specify the name of your project.

To evaluate the sampled results for NLL and ESS, you need to modify the energy yaml file to your sampled datapoints:

```bash
data_path_train: "/scratch/bq216/code/DEM/dem_results/logs/train/runs/2024-07-04_16-14-11/samples_100000.pt"
data_path_val: ${energy.data_path_train}
```

and also modify the model yaml file:

```bash
#turn on the below 3 config for eval mode
nll_with_cfm: true
# train cfm only on train data and not dem
debug_use_train_data: true
logz_with_cfm: true
```

At last, run:
```bash
python dem/eval.py experiment=gmm_idem
```