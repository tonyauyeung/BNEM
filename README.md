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
    - dem: iDEM with an energy parameterisation
    - nem: NEM
    - bnem: BNEM
<!-- 
To evaluate the sampled results for NLL and ESS, you need to modify the energy yaml file to your sampled datapoints:

```bash
data_path_train: "<your_save_path_for_generated_samples>/samples_100000.pt"
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
``` -->