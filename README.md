# GNN-QSAR-public
Public repository for GNN analysis for QSAR predictions. Authored by Natasha Sanjrani (natasha.x.sanjrani@gsk.com) and Damien Coupry (damien.x.coupry@gsk.com)

The required packages for a conda environment can be found in the `environment.yml` file. To install the make_data package into an environment, the following command needs to be run

```
pip install .
```

Please ensure that all files from the `graphgym_edited_scripts` folder have been moved to their respective locations in the local PyTorch Geometric installation under the graphgym package.

# Usage
## 1. Creating datasets
Once tab-separated files of cleaned SMILES and their corresponding response values have been obtained (ensuring the SMILES column contains smiles in the name and the value column contains _mean in the name), PyTorch geometric datasets can be constructed. To create the baseline dataset for testing GNN layers, use the following example with `compute_all_features` set to `False`, otherwise set to `True` if testing all computed features.

```
from make_data.dataset import CustomProcessedInMemoryDataset, create_preprocessed_data
import os

input_tsv_file = <location_of_tsv_data_file>
dir_to_save_dataset = <data_dir>
os.makedirs(os.path.join(dir_to_save_dataset, 'processed'))
create_preprocessed_data(input_tsv_file=input_tsv_file, processed_dir=os.path.join(dir_to_save_dataset, 'processed'), log_dir=<log_dir>, compute_all_features=<True/False>, cx_pK=<True/False>, time_features=<True/False>)
```

A bash script to run the full featurization directly can be found at `analysis/featurization/run_dataset_create.sh`. 

Finally, to generate tautomers for analysis, a bash script can be found at `analysis/featurization/run_taut_gen.sh`

## 2. Running Optuna
To run Optuna comparing different layers, the `main_optuna_layers.py` file should be used, otherwise if comparing all featurization schemes, the `main_optuna_feats.py` file should be used. An example can be seen below where an example configuration file can be found under `graphgym_edited_scripts/example_graph.yaml`

```
python main_optuna.py --cfg <graphgym_yaml_file> --repeat 1 # For graph regression
```

## 3. Analysing runs
A Jupyter notebook for analysing layers and features can be found under `analysis/architecture/analyse_Optuna_out.ipynb`
