import logging
import os
import numpy as np
import warnings
import optuna
import ast

from sklearn.model_selection import KFold
import custom_graphgym  # noqa, register custom modules
import torch
import pytorch_lightning as pl

from custom_graphgym.loader.custom_dataset import custom_dataset
from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import create_logger
from torch_geometric.loader import DataLoader

from itertools import combinations
import os

from make_data.dataset import FEATURES_DICT
from make_data.feats_to_ind import nx_node_indices, node_indices, edge_indices

def get_feats(comb, feat_dict, node_edge):
    graph_node_features = []
    node_features = list(feat_dict["node"]["generic"])
    edge_features = list(feat_dict["edge"]["generic"])
    for c in comb:
        if node_edge == "node":
            if c == "graph":
                graph_node_features.extend(feat_dict["node"][c])
            else:
                node_features.extend(feat_dict["node"][c])

        elif node_edge == "edge":
            if c in list(feat_dict["edge"].keys()):
                edge_features.extend(feat_dict["edge"][c])
            
    return graph_node_features,node_features,edge_features

def get_ind_X(X, keys, node_edge):
    """"
    Returns X with selected indices stored in value from dictionaries and key/s
    
    Parameters
    ----------
        X: PyTorch Geometric data objects
            Extracts node attributes and edge attributes from indices and keys
        keys: List
            List of keys/features that should be extracted
        node_edge: String
            Indicating whether node or edge attributes
            
    Returns
    -------
        X with selected features ready for training
    """
    
    nx_node_feats = list(nx_node_indices.keys())
    node_feats = list(node_indices.keys())
    edge_feats = list(edge_indices.keys())
    
    updated_x = None
    
    for feat in keys:
        if node_edge == "node":
            if feat in nx_node_feats:
                indices = nx_node_indices[feat]
            elif feat in node_feats:
                indices = node_indices[feat]
        elif node_edge == "edge":
            indices = edge_indices[feat]
        
        indices = indices.strip("[]")
        if ":" in indices:
            ind1 = int(indices.split(":")[0])
            ind2 = int(indices.split(":")[1])
            update = X[:,ind1:ind2]
        else:
            ind = int(indices)
            update = X[:,ind]
            update = update.resize_(update.shape[0], 1)
            
        if updated_x is None:
            updated_x = update
        else:
            updated_x = torch.cat((updated_x, update), dim=1)

    updated_x = torch.nan_to_num(updated_x, nan=-1)
            
    return updated_x

def create_trial_feats(trial):
    # Model is already selected in cfg, need to select the hyperparameters and set the cfg file
    # Set cfg file in __main__
    # Set optim parameters
    node_combs = []
    feature_types = ["knowledge", "feature", "atomic", "topological", "graph"]
    for i in range(1, len(feature_types) + 1):
        node_combs.extend([list(x) for x in combinations(feature_types, i)])

    edge_combs = []
    feature_types = ["atomic", "topological"]
    for i in range(1, len(feature_types) + 1):
        edge_combs.extend([list(x) for x in combinations(feature_types, i)])

    chosen_node_features=trial.suggest_categorical("node features", choices=node_combs)
    print("CHOSEN NODE FEATURES", chosen_node_features)
    graph_node_features,node_features,_ = get_feats(chosen_node_features, feat_dict=FEATURES_DICT, node_edge="node")

    chosen_edge_features=trial.suggest_categorical("edge features", choices=edge_combs)
    print("CHOSEN EDGE FEATURES", chosen_edge_features)
    _,_,edge_features = get_feats(chosen_edge_features, feat_dict=FEATURES_DICT, node_edge="edge")

    return graph_node_features,node_features,edge_features

def objective(trial):
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    
    # Should work with create_model() if cfg already set
    graph_node_features,node_features,edge_features = create_trial_feats(trial)
    loaders = create_loader()
    batch = next(iter(loaders[0]))
    # Get shape for input dimensions
    # Get correct features for X
    node_feats = graph_node_features+node_features
    updated_x = get_ind_X(batch.x, node_feats, node_edge="node")
    x_features = updated_x.shape[1]
    print("OLD X FEATURES", batch.x.shape)
    print("X FEATURES", updated_x.shape)
    try:
        y_features = batch.y.shape[1]
    except Exception as e:
        y_features = 1

    cfg.optim.lr_scheduler = 'reduce' # ReduceLROnPlateau
    cfg.optim.optimizer = 'adam'
    cfg.optim.steps =  [30, 60, 90]

    cfg.share.dim_in = x_features
    cfg.share.dim_out = y_features
    cfg.share.num_splits = len(loaders)

    # SET GIN BEST HPARAMS
    cfg.optim.max_epoch=100

    cfg.optim.base_lr=0.007
    cfg.optim.lr_decay=0.003
    cfg.optim.scheduler="cos"
    cfg.optim.weight_decay=0.000004
    
    # Set architecture parameters
    cfg.gnn.act="lrelu_01"
    cfg.gnn.batch_norm=True
    cfg.gnn.layertype="ginconv"
    cfg.gnn.dim_inner=300
    cfg.gnn.dropout=0.3
    cfg.gnn.l2norm=True
    cfg.gnn.layers_mp=2
    cfg.gnn.layers_post_mp=2
    cfg.gnn.layers_pre_mp=1
    cfg.gnn.normalize_adj=False
    cfg.gnn.self_msg="add"
    cfg.gnn.stage_type="skipconcat"
    
    # Model parameters
    cfg.model.graph_pooling="add"
    cfg.gnn.train_eps=False
    
    # Monitor RMSD if regression model
    if cfg.dataset.task_type == 'classification':
        cfg.model.loss_fun = 'cross_entropy'
        monitor_metric = 'accuracy'
    else:
        cfg.model.loss_fun = 'mse'
        monitor_metric = 'rmse'

    cfg.metric_best = monitor_metric

    dump_cfg(cfg)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Open dataset to be used for cross-validation
    dataset = custom_dataset(format=cfg.dataset.format, name=cfg.dataset.name, root=cfg.dataset.dir)
    dataset.data.to(torch.device(DEVICE))
    dataset.data.x = get_ind_X(dataset.data.x, node_feats, node_edge="node")
    dataset.data.edge_attr = get_ind_X(dataset.data.edge_attr, edge_features, node_edge="edge")

    # Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(range(len(dataset)))):
        # Edit MODEL_DIR
        MODEL_DIR = cfg.out_dir
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number), str(fold_idx)))

        #cfg.seed = 0
        set_run_dir(os.path.join(MODEL_DIR, "trial_{}".format(trial.number), str(fold_idx)))
        logging.info(cfg.run_dir)
        logger_callback = LoggerCallback()
        #logger = CSVLogger(save_dir=MODEL_DIR, name="trial_{}".format(trial.number), version=str(fold_idx))

        callbacks = []
        callbacks.append(logger_callback)
        callbacks.append(checkpoint_callback)
        callbacks.append(optuna.integration.PyTorchLightningPruningCallback(trial, monitor=monitor_metric))

        logging.info(cfg)

        device = torch.device(DEVICE)

        model = GraphGymModule(dim_in=x_features, dim_out=y_features, cfg=cfg)
        model.to(device)

        print('Set printing')
        set_printing()

        cfg.params = params_count(model)

        trainer = pl.Trainer(enable_checkpointing=True,
            callbacks=callbacks,
            precision="16-mixed",
            default_root_dir=os.path.join(MODEL_DIR, "trial_{}".format(trial.number), str(fold_idx)),
            max_epochs=cfg.optim.max_epoch,
            accelerator=cfg.accelerator,
            devices='auto' if not (torch.cuda.is_available() or DEVICE=="cpu") else [0]) # cfg.devices has to be ID of GPU e.g. 0 or 1 or 2...

        train_data = torch.utils.data.dataset.Subset(dataset, train_idx)
        valid_data = torch.utils.data.dataset.Subset(dataset, valid_idx)

        train_loader = DataLoader(
            train_data,
            batch_size=cfg.train.batch_size,
            shuffle=True,
        )

        valid_loader = DataLoader(
            valid_data,
            batch_size=cfg.train.batch_size,
            shuffle=True,
        )
        
        trainer.fit(model, train_loader, valid_loader)
        # Get val score - save test for after CV
        # Open stats.json and grab latest val metric
        with open(os.path.join(MODEL_DIR,"trial_{}".format(trial.number), str(fold_idx), str(cfg.seed), "val/stats.json"), "r") as json_file:
            json_str = json_file.read()

        str_split_list = [i for i in json_str.split('\n') if i]
        score = ast.literal_eval(str_split_list[-1])[monitor_metric]

        logging.info(fold_idx)
        logging.info(score)

        scores.append(score)
    
    ave_score = np.mean(scores)
   
    cfg.seed = 0
    
    return ave_score

if __name__ == '__main__':
    # Load cmd line args
    print('Loading args')
    args = parse_args()
    # Load config file
    print('Loading config file')
    load_cfg(cfg, args)
    print('Setting output dir')
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    print('Setting threads')
    torch.set_num_threads(cfg.num_threads)

    print('Setting run dir')
    # Set configurations for each run
    print('Setting seed')
    cfg.seed = cfg.seed + 1
    print('Seeding everything')
    seed_everything(cfg.seed)
    print('Autoselecting device')
    auto_select_device()
        
    # Set machine learning pipeline
    print('Setting ML pipeline')
    if cfg.metric_best == 'accuracy':
        direction='maximize'
    else:
        direction='minimize'
    
    study = optuna.create_study(storage=<optuna_db>, study_name=<study_name>, sampler=optuna.samplers.RandomSampler(), direction=direction, load_if_exists=True)
    study.optimize(objective, catch=(RuntimeError, RuntimeWarning), n_trials=20)

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
