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

def create_cfg(trial):
    # Model is already selected in cfg, need to select the hyperparameters and set the cfg file
    # Set cfg file in __main__
    # Set optim parameters
    cfg.optim.base_lr=trial.suggest_float("base_lr", 10**-6, 10**-2, log=True)
    cfg.optim.lr_decay=trial.suggest_float("lr_decay", 10**-6, 10**-2, log=True)
    cfg.optim.scheduler=trial.suggest_categorical("scheduler", ["step", "cos"])
    cfg.optim.weight_decay=trial.suggest_float("weight_decay", 10**-6, 10**-2, log=True)
    
    # Set architecture parameters
    cfg.gnn.act=trial.suggest_categorical("act", ["selu","prelu","lrelu_01"])
    cfg.gnn.batch_norm=trial.suggest_categorical("batch_norm", [True, False])
    cfg.gnn.layer_type=trial.suggest_categorical("layer_type", ["linear","mlp","gcnconv","sageconv","gatconv","ginconv","splineconv","generalconv","generaledgeconv","generalsampleedgeconv"])
    cfg.gnn.dim_inner=trial.suggest_int("dim_inner", 16, 512)
    cfg.gnn.dropout=trial.suggest_float("dropout", 0.0, 0.5)
    cfg.gnn.l2norm=trial.suggest_categorical("l2norm", [True, False])
    cfg.gnn.layers_mp=trial.suggest_int("layers_mp", 1, 6)
    cfg.gnn.layers_post_mp=trial.suggest_int("layers_post_mp", 1, 3)
    cfg.gnn.layers_pre_mp=trial.suggest_int("layers_pre_mp", 1, 3)
    cfg.gnn.normalize_adj=trial.suggest_categorical("normalize_adj", [True, False])
    cfg.gnn.self_msg=trial.suggest_categorical("self_msg", ["none","add","concat"])
    cfg.gnn.stage_type=trial.suggest_categorical("stage_type", ["stack","skipsum","skipconcat"])
    
    # Model parameters
    cfg.model.graph_pooling=trial.suggest_categorical("graph_pooling", ["add","mean","max"])

    # Extra params for edited layers
    if cfg.gnn.layer_type == 'gcnconv':
        cfg.gnn.improved=trial.suggest_categorical("improved", [True, False])
        cfg.gnn.add_self_loops=trial.suggest_categorical("add_self_loops", [True, False])
        cfg.gnn.normalize=trial.suggest_categorical("normalize", [True, False])

    if cfg.gnn.layer_type == 'gatconv':
        cfg.gnn.att_heads=trial.suggest_int("att_heads", 1, 8)
        cfg.gnn.concat_gat=trial.suggest_categorical("concat_gat", [True, False])
        cfg.gnn.dropout_gat=trial.suggest_float("dropout_gat", 0.0, 0.5)
        cfg.gnn.negative_slope_gat=trial.suggest_float("negative_slope_gat", 0.01, 0.9)
        cfg.gnn.add_self_loops=trial.suggest_categorical("add_self_loops", [True, False])
        cfg.dataset.edge_dim=10 #should always be 10 if we use the same scheme for featurization

        if cfg.dataset.edge_dim != None:
            cfg.gnn.fill_value=trial.suggest_categorical("fill_value", ["add", "mean", "min", "max", "mul"])

    if cfg.gnn.layer_type == 'sageconv':
        cfg.gnn.agg=trial.suggest_categorical("agg", ["add","mean","max"])
        cfg.gnn.normalize=trial.suggest_categorical("normalize", [True, False])
        cfg.gnn.root_weight=trial.suggest_categorical("root_weight", [True, False])
        cfg.gnn.project=trial.suggest_categorical("project", [True, False])

    if cfg.gnn.layer_type == 'ginconv':
        cfg.gnn.train_eps=trial.suggest_categorical("eps", [True, False])

    if cfg.gnn.layer_type == 'splineconv':
        cfg.gnn.dim_spline=trial.suggest_int("dim_spline", 1, 8)
        cfg.gnn.kernel_size_spline=trial.suggest_int("kernel_size_spline", 1, 8)
        cfg.gnn.agg=trial.suggest_categorical("agg", ["add","mean","max"])
        cfg.gnn.root_weight=trial.suggest_categorical("root_weight", [True, False])
        cfg.gnn.degree_spline=trial.suggest_int("degree_spline", 1, 5)
        cfg.gnn.is_open_spline=trial.suggest_categorical("is_open_spline", [True, False])

    if cfg.gnn.layer_type == 'generalconv':
        cfg.gnn.agg=trial.suggest_categorical("agg", ["add","mean","max"])
        cfg.gnn.att_heads=trial.suggest_int("att_heads", 1, 8)
        cfg.gnn.skip_linear_generalconv=trial.suggest_categorical("skip_linear_generalconv", [True, False])
        cfg.gnn.att_generalconv=trial.suggest_categorical("att_generalconv", [True, False])
        if cfg.gnn.att_generalconv == True:
            cfg.gnn.attention_type=trial.suggest_categorical("attention_type", ["additive", "dotproduct"])
        cfg.dataset.in_edge_channels=10

    if cfg.gnn.layer_type == 'generaledgeconv':
        cfg.gnn.agg=trial.suggest_categorical("agg", ["add","mean","max"])
        cfg.dataset.edge_dim=10

    if cfg.gnn.layer_type == 'generalsampleedgeconv':
        cfg.gnn.agg=trial.suggest_categorical("agg", ["add","mean","max"])
        cfg.gnn.keep_edge=trial.suggest_float("keep_edge", 0.5, 1.0)
        cfg.dataset.edge_dim=10

    cfg.optim.max_epoch=100
    return cfg   

def objective(trial):
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    
    # Should work with create_model() if cfg already set
    cfg = create_cfg(trial)
    loaders = create_loader()
    batch = next(iter(loaders[0]))
    # Get shape for input dimensions
    x_features = batch.x.shape[1]
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
    study.optimize(objective, catch=(RuntimeError, RuntimeWarning), n_trials=30)

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
