import copy
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric.graphgym.models.act
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.contrib.layer.generalconv import (
    GeneralConvLayer,
    GeneralEdgeConvLayer,
)
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg

import torch_geometric.nn.conv as pyg_conv

@dataclass
class LayerConfig:
    # batchnorm parameters.
    has_batchnorm: bool = False
    bn_eps: float = 1e-5
    bn_mom: float = 0.1

    # mem parameters.
    mem_inplace: bool = False

    # gnn parameters.
    dim_in: int = -1
    dim_out: int = -1
    edge_dim: int = -1
    dim_inner: int = None
    num_layers: int = 2
    has_bias: bool = True
    # regularizer parameters.
    has_l2norm: bool = True
    dropout: float = 0.0
    # activation parameters.
    has_act: bool = True
    final_act: bool = True
    act: str = 'relu'

    # other parameters.
    keep_edge: float = 0.5

    # Additional params
    aggregation: str = 'mean'
    att_heads: int = 1
    dropout_gat: float = 0.0
    concat_gat: bool = True
    negative_slope_gat: float = 0.2
    dim_spline: int = 1
    kernel_size_spline: int = 2
    skip_linear_generalconv: bool = False
    att_generalconv: bool = False
    attention_type: str = 'additive'
    gcnconv_improved: bool = False
    add_self_loops: bool = True
    normalize: bool = True
    root_weight: bool = True
    project_sage: bool = False
    fill_value: str = 'mean'
    eps: float = 0.0
    train_eps: bool = False
    degree_spline: int = 1
    is_open_spline: bool = True
    in_edge_channels: int = None

def new_layer_config(dim_in, dim_out, num_layers, has_act, has_bias, cfg):
    return LayerConfig(
        has_batchnorm=cfg.gnn.batchnorm,
        bn_eps=cfg.bn.eps,
        bn_mom=cfg.bn.mom,
        mem_inplace=cfg.mem.inplace,
        dim_in=dim_in,
        dim_out=dim_out,
        edge_dim=cfg.dataset.edge_dim,
        has_l2norm=cfg.gnn.l2norm,
        dropout=cfg.gnn.dropout,
        has_act=has_act,
        final_act=True,
        act=cfg.gnn.act,
        has_bias=has_bias,
        keep_edge=cfg.gnn.keep_edge,
        dim_inner=cfg.gnn.dim_inner,
        num_layers=num_layers,
        aggregation=cfg.gnn.agg,
        att_heads=cfg.gnn.att_heads,
        dropout_gat=cfg.gnn.dropout_gat,
        concat_gat=cfg.gnn.concat_gat,
        negative_slope_gat=cfg.gnn.negative_slope_gat,
        dim_spline=cfg.gnn.dim_spline,
        kernel_size_spline=cfg.gnn.kernel_size_spline,
        skip_linear_generalconv=cfg.gnn.skip_linear_generalconv,
        att_generalconv=cfg.gnn.att_generalconv,
        attention_type=cfg.gnn.attention_type,
        gcnconv_improved=cfg.gnn.improved,
        add_self_loops=cfg.gnn.add_self_loops,
        normalize=cfg.gnn.normalize,
        root_weight=cfg.gnn.root_weight,
        project_sage=cfg.gnn.project,
        fill_value=cfg.gnn.fill_value,
        eps=cfg.gnn.eps,
        train_eps=cfg.gnn.train_eps,
        degree_spline=cfg.gnn.degree_spline,
        is_open_spline=cfg.gnn.is_open_spline,
        in_edge_channels=cfg.gnn.in_edge_channels
        )


# General classes
class GeneralLayer(nn.Module):
    """
    General wrapper for layers

    Args:
        name (str): Name of the layer in registered :obj:`layer_dict`
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation after the layer
        has_bn (bool):  Whether has BatchNorm in the layer
        has_l2norm (bool): Wheter has L2 normalization after the layer
        **kwargs (optional): Additional args
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn
        self.layer = register.layer_dict[name](layer_config, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(layer_config.dim_out, eps=layer_config.bn_eps,
                               momentum=layer_config.bn_mom))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=layer_config.dropout,
                           inplace=layer_config.mem_inplace))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act]())
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class GeneralMultiLayer(nn.Module):
    """
    General wrapper for a stack of multiple layers

    Args:
        name (str): Name of the layer in registered :obj:`layer_dict`
        num_layers (int): Number of layers in the stack
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        dim_inner (int): The dimension for the inner layers
        final_act (bool): Whether has activation after the layer stack
        **kwargs (optional): Additional args
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        dim_inner = layer_config.dim_out \
            if layer_config.dim_inner is None \
            else layer_config.dim_inner
        for i in range(layer_config.num_layers):
            d_in = layer_config.dim_in if i == 0 else dim_inner
            d_out = layer_config.dim_out \
                if i == layer_config.num_layers - 1 else dim_inner
            has_act = layer_config.final_act \
                if i == layer_config.num_layers - 1 else True
            inter_layer_config = copy.deepcopy(layer_config)
            inter_layer_config.dim_in = d_in
            inter_layer_config.dim_out = d_out
            inter_layer_config.has_act = has_act
            layer = GeneralLayer(name, inter_layer_config, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


# ---------- Core basic layers. Input: batch; Output: batch ----------------- #


@register_layer('linear')
class Linear(nn.Module):
    """
    Basic Linear layer.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        bias (bool): Whether has bias term
        **kwargs (optional): Additional args
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = Linear_pyg(layer_config.dim_in, layer_config.dim_out,
                                bias=layer_config.has_bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class BatchNorm1dNode(nn.Module):
    """
    BatchNorm for node feature.

    Args:
        dim_in (int): Input dimension
    """
    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.bn = nn.BatchNorm1d(layer_config.dim_in, eps=layer_config.bn_eps,
                                 momentum=layer_config.bn_mom)

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(nn.Module):
    """
    BatchNorm for edge feature.

    Args:
        dim_in (int): Input dimension
    """
    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.bn = nn.BatchNorm1d(layer_config.dim_in, eps=layer_config.bn_eps,
                                 momentum=layer_config.bn_mom)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


@register_layer('mlp')
class MLP(nn.Module):
    """
    Basic MLP model.
    Here 1-layer MLP is equivalent to a Liner layer.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        bias (bool): Whether has bias term
        dim_inner (int): The dimension for the inner layers
        num_layers (int): Number of layers in the stack
        **kwargs (optional): Additional args
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        dim_inner = layer_config.dim_in \
            if layer_config.dim_inner is None \
            else layer_config.dim_inner
        layer_config.has_bias = True
        layers = []
        if layer_config.num_layers > 1:
            sub_layer_config = LayerConfig(
                num_layers=layer_config.num_layers - 1,
                dim_in=layer_config.dim_in, dim_out=dim_inner,
                dim_inner=dim_inner, final_act=True)
            layers.append(GeneralMultiLayer('linear', sub_layer_config))
            layer_config = replace(layer_config, dim_in=dim_inner)
            layers.append(Linear(layer_config))
        else:
            layers.append(Linear(layer_config))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


@register_layer('gcnconv_old')
class GCNConvOld(nn.Module):
    """
    Graph Convolutional Network (GCN) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

@register_layer('gcnconv')
class GCNConv(nn.Module):
    """
    Graph Convolutional Network (GCN) layer from torch_geometric.nn.conv
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg_conv.GCNConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias, improved=layer_config.gcnconv_improved,
                                    add_self_loops=layer_config.add_self_loops, normalize=layer_config.normalize)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

@register_layer('sageconv_old')
class SAGEConvOld(nn.Module):
    """
    GraphSAGE Conv layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SAGEConv(layer_config.dim_in, layer_config.dim_out,
                                     bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

@register_layer('sageconv')
class SAGEConv(nn.Module):
    """
    GraphSAGE Conv layer from torch_geometric.nn.conv
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg_conv.SAGEConv(layer_config.dim_in, layer_config.dim_out,
                                     bias=layer_config.has_bias, aggr=layer_config.aggregation,
                                     root_weight=layer_config.root_weight, project=layer_config.project_sage)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

@register_layer('gatconv_old')
class GATConvOld(nn.Module):
    """
    Graph Attention Network (GAT) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

@register_layer('gatconv')
class GATConv(nn.Module):
    """
    Graph Attention Network (GAT) layer from torch_geometric.nn.conv
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg_conv.GATConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias, heads=layer_config.att_heads, 
                                    negative_slope=layer_config.negative_slope_gat, 
                                    dropout=layer_config.dropout_gat, concat=layer_config.concat_gat,
                                    add_self_loops=layer_config.add_self_loops, edge_dim=layer_config.edge_dim,
                                    fill_value=layer_config.fill_value)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('ginconv_old')
class GINConvOld(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = nn.Sequential(
            Linear_pyg(layer_config.dim_in, layer_config.dim_out), nn.ReLU(),
            Linear_pyg(layer_config.dim_out, layer_config.dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

@register_layer('ginconv')
class GINConv(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer from torch_geometric.nn.conv
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = nn.Sequential(
            Linear_pyg(layer_config.dim_in, layer_config.dim_out), nn.ReLU(),
            Linear_pyg(layer_config.dim_out, layer_config.dim_out))
        self.model = pyg_conv.GINConv(gin_nn, eps=layer_config.eps, train_eps=layer_config.train_eps)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('splineconv_old')
class SplineConvOld(nn.Module):
    """
    SplineCNN layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SplineConv(layer_config.dim_in,
                                       layer_config.dim_out, dim=1,
                                       kernel_size=2,
                                       bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch

@register_layer('splineconv')
class SplineConv(nn.Module):
    """
    SplineCNN layer from torch_geometric.nn.conv
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg_conv.SplineConv(layer_config.dim_in,
                                       layer_config.dim_out, dim=layer_config.dim_spline,
                                       kernel_size=layer_config.kernel_size_spline,
                                       bias=layer_config.has_bias, aggr=layer_config.aggregation,
                                       root_weight=layer_config.root_weight, degree=layer_config.degree_spline,
                                       is_open_spline=layer_config.is_open_spline)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


@register_layer('generalconv_old')
class GeneralConvOld(nn.Module):
    """A general GNN layer"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralConvLayer(layer_config.dim_in,
                                      layer_config.dim_out,
                                      bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

@register_layer('generalconv')
class GeneralConv(nn.Module):
    """A general GNN layer from torch_geometric.nn.conv"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg_conv.GeneralConv(layer_config.dim_in,
                                      layer_config.dim_out,
                                      bias=layer_config.has_bias, aggr=layer_config.aggregation,
                                      in_edge_channels=layer_config.in_edge_channels,
                                      skip_linear=layer_config.skip_linear_generalconv, 
                                      heads=layer_config.att_heads, attention=layer_config.att_generalconv, 
                                      attention_type=layer_config.attention_type)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

@register_layer('generaledgeconv')
class GeneralEdgeConv(nn.Module):
    """A general GNN layer that supports edge features as well"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralEdgeConvLayer(layer_config.dim_in,
                                          layer_config.dim_out,
                                          layer_config.edge_dim,
                                          bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index,
                             edge_feature=batch.edge_attr)
        return batch

@register_layer('generalsampleedgeconv')
class GeneralSampleEdgeConv(nn.Module):
    """A general GNN layer that supports edge features and edge sampling"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralEdgeConvLayer(layer_config.dim_in,
                                          layer_config.dim_out,
                                          layer_config.edge_dim,
                                          bias=layer_config.has_bias)
        self.keep_edge = layer_config.keep_edge

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < self.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_attr[edge_mask, :]
        batch.x = self.model(batch.x, edge_index, edge_feature=edge_feature)
        return batch
