from make_data.dataset import CustomDataset, CustomProcessedInMemoryDataset, CustomDatasetOnlyLoad
from torch_geometric.graphgym.register import register_loader
import torch

@register_loader('custom_dataset')
def custom_dataset(format, name, root, data_list=[]):
    if format=='inmemory':
        dataset = CustomProcessedInMemoryDataset(root=root, data_list=data_list)
    elif format=='notinmemory':
        dataset = CustomDataset(root=root)
    elif format=='loadonly':
        dataset = CustomDatasetOnlyLoad(root=root)

    def set_dataset_attr(dataset, name, value, size):
        dataset._data_list = None
        dataset.data[name] = value

    splits = dataset.get_idx_split()
    split_names = [
        'train_graph_index', 'val_graph_index', 'test_graph_index'
    ]
    for i, key in enumerate(splits.keys()):
        id = splits[key]
        set_dataset_attr(dataset, split_names[i], id, len(id))

    return dataset
