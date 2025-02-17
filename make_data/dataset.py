import os.path as osp
import os
import numpy
import pandas as pd
from itertools import combinations
from rdkit import Chem
import pickle 
import jpype
import jpype.imports

import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset, InMemoryDataset

from make_data.featurize_data import featurize_single_compound
from make_data.all_featurize_data import computeAllFeatures

# All features grouped by type
FEATURES_DICT = {
    "node": {
        "knowledge": [
            "pKa",
            "pKa_bool",
            "pKb",
            "pKb_bool",
            "logps",
            "mrs",
            "asa_atom",
            "tpsa",
            "estates",
            "mass",
        ],
        "feature": ["isDonor", "isAcceptor", "isAcidic", "isBasic"],
        "atomic": [
            "ringSize",
            "isAromatic",
            "isInRing",
            "formalCharge",
            "element",
            "gasteiger",
            "mendeleev_atomic_radius",
            "mendeleev_atomic_volume",
            "mendeleev_density",
            "mendeleev_dipole_polarizability",
            "mendeleev_electron_affinity",
            "mendeleev_evaporation_heat",
            "mendeleev_fusion_heat",
            "mendeleev_lattice_constant",
            "mendeleev_specific_heat_capacity",
            "mendeleev_thermal_conductivity",
            "mendeleev_vdw_radius",
            "mendeleev_covalent_radius_cordero",
            "mendeleev_covalent_radius_pyykko",
            "mendeleev_en_pauling",
            "mendeleev_en_allen",
            "mendeleev_proton_affinity",
            "mendeleev_gas_basicity",
            "mendeleev_heat_of_formation",
            "mendeleev_covalent_radius_bragg",
            "mendeleev_vdw_radius_bondi",
            "mendeleev_vdw_radius_truhlar",
            "mendeleev_vdw_radius_rt",
            "mendeleev_vdw_radius_batsanov",
            "mendeleev_vdw_radius_dreiding",
            "mendeleev_vdw_radius_uff",
            "mendeleev_vdw_radius_mm3",
            "mendeleev_en_ghosh",
            "mendeleev_vdw_radius_alvarez",
            "mendeleev_atomic_weight",
            "mendeleev_atomic_weight_uncertainty",
            "mendeleev_is_monoisotopic",
            "mendeleev_atomic_radius_rahm",
            "mendeleev_metallic_radius",
            "mendeleev_dipole_polarizability_unc",
            "mendeleev_molar_heat_capacity",
            "mendeleev_mass",
            "mendeleev_mass_number",
            "mendeleev_mass_uncertainty",
            "mendeleev_spin",
            "mendeleev_g_factor",
            "mendeleev_quadrupole_moment",
        ],
        "generic": ["num_explicit_Hs", "num_implicit_Hs", "totalNumHs"],
        "topological": [
            "hybridization",
            "degrees",
            "chirality",
            "chiralityPossible",
            "USRCAT",
        ],
        "graph": [
            "degreeCentrality",
            "closenessCentrality",
            "betweennessCentrality",
            "katzCentrality",
            "fccCentrality",
            "fbcCentrality",
            "loadCentrality",
            "harmonicCentrality",
            "conHole",
            "degree",
            "ecc",
            "baryCenter",
            "center",
            "periphery",
        ],
    },
    "edge": {
        "generic": ["bondType"],
        "atomic": ["isInRing"],
        "topological": ["isConjugated", "isStereo"],
    },
}


def pytorch_geometric_data_obj_from_pytorch(data):
    """
    Creates PyTorch geometric graph objects from .pt files

    parameters
    ----------
        data: torch.load objects
            List of torch.load objects of labelled molecular graphs as: node_attrs, edge_attrs, edges, graph_labels, graph_indices_smiles

    returns
    -------
        data_list: list of torch_geometric.data.Data objects of labelled molecular graphs

    """

    node_attrs, edge_attrs, edges, graph_labels, graph_indices_smiles = data

    # Write to data list (with Data object) and graph list (without Data object)
    data_list = []
    for idx, smiles in graph_indices_smiles:
        X = node_attrs[idx]
        EF = edge_attrs[idx].to(torch.long)
        E = edges[idx]
        y_tensor = graph_labels[idx]
        smiles = graph_indices_smiles[idx][1]

        data_obj = Data(
            x=X,
            edge_index=E,
            edge_attr=EF,
            y=y_tensor,
            smiles=smiles,
            num_nodes=X.shape[0],
        )
        data_list.append(data_obj)

    return data_list

class molObj():
    def __init__(self):
        self.node_features = {}
        self.nx_node_features = {}
        self.edge_features = {}
        self.edges = None
        self.smiles = None
        self.val = None

def create_preprocessed_data(input_tsv_file, processed_dir, log_dir=None, compute_all_features=False, cx_pK=True, time_features=True):
    """
    Saves .pt files from input tsv file

    parameters
    ----------
        input_tsv_file: String
            Path to input tsv data containing SMILES and activity values
        processed_dir: String
            Path to save processed data
        log_dir: String
            Optional path to log directory to write output features if not using memory
        compute_all_features: Boolean
            True if computing all features for evaluation, otherwise computes generic features. Default False

    returns
    -------
        Saves .pt files as node_attrs.pt, edge_attrs.pt, edges.pt, graph_labels.pt, graph_indices_smiles.pt

    """

    dataframe = pd.read_csv(input_tsv_file, sep="\t")

    data_cols = dataframe.columns.tolist()
    smiles_col = None
    activity_col = None

    for col in data_cols:
        if "smiles" in col.lower():
            smiles_col = col
        if "_mean" in col.lower():
            activity_col = col

    if compute_all_features:
        print("Computing all features")
        # Initialize class
        if log_dir:
            print("Logdir exists, writing output files per molecule")
            comp_feats = computeAllFeatures(
                input_dataframe=dataframe,
                smiles_col=smiles_col,
                activity_col=activity_col,
                explicit_H=True,
                log_dir=log_dir
            )
            comp_feats.get_mol_descriptors(cx_pK=cx_pK, log_dir=log_dir, time_features=time_features)

            all_mol_objects = []
            
            all_indices_files = [i.split("_")[-1].split(".pickle")[0] for i in os.listdir(os.path.join(log_dir, "dataset")) if i.startswith("node_features")]

            for index in all_indices_files:
                molobj = molObj()

                node_feats_file = os.path.join(log_dir, "dataset", "node_features_"+index+".pickle")
                nx_node_feats_file = os.path.join(log_dir, "dataset", "nx_node_features_"+index+".pickle")
                edge_feats_file = os.path.join(log_dir, "dataset", "edge_features_"+index+".pickle")
                edges_file = os.path.join(log_dir, "dataset", "edges_"+index+".pickle")
                smiles_file = os.path.join(log_dir, "dataset", "mol_info_"+index+".pickle")
                val_file = os.path.join(log_dir, "dataset", "mol_label_"+index+".pickle")
 
                with open(node_feats_file, "rb") as openfile:
                    node_feats = pickle.load(openfile)
                    molobj.node_features = node_feats

                with open(nx_node_feats_file, "rb") as openfile:
                    node_feats = pickle.load(openfile)
                    molobj.nx_node_features = node_feats

                with open(edge_feats_file, "rb") as openfile:
                    edge_feats = pickle.load(openfile)
                    molobj.edge_features = edge_feats

                with open(edges_file, "rb") as openfile:
                    edges = pickle.load(openfile)
                    molobj.edges = edges

                with open(smiles_file, "rb") as openfile:
                    smi = pickle.load(openfile)
                    molobj.smiles = smi

                with open(val_file, "rb") as openfile:
                    val = pickle.load(openfile)
                    molobj.val = val

        
                all_mol_objects.append(molobj)

        else:
            comp_feats = computeAllFeatures(
                input_dataframe=dataframe,
                smiles_col=smiles_col,
                activity_col=activity_col,
                explicit_H=True
            )
            print("Getting all descriptors")
            # Get all features per mol object, including ChemAxon pKa by default
            comp_feats.get_mol_descriptors(cx_pK=cx_pK, time_features=time_features)

            all_mol_objects = comp_feats.molObjsWithFeats

        print("Getting base model")
        # Base model before groups
        preprocess_dataset = preprocessAllFeatures(
            all_mol_objects=all_mol_objects,
            nx_node_features_wanted=[],
            node_features_wanted=[],
            edge_features_wanted=[],
            output_root=osp.join(processed_dir, "_base"),
            base_model_only=True,
        )
        print("Save base model")
        # Save base preprocessed data
        preprocess_dataset.create_preprocessed_data()

        print("Loop through feature groups")
        # Save groupings of features
        # Groups/lists of features for easy assessment
        feature_types = ["knowledge", "feature", "atomic", "topological", "graph"]
        for i in range(1, len(feature_types) + 1):
            combs = list(combinations(feature_types, i))

            for comb in combs:
                print(comb)
                processed_dir_name = osp.join(processed_dir, "_".join(list(comb)))
                # Get all features for comb
                graph_node_features = []
                node_features = FEATURES_DICT["node"]["generic"]
                edge_features = FEATURES_DICT["edge"]["generic"]
                for c in comb:
                    if c == "graph":
                        graph_node_features.extend(FEATURES_DICT["node"][c])
                    else:
                        node_features.extend(FEATURES_DICT["node"][c])

                    if c in list(FEATURES_DICT["edge"].keys()):
                        edge_features.extend(FEATURES_DICT["edge"][c])

                print("Get model with selected features")
                preprocess_dataset = preprocessAllFeatures(
                    all_mol_objects=all_mol_objects,
                    nx_node_features_wanted=graph_node_features,
                    node_features_wanted=node_features,
                    edge_features_wanted=edge_features,
                    output_root=processed_dir_name,
                    base_model_only=False,
                )
                print("Save preprocessed data")
                # Save preprocessed data
                preprocess_dataset.create_preprocessed_data()

    else:
        node_attrs = []
        edge_attrs = []
        edges = []
        graph_labels = []
        graph_indices_smiles = []

        indices = 0

        # Zip input X and output y together and loop per SMILES and output
        for idx, row in dataframe.iterrows():
            smiles = row[smiles_col]
            y_val = row[activity_col]

            try:
                # Construct Pytorch Geometric Data object
                (
                    node_attrs_single,
                    edge_attrs_single,
                    edges_single,
                    graph_labels_single,
                    smiles_single,
                ) = featurize_single_compound(
                    smiles=smiles, y_val=y_val, create_data_object=False
                )
                node_attrs.append(node_attrs_single)
                edge_attrs.append(edge_attrs_single)
                edges.append(edges_single)
                graph_labels.append(graph_labels_single)
                graph_indices_smiles.append([indices, smiles_single])

                indices += 1

            except Exception as e:
                print("Process could not run for compound:", smiles_single)
                print("Error:", e)

        torch.save(node_attrs, osp.join(processed_dir, "node_attrs.pt"))
        torch.save(edge_attrs, osp.join(processed_dir, "edge_attrs.pt"))
        torch.save(edges, osp.join(processed_dir, "edges.pt"))
        torch.save(graph_labels, osp.join(processed_dir, "graph_labels.pt"))
        torch.save(
            graph_indices_smiles, osp.join(processed_dir, "graph_indices_smiles.pt")
        )


class preprocessAllFeatures:
    def __init__(
        self,
        all_mol_objects,
        nx_node_features_wanted,
        node_features_wanted,
        edge_features_wanted,
        output_root,
        base_model_only=False,
    ):
        """
        Saves .pt files from input custom mol objects containing all calculated features from the computeAllFeatures class

        parameters
        ----------
            all_mol_objects: List
                Custom mol objects from computeAllFeatures class
            nx_node_features_wanted: List
                List of networkx features wanted
            node_features_wanted: List
                List of all other node features wanted
            edge_features_wanted: List
                List of edge features wanted
            output_root: String
                Path to output directory for pre-processed data
            base_model_only: Boolean
                Optional argument, if True, does not use any features

        """

        self.molObjsWithFeats = all_mol_objects
        self.nx_node_feats = nx_node_features_wanted
        self.node_feats = node_features_wanted
        self.edge_feats = edge_features_wanted
        self.root = output_root
        self.base_model_only = base_model_only

    def get_feats(self, key_list, feat_dict):
        """
        Gets all custom mol object features from user-defined input

        parameters
        ----------
            key_list: List
                List of feature names
            feat_dict: Dictionary
                Dictionary of features from mol objects

        returns
        -------
            Flattened array of mol features

        """

        final_feats = []

        for feat in key_list:
            mol_features = feat_dict[feat]

            if type(mol_features) == list:
                final_feats += mol_features
            else:
                final_feats += [mol_features]

        return final_feats

    def create_preprocessed_data(self):
        """
        Saves .pt files from molecule objects

        returns
        -------
            Saves .pt files as node_attrs.pt, edge_attrs.pt, edges.pt, graph_labels.pt, graph_indices_smiles.pt

        """

        # Saves .pt files from input mol objects
        processed_dir = os.path.join(self.root, "processed")

        # Create root_dir/processed if doesn't exist
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        # Loop through molecule objects
        node_attrs = []
        edge_attrs = []
        edges = []
        graph_labels = []
        graph_indices_smiles = []

        indices = 0

        # Zip input X and output y together and loop per SMILES and output
        for molobj in self.molObjsWithFeats:
            try:
                mol_rdkit_obj = Chem.AddHs(Chem.MolFromSmiles(molobj.smiles))
                n_nodes = mol_rdkit_obj.GetNumAtoms()
                n_edges = 2 * mol_rdkit_obj.GetNumBonds()

                if self.base_model_only:
                    n_node_features = 1
                    n_edge_features = 1
                else:
                    n_node_features = len(
                        self.get_feats(
                            key_list=self.node_feats, feat_dict=molobj.node_features[0]
                        )
                    )
                    n_node_features += len(
                        self.get_feats(
                            key_list=self.nx_node_feats,
                            feat_dict=molobj.nx_node_features[0],
                        )
                    )

                    n_edge_features = len(
                        self.get_feats(
                            key_list=self.edge_feats, feat_dict=molobj.edge_features[0]
                        )
                    )

                X = numpy.zeros((n_nodes, n_node_features))
                EF = numpy.zeros((n_edges, n_edge_features))
                E = molobj.edges

                for atom in mol_rdkit_obj.GetAtoms():
                    if self.base_model_only:
                        X[atom.GetIdx(), :] = [1]
                    else:
                        # Add atom features to X using atom indices (position of atom in the molecule)
                        # So for each node (atom) we add [1----n_node_features] array at the row position corresponding to the atom index
                        node_features = self.get_feats(
                            key_list=self.nx_node_feats,
                            feat_dict=molobj.nx_node_features[atom.GetIdx()],
                        )
                        node_features += self.get_feats(
                            key_list=self.node_feats,
                            feat_dict=molobj.node_features[atom.GetIdx()],
                        )

                        X[atom.GetIdx(), :] = node_features

                # Convert X (n_nodes, n_node_features) to torch tensor (to run operations on CPU/GPU later)
                X = torch.tensor(X, dtype=torch.float)
                node_attrs.append(X)

                for k in list(molobj.edge_features.keys()):
                    edge_features = self.get_feats(
                        key_list=self.edge_feats, feat_dict=molobj.edge_features[k]
                    )
                    if self.base_model_only:
                        EF[k] = [1]
                    else:
                        EF[k] = edge_features

                EF = torch.tensor(EF, dtype=torch.float)

                edge_attrs.append(EF)
                edges.append(E)
                graph_labels.append(molobj.val)
                graph_indices_smiles.append([indices, molobj.smiles])

                indices += 1

            except Exception as e:
                print("Process could not run for compound:", molobj.smiles)
                print("Error:", e)

        torch.save(node_attrs, osp.join(processed_dir, "node_attrs.pt"))
        torch.save(edge_attrs, osp.join(processed_dir, "edge_attrs.pt"))
        torch.save(edges, osp.join(processed_dir, "edges.pt"))
        torch.save(graph_labels, osp.join(processed_dir, "graph_labels.pt"))
        torch.save(
            graph_indices_smiles, osp.join(processed_dir, "graph_indices_smiles.pt")
        )

class preprocessFeaturesToDataObjs:
    def __init__(self, log_dir, root):
        self.data_list = self.preprocess_save_dataset(log_dir=log_dir, root=root)

    def preprocess_save_dataset(self, log_dir, root):
        all_indices_files = [i.split("_")[-1].split(".pickle")[0] for i in os.listdir(os.path.join(log_dir, "dataset")) if i.startswith("node_features")]

        processed_dir = os.path.join(root, "processed")

        all_data = []

        # Create root_dir/processed if doesn't exist
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        counter = 0
        for index in all_indices_files:
            if int(counter) % 100 == 0:
                print(counter)

            molobj = molObj()

            node_feats_file = os.path.join(log_dir, "dataset", "node_features_"+index+".pickle")
            nx_node_feats_file = os.path.join(log_dir, "dataset", "nx_node_features_"+index+".pickle")
            edge_feats_file = os.path.join(log_dir, "dataset", "edge_features_"+index+".pickle")
            edges_file = os.path.join(log_dir, "dataset", "edges_"+index+".pickle")
            smiles_file = os.path.join(log_dir, "dataset", "mol_info_"+index+".pickle")
            val_file = os.path.join(log_dir, "dataset", "mol_label_"+index+".pickle")

            with open(node_feats_file, "rb") as openfile:
                node_feats = pickle.load(openfile)
                molobj.node_features = node_feats

            with open(nx_node_feats_file, "rb") as openfile:
                node_feats = pickle.load(openfile)
                molobj.nx_node_features = node_feats

            with open(edge_feats_file, "rb") as openfile:
                edge_feats = pickle.load(openfile)
                molobj.edge_features = edge_feats

            with open(edges_file, "rb") as openfile:
                edges = pickle.load(openfile)
                molobj.edges = edges

            with open(smiles_file, "rb") as openfile:
                smi = pickle.load(openfile)
                molobj.smiles = smi

            with open(val_file, "rb") as openfile:
                val = pickle.load(openfile)
                molobj.val = val

            comb = ('knowledge', 'feature', 'atomic', 'topological', 'graph')
            graph_node_features = []
            node_features = FEATURES_DICT["node"]["generic"]
            edge_features = FEATURES_DICT["edge"]["generic"]
            for c in comb:
                if c == "graph":
                    graph_node_features.extend(FEATURES_DICT["node"][c])
                else:
                    node_features.extend(FEATURES_DICT["node"][c])

                if c in list(FEATURES_DICT["edge"].keys()):
                    edge_features.extend(FEATURES_DICT["edge"][c])

            features = self.get_molobj_features(molobj, sorted(list(set(node_features))), sorted(list(set(graph_node_features))), sorted(list(set(edge_features))), index, base_model_only=False)
            if features is not None:
                node_attrs,edge_attrs,edges,graph_labels,graph_indices_smiles = features
                data = Data(x=node_attrs, edge_index=edges, edge_attr=edge_attrs, y=graph_labels, smiles=graph_indices_smiles[1], mol_idx=graph_indices_smiles[0])
                all_data.append(data)
            counter += 1

        return all_data

    def get_molobj_features(self, molobj, node_feats, nx_node_feats, edge_feats, indices, base_model_only=False):
        # Zip input X and output y together and loop per SMILES and output
        try:
            mol_rdkit_obj = Chem.AddHs(Chem.MolFromSmiles(molobj.smiles))
            n_nodes = mol_rdkit_obj.GetNumAtoms()
            n_edges = 2 * mol_rdkit_obj.GetNumBonds()

            if base_model_only:
                n_node_features = 1
                n_edge_features = 1
            else:
                n_node_features = len(
                    self.get_feats(
                        key_list=node_feats, feat_dict=molobj.node_features[0]
                    )
                )
                n_node_features += len(
                    self.get_feats(
                        key_list=nx_node_feats,
                        feat_dict=molobj.nx_node_features[0],
                    )
                )

                n_edge_features = len(
                    self.get_feats(
                        key_list=edge_feats, feat_dict=molobj.edge_features[0]
                    )
                )

            X = numpy.zeros((n_nodes, n_node_features))
            EF = numpy.zeros((n_edges, n_edge_features))
            E = molobj.edges

            for atom in mol_rdkit_obj.GetAtoms():
                if base_model_only:
                    X[atom.GetIdx(), :] = [1]
                else:
                    # Add atom features to X using atom indices (position of atom in the molecule)
                    # So for each node (atom) we add [1----n_node_features] array at the row position corresponding to the atom index
                    node_features = self.get_feats(
                        key_list=nx_node_feats,
                        feat_dict=molobj.nx_node_features[atom.GetIdx()],
                    )
                    node_features += self.get_feats(
                        key_list=node_feats,
                        feat_dict=molobj.node_features[atom.GetIdx()],
                    )

                    X[atom.GetIdx(), :] = node_features

            # Convert X (n_nodes, n_node_features) to torch tensor (to run operations on CPU/GPU later)
            X = torch.tensor(X, dtype=torch.float)

            for k in list(molobj.edge_features.keys()):
                edge_features = self.get_feats(
                    key_list=edge_feats, feat_dict=molobj.edge_features[k]
                )
                if base_model_only:
                    EF[k] = [1]
                else:
                    EF[k] = edge_features

            EF = torch.tensor(EF, dtype=torch.float)

            node_attrs = X
            edge_attrs = EF
            edges = E
            graph_labels = torch.tensor(numpy.array([molobj.val]), dtype=torch.float)
            graph_indices_smiles = [indices, molobj.smiles]

            return node_attrs,edge_attrs,edges,graph_labels,graph_indices_smiles

        except Exception as e:
            print("Process could not run for compound:", molobj.smiles)
            print("Error:", e)
            return None

    def get_feats(self, key_list, feat_dict):
        """
        Gets all custom mol object features from user-defined input

        parameters
        ----------
            key_list: List
                List of feature names
            feat_dict: Dictionary
                Dictionary of features from mol objects

        returns
        -------
            Flattened array of mol features

        """

        final_feats = []

        for feat in key_list:
            mol_features = feat_dict[feat]

            if type(mol_features) == list:
                final_feats += mol_features
            else:
                final_feats += [mol_features]

        return final_feats


class CustomDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        Keep name data.tsv in root dir file
        """

        return "data.tsv"

    @property
    def processed_file_names(self):
        """
        Keep empty for now
        """

        return "not_implemented.pt"

    def process(self):
        """
        Featurize all SMILES to graphs
        """

        # Open data.tsv
        self.data = pd.read_csv(self.raw_paths[0], sep="\t")

        # TODO: ONLY IMPLEMENTED FOR INTERNAL DATA CURRENTLY WHERE SMILES col = 'parent_isomeric_smiles' and activity col = '*_mean'. Extend to external ChEMBL data to look for 'SMILES' in col and 'pChEMBL_value' in activity col
        data_cols = self.data.columns.tolist()
        smiles_col = None
        activity_col = None

        for col in data_cols:
            if "smiles" in col.lower():
                smiles_col = col
            if "_mean" in col.lower():
                activity_col = col

        # Loop through data to featurize each molecule
        for idx, row in self.data.iterrows():
            # Get input and output values from row
            smiles = row[smiles_col]
            y_val = row[activity_col]

            # In case of Exception
            data = None

            # TODO: what to do about these Exceptions
            try:
                # Construct Pytorch Geometric Data object
                data = featurize_single_compound(
                    smiles=smiles, y_val=y_val, create_data_object=True
                )

                # Save Pytorch Geometric Data object as .pt
                torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))

            except Exception as e:
                print("Process could not run for compound:", smiles)
                print("Error:", e)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data


# Create InMemoryDataset for pre-processed files
class CustomProcessedInMemoryDataset(InMemoryDataset):
    def __init__(self, root, data_list=[], transform=None, pre_transform=None):
        self.data_list = data_list
  
        super(CustomProcessedInMemoryDataset, self).__init__(
            root, transform, pre_transform
        )

        if len(self.data_list) > 0:
            self.root = root
            self.data, self.slices = torch.load(os.path.join(root, "processed/data.pt"))
        else:
            self.data, self.slices = torch.load(self.processed_paths[-1])

    def get_idx_split(self):
        validation_test_split = 0.4
        shuffle_dataset = True
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(torch.load(self.processed_paths[4]))
        indices = list(range(dataset_size))
        split = int(numpy.floor(validation_test_split * dataset_size))
        if shuffle_dataset:
            numpy.random.seed(random_seed)
            numpy.random.shuffle(indices)

        train_indices, val_test_indices = indices[split:], indices[:split]
        val_indices = val_test_indices[int(len(val_test_indices) / 2) :]
        test_indices = val_test_indices[: int(len(val_test_indices) / 2)]

        splits = {
            "train_graph_index": torch.tensor(train_indices, dtype=torch.long),
            "val_graph_index": torch.tensor(val_indices, dtype=torch.long),
            "test_graph_index": torch.tensor(test_indices, dtype=torch.long),
        }

        return splits

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        return "not_implemented.pt"

    @property
    def processed_file_names(self):
        if len(self.data_list) > 0:
            return "not_implemented.pt"
        else:
            return [
                "node_attrs.pt",
                "edge_attrs.pt",
                "edges.pt",
                "graph_labels.pt",
                "graph_indices_smiles.pt",
                "data.pt",
            ]

    def process(self):
        if len(self.data_list) > 0:
            self.data, self.slices = self.collate(self.data_list)
            torch.save((self.data, self.slices), os.path.join(self.root, "processed/data.pt"))    
        else:
            # Assumes that processed_files are in order node_attrs, edge_attrs, edges, graph_labels, graph_indices_smiles
            self.node_attrs = torch.load(self.processed_paths[0])
            self.edge_attrs = torch.load(self.processed_paths[1])
            self.edges = torch.load(self.processed_paths[2])
            self.graph_labels = torch.load(self.processed_paths[3])
            self.graph_indices_smiles = torch.load(self.processed_paths[4])

            # Read data into huge `Data` list.
            data_list = pytorch_geometric_data_obj_from_pytorch(
                data=[
                    self.node_attrs,
                    self.edge_attrs,
                    self.edges,
                    self.graph_labels,
                    self.graph_indices_smiles,
                ]
            )

            self.data, self.slices = self.collate(data_list)
            torch.save((self.data, self.slices), self.processed_paths[-1])

class CustomDatasetOnlyLoad(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomDatasetOnlyLoad, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self):
        return self.__num_classes__
    
    @property
    def raw_file_names(self):
        return "not_implemented"

    @property
    def processed_file_names(self):
        """
        Only load data.pt
        """

        return "data.pt"

    def get_idx_split(self):
        validation_test_split = 0.4
        shuffle_dataset = True
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(torch.load(self.processed_paths[0]))
        indices = list(range(dataset_size))
        split = int(numpy.floor(validation_test_split * dataset_size))
        if shuffle_dataset:
            numpy.random.seed(random_seed)
            numpy.random.shuffle(indices)

        train_indices, val_test_indices = indices[split:], indices[:split]
        val_indices = val_test_indices[int(len(val_test_indices) / 2) :]
        test_indices = val_test_indices[: int(len(val_test_indices) / 2)]

        splits = {
            "train_graph_index": torch.tensor(train_indices, dtype=torch.long),
            "val_graph_index": torch.tensor(val_indices, dtype=torch.long),
            "test_graph_index": torch.tensor(test_indices, dtype=torch.long),
        }

        return splits

    def process(self):
        return "not_implemented"
