import pandas as pd
import numpy as np
import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

def one_hot_encoding(x, permitted_list):
    """
    Each atom gets mapped to a one-hot encoded vector with 1 populated for the correct atom type. 
    If the element is not allowed, it gets appended to the end of the list where this last element being populated with 1 can later be removed as not allowed
    From https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    
    parameters
    ----------
        x: RDKit atom type
            Atom element
        permitted_list: List
            Elements allowed for algorithm
            
    returns
    -------
        binary_encoding: OHE list of atom types
    """
    
    # If atom element not allowed, append it to the end of permitted_list
    if x not in permitted_list:
        x = permitted_list[-1]
    # One hot encode permitted_list if x is present (put 1 in the position that x==s otherwise 0)
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    """
    Takes an atom and converts it to a 1D array of features
    From https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    
    parameters
    ----------
        atom: RDKit atom object
            Converted to 1D array
        use_chirality: Boolean
            Default True
        hydrogens_implicit: Boolean
            Default True, if False adds explicit Hydrogens
            
    returns
    -------
        atom_feature_vector: NumPy array of final 1D atom feature vector where all features have been added together
    """
    
    # Taken from Guacamol (TODO: change later)
    permitted_list_of_atoms = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I']

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # Get atom features
    # Atom type as OHE
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    # Number of neighbours as OHE
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    # Formal charge as OHE
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    # Hybridisation as OHE
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    # Check if atom in a ring
    is_in_a_ring_enc = [int(atom.IsInRing())]
    # Check if atom aromatic
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    # Get scaled mass of atom
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    # Scaled VdW radius
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    # Scaled covalent radius
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    # Add up all the lists of encodings we have created
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
    
    # If we use chirality, add additional list to atom features of chiral tags for atom
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    # If no explicit Hydrogens, add additional list to atom features of total number of Hydrogens attached to atom
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
        
    return np.array(atom_feature_vector)

def get_bond_features(bond, use_stereochemistry=True):
    """
    Takes a bond and converts it to 1D bond feature output
    From https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    
    parameters
    ----------
        bond: RDKit bond object
            Used to convert to features
        use_stereochemistry: Boolean
            Default True
            
    returns
    -------
        bond_feature_vector: NumPy array of final 1D bond feature vector
            
    """
    
    # Allow single, double, triple, and aromatic bond types
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    # OHE of bond type
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    # 1 if bond is conjugated and 0 if not (like single bond next to double bond shows conjugation)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    # 1 if bond is in ring otherwise 0
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    # Create final feature vector by adding together OHE and 0/1 from conjugation and ring checks
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    # If stereochemistry is True, OHE of it for bonds for E/Z isomerisation
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
        
    return np.array(bond_feature_vector)

def featurize_single_compound(smiles, y_val, create_data_object):
    """
    Creates 
    From https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    
    parameters
    ----------
        smiles: String
            SMILES corresponding to molecule
        y_val: Float
            Prediction (pXC50) value
        create_data_object: Boolean
            True if creating PyTorch Geometric Data object
            
    returns
    -------
        data: PyG Data() object or list of node_attrs, edge_attrs, edges, graph_labels, smiles
            
    """
    
    # Convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
    # Get feature dimensions
    n_nodes = mol.GetNumAtoms()
    # Double the number of edges because each bond has 2 atoms on either side
    n_edges = 2*mol.GetNumBonds()

    # Use SMILES example to get back the sizes of the atom features and edge features
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

    # Construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        # Add atom features to X using atom indices (position of atom in the molecule)
        # So for each node (atom) we add [1----n_node_features] array at the row position corresponding to the atom index
        X[atom.GetIdx(), :] = get_atom_features(atom)

    # Convert X (n_nodes, n_node_features) to torch tensor (to run operations on CPU/GPU later)
    X = torch.tensor(X, dtype = torch.float)

    # Construct edge index array E of shape (2, n_edges)
    # Get indices of rows and columns of adjacency matrix that are not 0 (so find all the bonds basically) [COO format used for edge_index in the Data object from pytorch geometric]
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    # Convert indices to torch tensors
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    # Stack rows and columns together
    E = torch.stack([torch_rows, torch_cols], dim = 0)

    # Construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
    for (k, (i,j)) in enumerate(zip(rows, cols)):
        # Get bond features for each bond present by looping through adjacency matrix
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
    # Convert edge features to torch tensor
    EF = torch.tensor(EF, dtype=torch.float)

    # Construct label tensor
    y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)
    
    if create_data_object:
        # Construct Pytorch Geometric Data object
        data = Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor, smiles=smiles)
        return data
    else:
        return X, EF, E, y_tensor, smiles