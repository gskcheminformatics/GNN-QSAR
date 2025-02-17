import os
import logging
import time
import random
import string
from collections import defaultdict
from fractions import Fraction

import pickle
import numpy
import pandas
import rdkit
import networkx
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import AllChem, ChemicalFeatures, Crippen, EState, rdMolDescriptors
from mendeleev.fetch import fetch_table

import jpype
import jpype.imports

torch.set_default_dtype(torch.float32)

BOND_ORDERS = (1.0, 1.5, 2.0, 3.0)
NODE_DEGREES = (0, 1, 2, 3, 4, 5, 6)
H_DEGREES = (0, 1, 2, 3, 4)
CHARGES = (-2, -1, 0, 1, 2)
# the following subset was chosen after counting elements in the
# zinc database of druglike compounds
ATOMIC_SYMBOLS = ("B", "C", "N", "O", "S", "F", "Cl", "Br", "I", "Si")
HYBRIDIZATIONS = (
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.OTHER,
)
CHIRALITY = (
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
)
STEREO = (
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOZ,
)


class molObjFeat:
    def __init__(self, smi, val, explicit_H, idx):
        self.smiles = smi
        self.val = torch.tensor(numpy.array([val]), dtype=torch.float)
        if explicit_H:
            self.mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        else:
            self.mol = Chem.MolFromSmiles(smi)
        self.mol_idx = idx
        self.node_features = {}
        self.edges = None
        self.edge_features = {}
        self.nx_node_features = {}


class computeAllFeatures:
    def __init__(self, input_dataframe, smiles_col, activity_col, explicit_H=False, log_dir=None):
        self.dataframe = input_dataframe

        # create mol Class for each molecule
        self.molObjsWithFeats = []

        # get cmpd ID column if exists
        col_idx = None
        data_cols = self.dataframe.columns.tolist()
        for col in data_cols:
            if "id" in col.lower():
                col_idx = col

        # Initialize filename for logging
        log_filename = (
            "".join(random.choice(string.ascii_lowercase) for i in range(42)) + ".log"
        )
        if col_idx is not None:
            # Initialize logger with compound ID (if one, otherwise random string)
            cmpd_id = self.dataframe[col_idx].unique()
            if len(cmpd_id) == 1:
                log_filename = str(cmpd_id[0]) + ".log"

        if log_dir is not None:
            log_filename = os.path.join(log_dir, log_filename)

        self.logger = logging.getLogger()
        self.logger.handlers = []
        fh = logging.FileHandler(log_filename)
        self.logger.addHandler(fh)

        # Loop through data to featurize each molecule
        for idx, row in self.dataframe.iterrows():
            # Get input and output values from row
            smiles = row[smiles_col]
            y_val = row[activity_col]

            if col_idx is not None:
                idx = row[col_idx]

            try:
                mol_feat_obj = molObjFeat(
                    smi=smiles, val=y_val, explicit_H=explicit_H, idx=idx
                )
                
                if mol_feat_obj.mol is not None:
                    if Descriptors.ExactMolWt(mol_feat_obj.mol) <= 1000:
                        self.molObjsWithFeats.append(mol_feat_obj)
            except:
                continue

        # Initialize main Mendeleev features
        ptable = fetch_table("elements")
        isotopestable = fetch_table("isotopes", index_col="id")

        self.mendeleevFeatsTable = pandas.merge(
            ptable, isotopestable, how="outer", on="atomic_number"
        )

    def init_ChemAxon(self):
        """Initialize ChemAxon Java variables for pKa and pKb calculations"""

        self.molimporter = <molimporterClass>
        self.pkaplugin = <pkapluginClass>

    def atomic_pKa_b(self, mol_feat_obj):
        """Calculates pKa and pKb per atom given mol object"""

        # Add Hs for pKa calculations
        RDKit_mol_obj = Chem.AddHs(mol_feat_obj.mol)

        # TIME
        start_time = time.time()

        num_atoms = RDKit_mol_obj.GetNumAtoms()

        for i in range(num_atoms):
            atom = RDKit_mol_obj.GetAtomWithIdx(i)
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

        SMILES_with_mapping = Chem.MolToSmiles(RDKit_mol_obj)

        # Calculate pKa and pKb per atom
        molecule = self.molimporter.importMol(SMILES_with_mapping)
        pka_calc = self.pkaplugin()
        pka_calc.setMolecule(molecule)
        # NaN for atoms without H on them
        pka_calc.run()

        # acidic pKa
        acidicpKa = jpype.JDouble[3]
        acidicIndexes = jpype.JInt[3]
        pka_calc.getMacropKaValues(pka_calc.ACIDIC, acidicpKa, acidicIndexes)

        # basic pKa
        basicpKa = jpype.JDouble[3]
        basicIndexes = jpype.JInt[3]
        pka_calc.getMacropKaValues(pka_calc.BASIC, basicpKa, basicIndexes)

        # pKa per atom
        count = molecule.getAtomCount()
        for i in range(count):
            atom_idx_from_map = molecule.getAtom(i).getAtomMap()
            apka = pka_calc.getpKa(atom_idx_from_map, pka_calc.ACIDIC)
            bpka = pka_calc.getpKa(atom_idx_from_map, pka_calc.BASIC)
            mol_feat_obj.node_features[atom_idx_from_map]["pKa"] = (
                apka if str(apka) != "nan" else 0
            )
            mol_feat_obj.node_features[atom_idx_from_map]["pKa_bool"] = (
                True if str(apka) != "nan" else False
            )
            mol_feat_obj.node_features[atom_idx_from_map]["pKb"] = (
                bpka if str(bpka) != "nan" else 0
            )
            mol_feat_obj.node_features[atom_idx_from_map]["pKb_bool"] = (
                True if str(bpka) != "nan" else 0
            )

        del acidicpKa,acidicIndexes,basicpKa,basicIndexes,pka_calc,molecule
        # TIME
        pka_time_taken = time.time() - start_time

        return pka_time_taken

    def onek(self, x, allowable_set):
        if x not in allowable_set:
            return [
                0,
            ] * len(allowable_set)
        else:
            return list(map(lambda s: x == s, allowable_set))

    def featurize_atoms(self, mol_feat_obj):
        mol = mol_feat_obj.mol
        # Get lowest energy conformer
        AllChem.EmbedMolecule(mol)
        AllChem.ComputeGasteigerCharges(mol)
        # start with the common stuff
        num_atoms = mol.GetNumAtoms()

        crippen_start_time = time.time()
        # the crippen contributions
        crippen_contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
        logps, mrs = zip(*crippen_contribs)
        crippen_time_taken = time.time() - crippen_start_time

        asa_start_time = time.time()
        # ASA contributions
        asa_contribs = rdMolDescriptors._CalcLabuteASAContribs(mol)
        asa_atomic_contribs, asa_implicit_H_contribs = asa_contribs
        asa_time_taken = time.time() - asa_start_time

        tpsa_start_time = time.time()
        # TPSA
        tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        tpsa_time_taken = time.time() - tpsa_start_time

        estates_start_time = time.time()
        # the ESTATES
        estates = EState.EStateIndices(mol, force=True) * 0.1
        estates_time_taken = time.time() - estates_start_time

        sssr_start_time = time.time()
        # the ring sizes info
        sssr = Chem.GetSymmSSSR(mol)
        sssr_time_taken = time.time() - sssr_start_time

        feature_start_time = time.time()
        # Get information for donor and acceptor
        fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        rdmol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        rdmol_feats = rdmol_featurizer.GetFeaturesForMol(mol)
        is_donor = defaultdict(bool)
        is_acceptor = defaultdict(bool)
        is_acidic = defaultdict(bool)
        is_basic = defaultdict(bool)
        # Get hydrogen bond donor/acceptor information
        for feats in rdmol_feats:
            if feats.GetFamily() == "Donor":
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_donor[u] = True
            elif feats.GetFamily() == "Acceptor":
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_acceptor[u] = True
            if (feats.GetFamily() == "NegIonizable") & (
                feats.GetType() == "AcidicGroup"
            ):
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_acidic[u] = True
            if (feats.GetFamily() == "PosIonizable") & (
                feats.GetType() == "BasicGroup"
            ):
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_basic[u] = True
        feature_time_taken = time.time() - feature_start_time

        (
            USRCAT_time_taken,
            ex_H_time_taken,
            imp_H_time_taken,
            tot_H_time_taken,
            hybrid_time_taken,
            degree_time_taken,
            aromatic_time_taken,
            ring_time_taken,
            chirality_time_taken,
            charge_time_taken,
            element_time_taken,
            gasteiger_time_taken,
            mass_time_taken,
            chirality_poss_time_taken,
            mendeleev_feats_time_taken,
        ) = [0] * 15
        # concatenate everything
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            # Features that can be computed directly from RDKit atom instances, which is a list
            (
                USRCAT_time_taken_tmp,
                ex_H_time_taken_tmp,
                imp_H_time_taken_tmp,
                tot_H_time_taken_tmp,
                hybrid_time_taken_tmp,
                degree_time_taken_tmp,
                aromatic_time_taken_tmp,
                ring_time_taken_tmp,
                chirality_time_taken_tmp,
                charge_time_taken_tmp,
                element_time_taken_tmp,
                gasteiger_time_taken_tmp,
                mass_time_taken_tmp,
                chirality_poss_time_taken_tmp,
                mendeleev_feats_time_taken_tmp,
            ) = self.featurize_atom(mol_feat_obj, i, atom)

            USRCAT_time_taken += USRCAT_time_taken_tmp
            ex_H_time_taken += ex_H_time_taken_tmp
            imp_H_time_taken += imp_H_time_taken_tmp
            tot_H_time_taken += tot_H_time_taken_tmp
            hybrid_time_taken += hybrid_time_taken_tmp
            degree_time_taken += degree_time_taken_tmp
            aromatic_time_taken += aromatic_time_taken_tmp
            ring_time_taken += ring_time_taken_tmp
            chirality_time_taken += chirality_time_taken_tmp
            charge_time_taken += charge_time_taken_tmp
            element_time_taken += element_time_taken_tmp
            gasteiger_time_taken += gasteiger_time_taken_tmp
            mass_time_taken += mass_time_taken_tmp
            chirality_poss_time_taken += chirality_poss_time_taken_tmp
            mendeleev_feats_time_taken += mendeleev_feats_time_taken_tmp

            # Donor/acceptor indicator
            mol_feat_obj.node_features[i]["isDonor"] = is_donor[i]
            mol_feat_obj.node_features[i]["isAcceptor"] = is_acceptor[i]
            # Acidic/basic indicator
            mol_feat_obj.node_features[i]["isAcidic"] = is_acidic[i]
            mol_feat_obj.node_features[i]["isBasic"] = is_basic[i]
            # crippen contribs
            mol_feat_obj.node_features[i]["logps"] = logps[i]
            mol_feat_obj.node_features[i]["mrs"] = mrs[i]
            # asa contribs
            mol_feat_obj.node_features[i]["asa_atom"] = asa_atomic_contribs[i]
            # TPSA contribs
            mol_feat_obj.node_features[i]["tpsa"] = tpsa_contribs[i]
            # estates
            mol_feat_obj.node_features[i]["estates"] = estates[i]

            sssr_start_time = time.time()
            # Count the number of rings the atom belongs to for ring size between 3 and 12
            count = [0 for _ in range(3, 13)]
            for ring in sssr:
                ring_size = len(ring)
                if i in ring and 3 <= ring_size <= 8:
                    count[ring_size - 3] += 1
            mol_feat_obj.node_features[i]["ringSize"] = count
            sssr_time_taken += time.time() - sssr_start_time

        return (
            crippen_time_taken,
            asa_time_taken,
            tpsa_time_taken,
            estates_time_taken,
            feature_time_taken,
            sssr_time_taken,
            USRCAT_time_taken,
            ex_H_time_taken,
            imp_H_time_taken,
            tot_H_time_taken,
            hybrid_time_taken,
            degree_time_taken,
            aromatic_time_taken,
            ring_time_taken,
            chirality_time_taken,
            charge_time_taken,
            element_time_taken,
            gasteiger_time_taken,
            mass_time_taken,
            chirality_poss_time_taken,
            mendeleev_feats_time_taken,
        )

    def featurize_atom(self, mol_feat_obj, idx, atom: rdkit.Chem.rdchem.Atom) -> list:
        mol_feat_obj.node_features[idx] = {}
        USRCAT_time_start = time.time()
        # USRCAT descriptor per-atom
        mol_feat_obj.node_features[idx]["USRCAT"] = rdMolDescriptors.GetUSRCAT(
            mol_feat_obj.mol, atomSelections=[[atom.GetIdx() + 1]]
        )
        USRCAT_time_taken = time.time() - USRCAT_time_start

        ex_H_time_start = time.time()
        # explicit Hs
        mol_feat_obj.node_features[idx]["num_explicit_Hs"] = [atom.GetNumExplicitHs()]
        ex_H_time_taken = time.time() - ex_H_time_start

        imp_H_time_start = time.time()
        # implicit Hs
        mol_feat_obj.node_features[idx]["num_implicit_Hs"] = [atom.GetNumImplicitHs()]
        imp_H_time_taken = time.time() - imp_H_time_start

        tot_H_time_start = time.time()
        # number of hydrogens, both implicit and explicit
        mol_feat_obj.node_features[idx]["totalNumHs"] = self.onek(
            atom.GetTotalNumHs(includeNeighbors=True), H_DEGREES
        )
        tot_H_time_taken = time.time() - tot_H_time_start

        hybrid_time_start = time.time()
        # hybridization state for symmetry
        mol_feat_obj.node_features[idx]["hybridization"] = self.onek(
            atom.GetHybridization(), HYBRIDIZATIONS
        )
        hybrid_time_taken = time.time() - hybrid_time_start

        degree_time_start = time.time()
        # degree of connection for the node
        mol_feat_obj.node_features[idx]["degrees"] = self.onek(
            atom.GetTotalDegree(), NODE_DEGREES
        )
        degree_time_taken = time.time() - degree_time_start

        aromatic_time_start = time.time()
        # is the atom part of an aromatic chain?
        mol_feat_obj.node_features[idx]["isAromatic"] = [atom.GetIsAromatic()]
        aromatic_time_taken = time.time() - aromatic_time_start

        ring_time_start = time.time()
        # is the atom in a ring?
        mol_feat_obj.node_features[idx]["isInRing"] = [atom.IsInRing()]
        ring_time_taken = time.time() - ring_time_start

        chirality_time_start = time.time()
        # chirality of the atom if specified
        mol_feat_obj.node_features[idx]["chirality"] = self.onek(
            atom.GetChiralTag(), CHIRALITY
        )
        chirality_time_taken = time.time() - chirality_time_start

        charge_time_start = time.time()
        # formal charge of atom
        mol_feat_obj.node_features[idx]["formalCharge"] = self.onek(
            atom.GetFormalCharge(), CHARGES
        )
        charge_time_taken = time.time() - charge_time_start

        element_time_start = time.time()
        # element type
        mol_feat_obj.node_features[idx]["element"] = self.onek(
            atom.GetSymbol(), ATOMIC_SYMBOLS
        )
        element_time_taken = time.time() - element_time_start

        gasteiger_time_start = time.time()
        # gasteiger charge
        gasteiger_charge = atom.GetProp("_GasteigerCharge")
        if gasteiger_charge in ["-nan", "nan", "-inf", "inf"]:
            gasteiger_charge = 0.0
        mol_feat_obj.node_features[idx]["gasteiger"] = float(gasteiger_charge)
        gasteiger_time_taken = time.time() - gasteiger_time_start

        mass_time_start = time.time()
        # atom mass
        mol_feat_obj.node_features[idx]["mass"] = atom.GetMass() * 0.01
        mass_time_taken = time.time() - mass_time_start

        chirality_poss_time_start = time.time()
        # chirality possible
        mol_feat_obj.node_features[idx]["chiralityPossible"] = atom.HasProp(
            "_ChiralityPossible"
        )
        chirality_poss_time_taken = time.time() - chirality_poss_time_start

        mendeleev_feats_time_start = time.time()
        # Mendeleev features
        atomic_num = atom.GetAtomicNum()
        mendeleev_feats_cols = self.mendeleevFeatsTable.columns
        mendeleev_feats_atomic_num = self.mendeleevFeatsTable[
            self.mendeleevFeatsTable["atomic_number"] == atomic_num
        ]
        for col in mendeleev_feats_cols:
            if col not in ["atomic_number", "symbol"]:
                if "spin" in col:
                    mol_feat_obj.node_features[idx]["mendeleev_" + col] = float(Fraction(list(set(mendeleev_feats_atomic_num[col]))[0]))
                else:
                    mol_feat_obj.node_features[idx]["mendeleev_" + col] = list(set(mendeleev_feats_atomic_num[col]))[0]
        mendeleev_feats_time_taken = time.time() - mendeleev_feats_time_start

        return (
            USRCAT_time_taken,
            ex_H_time_taken,
            imp_H_time_taken,
            tot_H_time_taken,
            hybrid_time_taken,
            degree_time_taken,
            aromatic_time_taken,
            ring_time_taken,
            chirality_time_taken,
            charge_time_taken,
            element_time_taken,
            gasteiger_time_taken,
            mass_time_taken,
            chirality_poss_time_taken,
            mendeleev_feats_time_taken,
        )

    def featurize_bonds(self, mol_feat_obj):
        bond_type_time_taken = 0
        aromatic_time_taken = 0
        conjugated_time_taken = 0
        ring_time_taken = 0
        stereo_time_taken = 0

        mol = mol_feat_obj.mol

        (rows, cols) = numpy.nonzero(GetAdjacencyMatrix(mol))

        torch_rows = torch.from_numpy(rows.astype(numpy.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(numpy.int64)).to(torch.long)
        # Stack rows and columns together
        mol_feat_obj.edges = torch.stack([torch_rows, torch_cols], dim=0)

        # Construct edge feature array
        for (k, (i, j)) in enumerate(zip(rows, cols)):
            mol_feat_obj.edge_features[k] = {}
            # Get bond features for each bond present by looping through adjacency matrix
            (
                bond_type_time_taken_tmp,
                aromatic_time_taken_tmp,
                conjugated_time_taken_tmp,
                ring_time_taken_tmp,
                stereo_time_taken_tmp,
            ) = self.featurize_bond(
                mol_feat_obj=mol_feat_obj,
                idx=k,
                bond=mol.GetBondBetweenAtoms(int(i), int(j)),
            )

            bond_type_time_taken += bond_type_time_taken_tmp
            aromatic_time_taken += aromatic_time_taken_tmp
            conjugated_time_taken += conjugated_time_taken_tmp
            ring_time_taken += ring_time_taken_tmp
            stereo_time_taken += stereo_time_taken_tmp

        return (
            bond_type_time_taken,
            aromatic_time_taken,
            conjugated_time_taken,
            ring_time_taken,
            stereo_time_taken,
        )

    def featurize_bond(self, mol_feat_obj, idx, bond):
        bond_type_start_time = time.time()
        # first the bond types
        mol_feat_obj.edge_features[idx]["bondType"] = self.onek(
            bond.GetBondTypeAsDouble(), BOND_ORDERS
        )
        bond_type_time_taken = time.time() - bond_type_start_time

        aromatic_start_time = time.time()
        # is the bond aromatic?
        mol_feat_obj.edge_features[idx]["isAromatic"] = [bond.GetIsAromatic()]
        aromatic_time_taken = time.time() - aromatic_start_time

        conjugated_start_time = time.time()
        # is it conjugated? (nb: not the same as aromatic)
        mol_feat_obj.edge_features[idx]["isConjugated"] = [bond.GetIsConjugated()]
        conjugated_time_taken = time.time() - conjugated_start_time

        ring_start_time = time.time()
        # is it part of a ring
        mol_feat_obj.edge_features[idx]["isInRing"] = [bond.IsInRing()]
        ring_time_taken = time.time() - ring_start_time

        stereo_start_time = time.time()
        # does it have defined stereochemistry
        mol_feat_obj.edge_features[idx]["isStereo"] = self.onek(
            bond.GetStereo(), STEREO
        )
        stereo_time_taken = time.time() - stereo_start_time

        return (
            bond_type_time_taken,
            aromatic_time_taken,
            conjugated_time_taken,
            ring_time_taken,
            stereo_time_taken,
        )

    def mol_to_nx(self, mol: rdkit.Chem.rdchem.Mol) -> networkx.Graph:
        G = networkx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=bond.GetBondType(),
            )
        return G

    def featurize_nx_nodes(self, mol_feat_obj):
        nx_time_start = time.time()

        mol = mol_feat_obj.mol
        G = self.mol_to_nx(mol)
        try:
            avg_degr = networkx.assortativity.average_neighbor_degree(G)
            deg_cent = networkx.centrality.degree_centrality(G)
            cls_cent = networkx.centrality.closeness_centrality(G)
            btw_cent = networkx.centrality.betweenness_centrality(G)
            ktz_cent = networkx.centrality.katz_centrality(G)
            fcc_cent = networkx.centrality.current_flow_closeness_centrality(G)
            fbc_cent = networkx.centrality.current_flow_betweenness_centrality(G)
            lod_cent = networkx.centrality.load_centrality(G)
            hrm_cent = networkx.centrality.harmonic_centrality(G)
            con_hole = networkx.structuralholes.constraint(G)
            bar_dist = networkx.distance_measures.barycenter(G)
            cen_dist = networkx.distance_measures.center(G)
            per_dist = networkx.distance_measures.periphery(G)
            ecc_dist = networkx.distance_measures.eccentricity(G)
            for node in G.nodes():
                mol_feat_obj.nx_node_features[node] = {}
                mol_feat_obj.nx_node_features[node]["degreeCentrality"] = deg_cent[node]
                mol_feat_obj.nx_node_features[node]["closenessCentrality"] = cls_cent[
                    node
                ]
                mol_feat_obj.nx_node_features[node]["betweennessCentrality"] = btw_cent[
                    node
                ]
                mol_feat_obj.nx_node_features[node]["katzCentrality"] = ktz_cent[node]
                mol_feat_obj.nx_node_features[node]["fccCentrality"] = fcc_cent[node]
                mol_feat_obj.nx_node_features[node]["fbcCentrality"] = fbc_cent[node]
                mol_feat_obj.nx_node_features[node]["loadCentrality"] = lod_cent[node]
                mol_feat_obj.nx_node_features[node]["harmonicCentrality"] = (
                    hrm_cent[node] * 0.05
                )
                mol_feat_obj.nx_node_features[node]["conHole"] = con_hole[node] * 0.05
                mol_feat_obj.nx_node_features[node]["degree"] = avg_degr[node] / max(
                    NODE_DEGREES
                )
                mol_feat_obj.nx_node_features[node]["ecc"] = ecc_dist[node] / len(G)
                mol_feat_obj.nx_node_features[node]["baryCenter"] = int(
                    node in bar_dist
                )
                mol_feat_obj.nx_node_features[node]["center"] = int(node in cen_dist)
                mol_feat_obj.nx_node_features[node]["periphery"] = int(node in per_dist)
        except networkx.NetworkXError:
            self.logger.error("Cannot convert molecule to NetworkX", exc_info=True)

        nx_time_taken = time.time() - nx_time_start

        return nx_time_taken

    def get_mol_descriptors(self, cx_pK=False, log_dir=None, time_features=True):
        counter = 0

        (
            crippen_time_taken,
            asa_time_taken,
            tpsa_time_taken,
            estates_time_taken,
            feature_time_taken,
            sssr_time_taken,
            USRCAT_time_taken,
            ex_H_time_taken,
            imp_H_time_taken,
            tot_H_time_taken,
            hybrid_time_taken,
            degree_time_taken,
            aromatic_time_taken,
            ring_time_taken,
            chirality_time_taken,
            charge_time_taken,
            element_time_taken,
            gasteiger_time_taken,
            mass_time_taken,
            chirality_poss_time_taken,
            mendeleev_feats_time_taken,
        ) = [0] * 21
        (
            bond_type_time_taken,
            aromatic_time_taken_bond,
            conjugated_time_taken_bond,
            ring_time_taken_bond,
            stereo_time_taken_bond,
        ) = [0] * 5
        nx_time_taken = 0
        atomic_pKa_b_time_taken = 0

        if cx_pK:
            self.init_ChemAxon()

        if log_dir is not None:
            counters_already_ran = [int(i.split(".pickle")[0].split("_")[-1]) for i in os.listdir(os.path.join(log_dir, "dataset")) if i.startswith("mol_label")]
        else:
            counters_already_ran = []

        for mol_obj_idx in range(len(self.molObjsWithFeats)):
            mol_obj = self.molObjsWithFeats[mol_obj_idx]
            try:
                counter += 1

                if counter % 100 == 0:
                    self.logger.info("%d molecules processed", counter)

                
                if counter not in counters_already_ran:
                    # Featurize atoms
                    atom_feats = self.featurize_atoms(mol_obj)
                    if time_features:
                        crippen_time_taken += atom_feats[0]
                        asa_time_taken += atom_feats[1]
                        tpsa_time_taken += atom_feats[2]
                        estates_time_taken += atom_feats[3]
                        feature_time_taken += atom_feats[4]
                        sssr_time_taken += atom_feats[5]
                        USRCAT_time_taken += atom_feats[6]
                        ex_H_time_taken += atom_feats[7]
                        imp_H_time_taken += atom_feats[8]
                        tot_H_time_taken += atom_feats[9]
                        hybrid_time_taken += atom_feats[10]
                        degree_time_taken += atom_feats[11]
                        aromatic_time_taken += atom_feats[12]
                        ring_time_taken += atom_feats[13]
                        chirality_time_taken += atom_feats[14]
                        charge_time_taken += atom_feats[15]
                        element_time_taken += atom_feats[16]
                        gasteiger_time_taken += atom_feats[17]
                        mass_time_taken += atom_feats[18]
                        chirality_poss_time_taken += atom_feats[19]
                        mendeleev_feats_time_taken += atom_feats[20]
                    # Featurize bonds
                    bond_feats = self.featurize_bonds(mol_obj)
                    if time_features:
                        bond_type_time_taken += bond_feats[0]
                        aromatic_time_taken_bond += bond_feats[1]
                        conjugated_time_taken_bond += bond_feats[2]
                        ring_time_taken_bond += bond_feats[3]
                        stereo_time_taken_bond += bond_feats[4]
                    # Featurize networkx
                    nx_time =  self.featurize_nx_nodes(mol_obj)
                    if time_features:
                        nx_time_taken += nx_time
                     
                    # If using ChemAxon pK, also featurize this
                    if cx_pK:
                        pka_time = self.atomic_pKa_b(mol_obj)
                        if time_features:
                            atomic_pKa_b_time_taken += pka_time
                    # Save mol object dictionary to files with counter then reset object
                    if log_dir is not None:
                        log_dir_data = os.path.join(log_dir, "dataset")
                        if not os.path.isdir(log_dir_data):
                            os.makedirs(log_dir_data)

                        with open(os.path.join(log_dir_data, "mol_label_"+str(counter)+".pickle"), "wb") as f:
                            pickle.dump(mol_obj.val, f)

                        with open(os.path.join(log_dir_data, "mol_info_"+str(counter)+".pickle"), "wb") as f:
                            pickle.dump(mol_obj.smiles, f)

                        with open(os.path.join(log_dir_data, "node_features_"+str(counter)+".pickle"), "wb") as f:
                            pickle.dump(mol_obj.node_features, f)

                        with open(os.path.join(log_dir_data, "edges_"+str(counter)+".pickle"), "wb") as f:
                            pickle.dump(mol_obj.edges, f)

                        with open(os.path.join(log_dir_data, "edge_features_"+str(counter)+".pickle"), "wb") as f:
                            pickle.dump(mol_obj.edge_features, f)

                        with open(os.path.join(log_dir_data, "nx_node_features_"+str(counter)+".pickle"), "wb") as f:
                            pickle.dump(mol_obj.nx_node_features, f)

                        self.molObjsWithFeats[mol_obj_idx] = None
                        # Remove RDKit object to save memory
                        #mol_obj.mol = None
            except Exception as e:
                self.logger.error("Could not process molecule", exc_info=True)
                continue

        if counter > 0 and time_features:
            self.logger.info("TIME TAKEN PER MOLECULE STATISTICS:\n")
            self.logger.info("NODE FEATURES\n")
            self.logger.info("Crippen contribs: %f", crippen_time_taken / counter)
            self.logger.info("Asa contribs: %f", asa_time_taken / counter)
            self.logger.info("Tpsa contribs: %f", tpsa_time_taken / counter)
            self.logger.info("Estates: %f", estates_time_taken / counter)
            self.logger.info("Features: %f", feature_time_taken / counter)
            self.logger.info("SSSR: %f", sssr_time_taken / counter)
            self.logger.info("USRCAT: %f", USRCAT_time_taken / counter)
            self.logger.info("Explicit H: %f", ex_H_time_taken / counter)
            self.logger.info("Implicit H: %f", imp_H_time_taken / counter)
            self.logger.info("Total H: %f", tot_H_time_taken / counter)
            self.logger.info("Hybridization: %f", hybrid_time_taken / counter)
            self.logger.info("Node degree: %f", degree_time_taken / counter)
            self.logger.info("Aromatic: %f", aromatic_time_taken / counter)
            self.logger.info("Ring: %f", ring_time_taken / counter)
            self.logger.info("Chirality: %f", chirality_time_taken / counter)
            self.logger.info("Charge: %f", charge_time_taken / counter)
            self.logger.info("Element: %f", element_time_taken / counter)
            self.logger.info("Gasteiger charge: %f", gasteiger_time_taken / counter)
            self.logger.info("Mass: %f", mass_time_taken / counter)
            self.logger.info("Chirality possible: %f", chirality_poss_time_taken / counter)
            self.logger.info("Mendeleev: %f", mendeleev_feats_time_taken / counter)
            self.logger.info("NetworkX: %f", nx_time_taken / counter)
            self.logger.info("ChemAxon pKa: %f", atomic_pKa_b_time_taken / counter)
            self.logger.info("\n")

            self.logger.info("BOND FEATURES\n")
            self.logger.info("Bond type: %f", bond_type_time_taken / counter)
            self.logger.info("Aromatic bond: %f", aromatic_time_taken_bond / counter)
            self.logger.info("Conjugated bond: %f", conjugated_time_taken_bond / counter)
            self.logger.info("Ring bond: %f", ring_time_taken_bond / counter)
            self.logger.info("Stereo bond: %f", stereo_time_taken_bond / counter)
