from make_data.get_tautomers import getTautomers
from make_data.all_featurize_data import computeAllFeatures
from make_data.dataset import FEATURES_DICT
import argparse
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import logging

ALL_NX_FEATURES = FEATURES_DICT["node"]["graph"]
ALL_NODE_FEATURES = (
    FEATURES_DICT["node"]["knowledge"]
    + FEATURES_DICT["node"]["feature"]
    + FEATURES_DICT["node"]["atomic"]
    + FEATURES_DICT["node"]["generic"]
    + FEATURES_DICT["node"]["topological"]
)
ALL_EDGE_FEATURES = (
    FEATURES_DICT["edge"]["generic"]
    + FEATURES_DICT["edge"]["atomic"]
    + FEATURES_DICT["edge"]["topological"]
)


def get_feat_deviations(
    taut_mol_objs, all_features_required, nx_node_edge_flag, dict_to_append
):
    for feat in all_features_required:
        mean_atomwise_vals_per_cmpd = []
        for atom in list(taut_mol_objs[0].node_features.keys()):
            atomwise_vals = []
            for taut_mol_obj in taut_mol_objs:
                try:
                    atom_mass = taut_mol_obj.node_features[atom]["mass"]
                    if atom_mass != 0.01008:
                        if nx_node_edge_flag == "nx":
                            val = taut_mol_obj.nx_node_features[atom][feat]
                        elif nx_node_edge_flag == "node":
                            val = taut_mol_obj.node_features[atom][feat]
                        elif nx_node_edge_flag == "edge":
                            val = taut_mol_obj.edge_features[atom][feat]
                except:
                    val = None

                if val is not None:
                    if type(val) != list:
                        atomwise_vals.append(float(val))
                    else:
                        atomwise_vals.append(np.mean([float(i) for i in val]))

            # dict_to_append[feat][atom] = [numpy.mean(atomwise_vals), numpy.std(atomwise_vals)] # UNCOMMENT TO CALCULATE MEAN CONTRIBUTIONS PER ATOM, initialise in compare_taut_features as {i: {} for i in ALL_NX_FEATURES}
            mean_atomwise_vals_per_cmpd.append(np.mean(atomwise_vals))

        dict_to_append[feat] = [
            np.mean(mean_atomwise_vals_per_cmpd),
            np.std(mean_atomwise_vals_per_cmpd),
        ]

    return dict_to_append


def compare_taut_features(taut_mol_objs):
    # For each atom, get the deviation in node features across tautomers
    # Dictionaries are {feature_name: [mean, standard_deviation]} across all tautomers
    nx_feat_deviation = {i: [] for i in ALL_NX_FEATURES}
    node_feat_deviation = {i: [] for i in ALL_NODE_FEATURES}
    edge_feat_deviation = {i: [] for i in ALL_EDGE_FEATURES}

    nx_feat_deviation = get_feat_deviations(
        taut_mol_objs=taut_mol_objs,
        all_features_required=ALL_NX_FEATURES,
        nx_node_edge_flag="nx",
        dict_to_append=nx_feat_deviation,
    )
    node_feat_deviation = get_feat_deviations(
        taut_mol_objs=taut_mol_objs,
        all_features_required=ALL_NODE_FEATURES,
        nx_node_edge_flag="node",
        dict_to_append=node_feat_deviation,
    )
    edge_feat_deviation = get_feat_deviations(
        taut_mol_objs=taut_mol_objs,
        all_features_required=ALL_EDGE_FEATURES,
        nx_node_edge_flag="edge",
        dict_to_append=edge_feat_deviation,
    )

    return nx_feat_deviation, node_feat_deviation, edge_feat_deviation


def worker(cmpd_id):
    # Get activity and structure columns
    data_cols = taut_df.columns.tolist()
    smiles_col = None
    activity_col = None

    for col in data_cols:
        if "smiles" in col.lower():
            smiles_col = col
        if "_mean" in col.lower():
            activity_col = col

    # Compute all features for tautomers
    comp_feats = computeAllFeatures(
        input_dataframe=taut_df[taut_df["CMPD_ID"] == int(cmpd_id)],
        smiles_col=smiles_col,
        activity_col=activity_col,
        explicit_H=True,
        log_dir=log_dir,
    )
    comp_feats.get_mol_descriptors(cx_pK=True)

    nx_feat_deviation, node_feat_deviation, edge_feat_deviation = compare_taut_features(
        [i for i in comp_feats.molObjsWithFeats]
    )

    full_node_dict = {**nx_feat_deviation, **node_feat_deviation}
    edge_dict = edge_feat_deviation

    mean_node_df = pd.DataFrame(columns=ALL_NX_FEATURES + ALL_NODE_FEATURES)
    standard_deviation_node_df = pd.DataFrame(
        columns=ALL_NX_FEATURES + ALL_NODE_FEATURES
    )

    mean_edge_df = pd.DataFrame(columns=ALL_EDGE_FEATURES)
    standard_deviation_edge_df = pd.DataFrame(columns=ALL_EDGE_FEATURES)

    # Node dictionary
    # Mean values
    full_node_dict_mean = {i: [full_node_dict[i][0]] for i in full_node_dict.keys()}
    mean_node_df = pd.DataFrame.from_dict(full_node_dict_mean)
    # Stddev values
    full_node_dict_stddev = {i: [full_node_dict[i][1]] for i in full_node_dict.keys()}
    standard_deviation_node_df = pd.DataFrame.from_dict(full_node_dict_stddev)

    # Edge dictionary
    # Mean values
    edge_dict_mean = {i: [edge_dict[i][0]] for i in edge_dict.keys()}
    mean_edge_df = pd.DataFrame.from_dict(edge_dict_mean)
    # Stddev values
    edge_dict_stddev = {i: [edge_dict[i][1]] for i in edge_dict.keys()}
    standard_deviation_edge_df = pd.DataFrame.from_dict(edge_dict_stddev)

    mean_node_df.to_csv(
        os.path.join(out_dir, str(cmpd_id) + "mean_node_df.csv"), index=False
    )
    standard_deviation_node_df.to_csv(
        os.path.join(out_dir, str(cmpd_id) + "stddev_node_df.csv"), index=False
    )
    mean_edge_df.to_csv(
        os.path.join(out_dir, str(cmpd_id) + "mean_edge_df.csv"), index=False
    )
    standard_deviation_edge_df.to_csv(
        os.path.join(out_dir, str(cmpd_id) + "stddev_edge_df.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    # Save to output directory
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    global out_dir
    out_dir = args.out_dir

    # Get tautomers and save in path
    global taut_df
    taut_df = getTautomers(data_path=args.data_path, out_path=None).generate_tauts()

    all_cmpd_ids_iterate = list(taut_df["CMPD_ID"].unique())

    # Initialize master logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info("Compounds to process: %d", len(all_cmpd_ids_iterate))

    global log_dir
    log_dir = os.path.join(out_dir, "log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    # Get features per compound
    with mp.Pool(processes=10) as pool:
        pool.map(worker, all_cmpd_ids_iterate)
        pool.close()
