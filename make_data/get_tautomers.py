import pandas as pd
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem


class getTautomers:
    def __init__(self, data_path, out_path=None):
        if ".tsv" in data_path:
            self.input_df = pd.read_csv(data_path, sep="\t")
        else:
            self.input_df = pd.read_csv(data_path)

        self.out_path = out_path

    def generate_tauts(self):
        data_cols = self.input_df.columns.tolist()
        smiles_col = None
        activity_col = None

        for col in data_cols:
            if "smiles" in col.lower():
                smiles_col = col
            if "_mean" in col.lower():
                activity_col = col

        # Enumerate tautomers
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(
            64
        )  # Set max to be the same as the internal tautomer calculator

        cmpd_id = []
        taut_smi = []
        taut_activity = []

        for idx, row in self.input_df.iterrows():
            tauts = []
            if Chem.MolFromSmiles(row[smiles_col]) is not None:
                try:
                    tauts = enumerator.Enumerate(Chem.MolFromSmiles(row[smiles_col]))
                except:
                    tauts = []

            # Only save compounds that have tautomers
            if len(tauts) > 1:
                for taut in tauts:
                    cmpd_id.append(idx)
                    taut_smi.append(Chem.MolToSmiles(taut))
                    taut_activity.append(row[activity_col])

        taut_df = pd.DataFrame()
        taut_df[activity_col] = taut_activity
        taut_df[smiles_col] = taut_smi
        taut_df["CMPD_ID"] = cmpd_id

        return taut_df

    def main(self):
        if self.out_path is not None:
            # Runs and saves tautomers in new file at out_path
            taut_df = self.generate_tauts()
            if ".tsv" in self.out_path:
                taut_df.to_csv(self.out_path, sep="\t", index=False)
            else:
                pd.to_csv(self.out_path, index=False)
        else:
            raise ValueError("Output filename required")
