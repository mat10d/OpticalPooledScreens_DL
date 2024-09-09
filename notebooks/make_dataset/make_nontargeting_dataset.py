
# %% Make an intermediate dataset
import pandas as pd

location = "/mnt/efs/dlmbl/G-et/csv/dataset_split_1168.csv"

metadata = pd.read_csv(location)

# %%
assert "nontargeting" in metadata['gene'].values
# %% Keep only the nontargeting and CCT2 genes
sample = metadata[metadata['gene'] == "nontargeting"]

# %%
sample.to_csv("/mnt/efs/dlmbl/G-et/csv/dataset_split_nontargeting.csv", index=False)

# %%
