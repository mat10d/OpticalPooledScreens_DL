
# %% Make an intermediate dataset
# This includes *only* a subset of barcodes that are nontargeting and *all* barcodes that are CCT2
import pandas as pd
import numpy as np

# %%
location = "/mnt/efs/dlmbl/G-et/csv/dataset_split_1168.csv"
nontargeting_location = "/mnt/efs/dlmbl/G-et/csv/dataset_split_benchmark_nontargeting_barcode.csv" 

metadata = pd.read_csv(location)
nontargeting_metadata = pd.read_csv(nontargeting_location)
# %%
cct2 = metadata[metadata['gene'] == "CCT2"]
# %%
sample = pd.concat([nontargeting_metadata, cct2])
sample["split"].value_counts()
# %%
barcodes = sample["barcode"].sort_values().unique()
genes = sample["gene"].sort_values().unique()
# %%
# make sure each barcode is in each split
for split in ["train", "val", "test"]:
    assert set(barcodes) == set(sample[sample["split"] == split]["barcode"].unique())

# %%
sample.to_csv("/mnt/efs/dlmbl/G-et/csv/dataset_split_benchmark_nontargeting_barcode_with_cct2.csv", index=False)

# %%
