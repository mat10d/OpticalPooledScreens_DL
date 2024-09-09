
# %% Make an intermediate dataset
# This includes *only* a subset of barcodes that are nontargeting
import pandas as pd
import numpy as np

# %%
location = "/mnt/efs/dlmbl/G-et/csv/dataset_split_1168.csv"

metadata = pd.read_csv(location)
# %%
sample = metadata[metadata['gene'] == "nontargeting"]
np.random.seed(42)
barcodes = np.random.choice(
    sample["barcode"].sort_values().unique(), 
    size=10, 
    replace=False, 
)
# %% Randomly samply a subset of metadata that is the same size as the benchmark data
sample = metadata[metadata['barcode'].isin(barcodes)]

# %%
sample["split"].value_counts()
# %%
# make sure each barcode is in each split
for split in ["train", "val", "test"]:
    assert set(barcodes) == set(sample[sample["split"] == split]["barcode"].unique())

# %%
sample.to_csv("/mnt/efs/dlmbl/G-et/csv/dataset_split_benchmark_nontargeting_barcode.csv", index=False)


# %%
