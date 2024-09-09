
# %% Make an intermediate dataset
import pandas as pd

location = "/mnt/efs/dlmbl/G-et/csv/dataset_split_1168.csv"

metadata = pd.read_csv(location)

# %%
assert "nontargeting" in metadata['gene'].values
assert "CCT2" in metadata['gene'].values
# %% Keep only the nontargeting and CCT2 genes
sample = metadata[metadata['gene'].isin(["nontargeting", "CCT2"])]

# %%
sample[sample.split=="train"].gene.value_counts()

# %%
# Sub-sample the non-targeting ones to have the same number of cells as CCT2
sampled_nontargeting = sample[sample.gene=="nontargeting"].sample(n=len(sample[sample.gene=="CCT2"]), random_state=42)
sampled_cct2 = sample[sample.gene=="CCT2"]

# %%
sampled = pd.concat([sampled_nontargeting, sampled_cct2])

# %%
sampled.to_csv("/mnt/efs/dlmbl/G-et/csv/dataset_split_benchmark.csv", index=False)

# %%
