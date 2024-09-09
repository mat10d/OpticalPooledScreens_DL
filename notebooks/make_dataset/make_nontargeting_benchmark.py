
# %% Make an intermediate dataset
import pandas as pd

location = "/mnt/efs/dlmbl/G-et/csv/dataset_split_1168.csv"
benchmark_location = "/mnt/efs/dlmbl/G-et/csv/dataset_split_benchmark.csv"

metadata = pd.read_csv(location)
benchmark_metadata = pd.read_csv(benchmark_location)

# %% Randomly samply a subset of metadata that is the same size as the benchmark data
sample = metadata[metadata['gene'] == "nontargeting"]
sample = sample.sample(n=benchmark_metadata.shape[0])

# %%
sample.to_csv("/mnt/efs/dlmbl/G-et/csv/dataset_split_benchmark_nontargeting.csv", index=False)

# %%
