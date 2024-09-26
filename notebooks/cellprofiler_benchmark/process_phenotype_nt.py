import pyarrow
import dask.dataframe as dd
import pandas as pd

# Read the parquet file
df = dd.read_parquet('interphase-reclassified_cp_phenotype_normalized.parquet', engine='pyarrow')

# Specify the gene symbols you want to extract
gene_symbols = ['nontargeting']  # Replace with your specific gene symbols

# Filter the DataFrame for the specified gene symbols
filtered_df = df[df['gene_symbol_0'].isin(gene_symbols)]

# Compute the result (this will execute the Dask task graph)
result = filtered_df.compute()

# Convert to pandas DataFrame (if not already)
result_pd = pd.DataFrame(result)

# Save to CSV
result_pd.to_csv('nontargeting.csv', index=False)