import pandas as pd
import numpy as np

# read in /lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_nontargeting-CCT2_diane.csv
nontargeting = pd.read_csv('/lab/barcheese01/aconcagua_results/primary_screen_patches_splits/dataset_CCT2-nontargeting-diane.csv')

# filter the dataframe to only nontargeting gene
nontargeting = nontargeting[nontargeting['gene'] == 'nontargeting']

# compute the total number of rows
total_rows = len(nontargeting)

# Read the original CSV file
nontargeting = pd.read_csv('nontargeting.csv')

# Sample from the dataframe
nontargeting_sample = nontargeting.sample(n=total_rows, random_state=42, replace=False)

# Write the sampled data to a new CSV file
output_file = 'nontargeting_subsample_diane.csv'
nontargeting_sample.to_csv(output_file, index=False)

print(f"Sampled dataset with {len(nontargeting_sample)} rows has been saved to {output_file}")