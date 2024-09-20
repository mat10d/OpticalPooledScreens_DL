import yaml 
import numpy as np

# yaml file reader
def read_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract 'mean' and 'std' from the config
    mean = config['mean']
    std = config['std']
    
    # Convert to ndarrays
    mean = np.array(mean)
    std = np.array(std)

    return mean, std
