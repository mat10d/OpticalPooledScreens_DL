import yaml 
import numpy as np


# Yaml file reader
def read_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract 'Dataset mean' and 'Dataset std' from the config
    mean = config['Dataset mean'][0]  # Access the first (and only) element of the list
    std = config['Dataset std'][0]

    # Split the strings and convert to floats
    mean = [float(i) for i in mean.split()]
    std = [float(i) for i in std.split()]
    
    # Convert to ndarrays
    mean = np.array(mean)
    std = np.array(std)

    return mean, std
