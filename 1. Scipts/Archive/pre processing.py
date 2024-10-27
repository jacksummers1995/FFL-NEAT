# %% -- libs

import pandas as pd

# %% -- Read in Data

# Meta Data
meta = pd.read_csv('meta_data.csv')
feature_types_dict = dict(zip(meta['feature'], meta['feature_type']))

apply_stats_features = meta[meta['apply_stats'] == "TRUE"]['feature'].tolist()

# Current Season
data_22_23 = pd.read_csv('22-23 FFL.csv', dtype=feature_types_dict)