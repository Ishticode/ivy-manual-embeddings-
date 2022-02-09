import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from haversine import haversine_distance
import ivy
#%matplotlib inline
ivy.set_framework('torch')
cuda = torch.device('cuda')
df = pd.read_csv("NYCTaxiFares.csv")

#haversine_distance function takes dataframe and its corresponding columns as input
df["dist_km"] = haversine_distance(df,"pickup_latitude", 'pickup_longitude', 'dropoff_latitude','dropoff_longitude')
