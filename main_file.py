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

#using pickup_datetime column to derive some useful attributes
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["EDTdate"] = df['pickup_datetime']- pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df["AMorPM"] = np.where(df['Hour']<12,'am','pm')
df["Weekday"] = df['EDTdate'].dt.strftime("%a")
