import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from haversine import haversine_distance
import ivy
import sys
sys.setrecursionlimit(10000)

ivy.set_framework('torch')
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv("NYCTaxiFares.csv")

#haversine_distance function takes dataframe and its corresponding columns as input
df["dist_km"] = haversine_distance(df,"pickup_latitude", 'pickup_longitude', 'dropoff_latitude','dropoff_longitude')

#using pickup_datetime column to derive some useful attributes
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["EDTdate"] = df['pickup_datetime']- pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df["AMorPM"] = np.where(df['Hour']<12,'am','pm')
df["Weekday"] = df['EDTdate'].dt.strftime("%a")

#categorical and continous columns distinction
cat_cols = ['Hour','AMorPM','Weekday']
cont_cols = ['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count', 'dist_km']
y_col = ['fare_amount']

for cat in cat_cols:
  df[cat]=df[cat].astype('category')

cats = np.stack([df[col].cat.codes.values for col in cat_cols],1)
cats = torch.tensor(cats,dtype=torch.int64)
conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)
y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1,1)
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]

##A simple model
class SimpleModel(nn.Module):
  def __init__(self, emb_szs,n_cont,out_sz, layers,p=0.5):
    super().__init__()
    #self.embeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])
    #self.embeds = [UniformEmbedding(ni,nf) for ni,nf in emb_szs]
    self.embeds = [UniformEmbeddingivy(ni,nf) for ni,nf in emb_szs]
    self.emb_drop = nn.Dropout(p)
    self.bn_cont=nn.BatchNorm1d(n_cont)

    layerlist = []
    n_emb = sum([nf for ni,nf in emb_szs])
    n_in = n_emb+n_cont
    for i in layers:
      layerlist.append(nn.Linear(n_in,i))
      layerlist.append(nn.ReLU(inplace=True))
      layerlist.append(nn.BatchNorm1d(i))
      layerlist.append(nn.Dropout(p))
      n_in = i
    layerlist.append(nn.Linear(layers[-1], out_sz))
    self.layers = nn.Sequential(*layerlist)

  def forward(self, x_cat, x_cont):
    embeddings = []
    for i,e in enumerate(self.embeds):
      embeddings.append(e(x_cat[:,i]))
    x = torch.cat(embeddings,1)
    x=self.emb_drop(x)
    x_cont = self.bn_cont(x_cont)
    x = torch.cat([x,x_cont],1)
    x = self.layers(x)
    return x

model = SimpleModel(emb_szs, conts.shape[1],1,[200,100], p=0.4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#sort test and train data for categorical and continous data
batch_size = 1600
test_size = int(batch_size*0.2)
cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]


#moving all data to same device to avoid device conflicts
cat_train = cat_train.to(device)
cat_test = cat_test.to(device)
con_train = con_train.to(device)
con_test = con_test.to(device)
model = model.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

epochs = 50
losses = []
for i in range(epochs):
  i+=1
  y_pred = model(cat_train, con_train)
  loss = torch.sqrt(criterion(y_pred, y_train))
  losses.append(loss)
  print(f"Epoch: {i},  Loss={loss}")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
