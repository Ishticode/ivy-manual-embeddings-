# ivy-manual-embeddings
Some embedding layer implementation using ivy library. Just for fun. It is based on [NYCTaxiFare dataset](https://www.kaggle.com/tanyildizderya/nyctaxifares) from kaggle (smaller). Most of the feature extraction is similar to Jose Portilla's.
Uses [haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) to extract the feature of distance (km) from a given latitude and longtitude of pickup and dropoff locations.
Initial loss graph with lr = 0.01 and 50 epochs:

![50_epochs](https://user-images.githubusercontent.com/53497039/153295909-dd86b092-ab4a-4119-aa1b-f9275eb5e8ba.png)
