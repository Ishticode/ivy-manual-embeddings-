# ivy-manual-embeddings
Note: This is not how embedding functionality would be implemented into the actual ivy lib. That has been done in embedding_layer_proto or embedding_frozen branch in my ivy repository (forked from unifyai/ivy)


Some embedding layer implementation using ivy library. Just for fun. It is based on [NYCTaxiFare dataset](https://www.kaggle.com/tanyildizderya/nyctaxifares) from kaggle (cut down to 120,000 rows). Most of the feature extraction is similar to Jose Portilla's.
Uses [haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) to extract the feature of distance (km) from a given latitude and longtitude of pickup and dropoff locations.
Initial loss graph with lr = 0.01 and 50 epochs:

![50_epochs](https://user-images.githubusercontent.com/53497039/153298051-e2d47083-0791-4ac6-9a9d-2ceb82711e57.png)

