import numpy

def haversine_distance(df, lt1, lg1, lt2, lg2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    df:  dataframe
    lt1: latitude of the starting point. float.
    lg1: longtitde of the starting point. float.
    lt2: latitude of the ending point. float.
    lg2: longtitude of the ending point. float.
    """
    earth_radius = 6371  #approximate in kilometers

    lt_1_rad = np.radians(df[lt1])
    lt_2_rad = np.radians(df[lt2])

    lt_diff = np.radians(df[lt2]-df[lt1])
    lg_difference = np.radians(df[lg2]-df[lg1])

    aux = np.sin(lt_diff/2)**2 + np.cos(lt_1_rad) * np.cos(lt_2_rad) * np.sin(lg_difference/2)**2
    aux_2 = 2 * np.arctan2(np.sqrt(aux), np.sqrt(1-aux))
    d = (earth_radius * aux_2)

    return d
