"""
@colormap

"""

import matplotlib.colors as mcolors

#Blue - white - Red

def get_jet2_colormap():

    colors = [
        (0, 0, 1),     
        (0, 1, 1),     
        (1, 1, 1),     
        (1, 0.6, 0.6), 
        (1, 0, 0)      
    ]
    return mcolors.LinearSegmentedColormap.from_list("jet2", colors, N=256)
    
def get_jet2deep_colormap():

    colors = [
        (0.1, 0.1, 0.4),
        (0.2, 0.5, 0.9),
        (1.0, 1.0, 1.0),
        (0.9, 0.3, 0.3),
        (0.3, 0.0, 0.0)
    ]
    return mcolors.LinearSegmentedColormap.from_list("jet2deep", colors, N=256)