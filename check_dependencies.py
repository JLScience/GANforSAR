
# numpy
try:
    import numpy as np
    print('numpy version: \t\t\t',  np.__version__)
except ImportError:
    print('! Can not import numpy.')

# h5py
try:
    import h5py
    print('h5py version: \t\t\t',  h5py.__version__)
except ImportError:
    print('! Can not import h5py.')

# matplotlib
try:
    import matplotlib
    print('matplotlib version: \t',  matplotlib.__version__)
except ImportError:
    print('! Can not import matplotlib.')

# imageio
try:
    import imageio
    print('imageio version: \t\t',  imageio.__version__)
except ImportError:
    print('! Can not import imageio.')

# scipy
try:
    import scipy as sp
    print('scipy version: \t\t\t',  sp.__version__)
except ImportError:
    print('! Can not import scipy.')

# tensorflow-gpu
try:
    import tensorflow as tf
    print('tensorflow version: \t',  tf.__version__)
except ImportError:
    print('! Can not import tensorflow.')

# keras
try:
    import keras
    print('keras version: \t\t\t',  keras.__version__)
except ImportError:
    print('! Can not import keras.')
