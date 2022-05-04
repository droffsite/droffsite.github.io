import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from netCDF4 import Dataset

class SEMPAImage(object):
    sempa_file_suffix = 'sempa'
    scale_multiplier = 10e8

    def __init__(self, file_path):
        self.dataset =  Dataset(file_path, 'r')
        self.image_data = np.asarray(self.get_key('image_data').T)
        self.axis = str(self.get_key('channel_name')[1], 'utf-8')
        self.magnification = np.abs(self.get_key('magnification'])


    def get_key(self, key):
        data = self.dataset.variables[key][...].data

        if data.shape != ():
            data = b''.join(data).decode()

        return data


    def get_dimension(self):
        return self.image_data.shape[0]


    def get_scale(self):
        full_scale = np.abs(self.get_key('vertical_full_scale'))

        return full_scale / self.magnification / self.get_dimension()


    def max(self):
        return self.image_data.max()


    def min(self):
        return self.image_data.min()

