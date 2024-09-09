import numpy as np
from skimage.exposure import rescale_intensity
from torch import from_numpy
from skimage.measure import centroid


def rescale_bf(img,quantiles = [0.01,0.99]):
    min_max = np.quantile(img,quantiles)
    rescaled = (
        rescale_intensity(
            img,
            in_range=(min_max[0],min_max[1]),
            out_range=(0,1)) -1
    ) * -1
    rescaled = np.clip(rescaled,0,1)
    return rescaled

def rescale_bra(bra_tl,quantiles = [0.001,0.999]):
    min_max = np.quantile(bra_tl,quantiles)
    rescaled = rescale_intensity(
        bra_tl,
        in_range=(min_max[0],min_max[1]),
        out_range=(0,1)
    )
    return rescaled

def complex_normalisation(
        input_series, 
        bf_quant, 
        bra_quant
    ):
    """
    input_series: np.ndarray 
        dimensions = time, channel, y, x
    bf_quant: list
        lower and upper quantiles for rescaling brightfield images (channel 0)
        Performed for each image individually
    bra_quant: list
        lower and upper quantiles for rescaling brachyury images (channel 1)
        rescaled across the timelapse
    """
    bf_tl = input_series[:,0,:,:]
    bra_tl = input_series[:,1,:,:]
    out_bf = np.expand_dims(np.array([rescale_bf(img,bf_quant) for img in bf_tl]),1)
    out_bra = np.expand_dims(rescale_bra(bra_tl,bra_quant),1)
    return np.concatenate((out_bf,out_bra),axis=1)



class NormalizeCustom(object):
    """Normalise live TLS data with dimesnions t, c, y, x

    Args:
        bf_quantiles: list
            lower and upper quantiles for rescaling brightfield images (channel 0)
            Performed for each image individually
        bra_quantiles: list
            lower and upper quantiles for rescaling brachyury images (channel 1)
            rescaled across the timelapse
    """

    def __init__(self, bf_quantiles, bra_quantiles):
        self.bf_quantiles = bf_quantiles
        self.bra_quantiles = bra_quantiles

    def __call__(self, sample):
        return complex_normalisation(sample,self.bf_quantiles,self.bra_quantiles)

class SelectRandomTimepoint(object):
    """select a random timepoint form the time series

    time_dimension: int
        dimension index of time 
    """

    def __init__(self, time_dimension):
        self.td = time_dimension

    def __call__(self, sample):
        shape = sample.shape
        random_tp = np.random.randint(0,shape[self.td])
        
        slice_objects = [
            random_tp if i == self.td else slice(0,shape[i]) for i in range(len(shape))
        ]
        return sample[slice_objects]

class SelectRandomTPNumpy(object):
    """select a random timepoint form the time series

    time_dimension: int
        dimension index of time 
    """

    def __init__(self, time_dimension):
        self.td = time_dimension

    def __call__(self, sample):
        shape = sample.shape
        random_tp = np.random.randint(0,shape[self.td])
        
        out = np.take(sample,[random_tp],axis=self.td).squeeze(self.td)
        # print(out.shape)
        return out

class CustomToTensor(object):
    """Custom ToTensor: works with any shape and does not normalisation
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        return from_numpy(sample)

class CustomCropCentroid(object):
    def __init__(self,intensity_channel, channel_dim,crop_size):
        self.intensity_channel = intensity_channel
        self.channel_dim = channel_dim
        self.crop_size = crop_size

    def __call__(self, sample):
        #shape = sample.shape
        intensity_image = np.take(sample,[self.intensity_channel],axis=self.channel_dim).squeeze(self.channel_dim)
        cent = centroid(intensity_image)[-2:]

        cropped = crop_around_centroid_2D(sample,cent,self.crop_size,self.crop_size)

        return cropped

def crop_around_centroid_2D(image, centroid, crop_height = 800, crop_width = 800):
    half_wid = int(crop_width//2)
    half_hgt = int(crop_height//2)
    c_0, c_1 = [int(c) for c in centroid]
    
    if c_0-half_wid < 0:
        x_borders = np.amax(np.array([
            [c_0-half_wid,0],
            [c_0+half_wid,crop_width]]),axis = 1)
    else:
        x_borders = np.amin(np.array([
            [c_0-half_wid,image.shape[0]-crop_width],
            [c_0+half_wid,image.shape[0]]]),axis = 1)
    if c_1-half_wid < 0:
        y_borders = np.amax(np.array([
            [c_1-half_hgt,0],
            [c_1+half_hgt,crop_height]]),axis = 1)
    else:
        y_borders = np.amin(np.array([
            [c_1-half_hgt,image.shape[1]-crop_height],
            [c_1+half_hgt,image.shape[1]]]),axis = 1)

    cropped_img = np.take(image,np.arange(y_borders[0],y_borders[1],1),axis=-2)
    cropped_img = np.take(cropped_img,np.arange(x_borders[0],x_borders[1],1),axis=-1)
    return cropped_img

