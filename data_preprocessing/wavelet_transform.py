"""
This is a script to perform a wavelet transformation on several predictors

Version = 1.0
Author = drachti (drachti@bgc-jena.mpg.de)
Credit = Max-Planck-Institute for Biogeochemistry
"""


import numpy as np
import pywt


class WaveletTransform:
    """
    Model class to predict GCC and RCC

    Args:
        model (torch.nn.Module): Pretrained and modified resnet model
        scale_size (int): Number of scales for the wavelet transform
    """

    def __init__(self, scale_size: int) -> np.array:
        self.scale_size = scale_size


    def gen_log_space(self, limit: int, n: int) -> np.array:
        """
        Generates logarithmic spaced integer array for the wavelet scales
        """
        result = [1]
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
        while len(result) < n:
            next_value = result[-1] * ratio
            if next_value - result[-1] >= 1:
                result.append(next_value)
            else:
                result.append(result[-1] + 1)
                # recalculate the ratio so that the remaining values will scale correctly
                ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
        # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
        return np.array(list(map(lambda x: round(x) - 1, result)), dtype=np.uint64)

    def wavelet_transform(self, data: np.array, scale_size: int) -> np.array:
        """
        Wavelet transforming the input data
        """
        if scale_size == 52:
            scales = np.arange(np.log10(0.5), np.log10(180), 0.05)
            scales_logsp = 10 ** (scales)

        else:
            scales = self.gen_log_space(731, scale_size + 2)
            scales_logsp = scales[2:] / 4


        wt_data = pywt.cwt(data, scales_logsp, "mexh")
        wt_data = wt_data[0]

        return wt_data

    def __call__(self, data: np.array) -> np.array:
        """
        Model function: Predicting GCC and RCC with meteorological
        and static data
        """
        features = data
        meteo_input = self.wavelet_transform(features, self.scale_size)
        
        return meteo_input


if __name__ == "__main__":
    pass