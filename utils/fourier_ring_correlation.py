# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:54:20 2017

@author: Till, till.dreier@med.lu.se

Adjusted from S Sajid Ali: https://github.com/s-sajid-ali/FRC
( which is based on the MATLAB code by Michael Wojcik )

With inspiration from the Toupy package: https://toupy.readthedocs.io/en/stable/_modules/toupy/resolution/FSC.html#FSCPlot

Adjusted to match with: https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-19-22-21333&id=223190

M. van Heela, and M. Schatzb, "Fourier shell correlation threshold
criteria," Journal of Structural Biology 151, 250-262 (2005)
"""

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt




def __spin_average__(x):
    """
    Spin average of an input image.
    :param x:       Image as numpy array, has to be square.
    :return:        Sum of all unique elements per position as numpy array.
    """

    # Depending on the dimension of the image 2D/3D, create an array of integers
    # which increase with distance from the center of the array
    if np.size(np.shape(x)) == 2:
        nr, nc = np.shape(x)
        nrdc = np.floor(nr / 2) + 1
        ncdc = np.floor(nc / 2) + 1
        r = np.arange(nr) - nrdc + 1
        c = np.arange(nc) - ncdc + 1
        [R, C] = np.meshgrid(r, c)
        index = np.round(np.sqrt(R ** 2 + C ** 2)) + 1

    elif np.size(np.shape(x)) == 3:
        nr, nc, nz = np.shape(x)
        nrdc = np.floor(nr / 2) + 1
        ncdc = np.floor(nc / 2) + 1
        nzdc = np.floor(nz / 2) + 1
        r = np.arange(nr) - nrdc + 1
        c = np.arange(nc) - ncdc + 1
        z = np.arange(nc) - nzdc + 1
        [R, C, Z] = np.meshgrid(r, c, z)
        index = np.round(np.sqrt(R ** 2 + C ** 2 + Z ** 2)) + 1

    else:
        print('input is neither a 2d or 3d array')

    # The index array has integers from 1 to maxindex arranged according to distance from the center
    maxindex = np.max(index)
    output = np.zeros(int(maxindex), dtype=complex)

    # In the next step the output is generated. The output is an array of length
    # maxindex. The elements in this array correspond to the sum of all the elements
    # in the original array corresponding to the integer position of the output array
    # divided by the number of elements in the index array with the same value as the
    # integer position.

    # Depening on the size of the input array, use either the pixel or index method.
    # By-pixel method for large arrays and by-index method for smaller ones.
    if nr >= 512:
        # print('performed by pixel method')
        sumf = np.zeros(int(maxindex), dtype=complex)
        count = np.zeros(int(maxindex), dtype=complex)
        for ri in range(nr):
            for ci in range(nc):
                sumf[int(index[ri, ci]) - 1] = sumf[int(index[ri, ci]) - 1] + x[ri, ci]
                count[int(index[ri, ci]) - 1] = count[int(index[ri, ci]) - 1] + 1
        output = sumf / count
        return output

    else:
        # print('performed by index method')
        indices = []
        for i in np.arange(int(maxindex)):
            indices.append(np.where(index == i + 1))
        for i in np.arange(int(maxindex)):
            output[i] = sum(x[indices[i]]) / len(indices[i][0])
        return output


def match_shape(a, shape, padding='constant'):
    """
    Add padding around an array to match a provided shape.
    :param a:           Array to pad.
    :param shape:       Desired array shape, tuple or list with 2 integers.
    :param padding:     Padding mode. Default 'constant', which will add zeros.
    :return:            Padded array.
    """
    y_, x_ = shape
    y, x = a.shape
    if x > x_ or y > y_:
        print('[ERROR:] input array shape is smaller than desired shape.')
        return False
    else:
        y_pad = (y_-y)
        x_pad = (x_-x)
        padded = np.pad(a, ((y_pad//2, y_pad//2 + y_pad % 2), (x_pad//2, x_pad//2 + x_pad % 2)), mode=padding)
        return padded


def smooth(y, n_point_avg, fix_start_values=True):
    """
    Smoothing a curve by convolution with n_point_avg moving average.
    :param y:                   Data to smooth as 1D numpy array.
    :param n_point_avg:         Size of the moving average used to smooth the data.
    :param fix_start_values:    When True (default), correct start values.
    :return:                    Smoothed data as numpy array, same shape as input.
    """
    box = np.ones(n_point_avg)/n_point_avg
    y_smooth = np.convolve(y, box, mode='same')
    if fix_start_values:
        y_smooth[:int(np.floor(n_point_avg / 2))] = np.amax(y_smooth)
    return y_smooth


def find_intersect(smoothed, thl, offset=1):
    """
    Find x-coordinates where smoothed and thl intersect.
    :param smoothed:    Smoothed FRC curve as 1D numpy array.
    :param thl:         Threshold as 1D numpy array, same shape as smoothed.
    :param offset:      Start with an offset, points at coordinate 0 might create a false overlap.
    :return:            X-coordinates where curves intersect, corrected with offset.
    """
    idx = np.argwhere(np.diff(np.sign(smoothed[offset:] - thl[offset:]))).flatten()
    if len(idx) == 0:
        print("[WARNING:] Curves don't intersect!")
    return idx + offset


def calculate_resolution_limit(thl, p_eff, intersect, linspacemax=1.5):
    """
    Calculate the resolution limit using a selected intersection coordinate.
    Select the first intersection (seen from left to right).
    Assuming a spatial frequency / nyquist represents a resolution of p_eff.
    :param thl:         Threshold curve as 1D numpy array.
    :param p_eff:       Effective pixel size.
    :param intersect:   X-coordinate of a chosen intersection as int.
    :return:            resolution limit in same unit as p_eff and in pixels.
    """
    x_thl = np.linspace(0, linspacemax, len(thl))  # FIXME: figure out what to use here ....
    res_limit = p_eff / x_thl[intersect]
    return res_limit, res_limit / p_eff


def frc(img_1, img_2, thl_criterion='1bit'):
    """
    Fourier Ring Correlation of 2 square images of the same shape.
    :param img_1:           First input image as numpy array. Must be square.
    :param img_2:           Second input image as numpy array. Must be square.
    :param thl_criterion:   Threshold criterion (SNRt). Provide snrt or '1bit' or 'halfbit'.
                            0.5 is 1-bit threshold (default).
                            0.2071 is 1/2-bit threshold.
    :return:                Correlation curve.
    """
    # check that images are square
    if np.shape(img_1) != np.shape(img_2):
        print('input images must have the same dimensions')
    if np.shape(img_1)[0] != np.shape(img_1)[1]:
        print('input images must be squares')
    if type(thl_criterion) is str:
        if thl_criterion in ['1bit', 'onebit', '1-bit']:
            snrt = 0.5
        elif thl_criterion in ['halfbit', '1/2bit', '1/2-bit']:
            snrt = 0.2071
    elif type(thl_criterion) in [int, float]:
        snrt = thl_criterion
    else:
        print('[ERROR:] invalid threshold criterion.')

    # Fourier transform
    I1 = fft.fftshift(fft.fft2(img_1))
    I2 = fft.fftshift(fft.fft2(img_2))

    # spin average
    C = __spin_average__(np.multiply(I1, np.conj(I2)))
    C1 = __spin_average__(np.multiply(I1, np.conj(I1)))
    C2 = __spin_average__(np.multiply(I2, np.conj(I2)))

    # Fourier Ring Correlation
    corr = abs(C) / np.sqrt(abs(np.multiply(C1, C2)))

    # thl curve
    n = 2 * np.pi * np.arange(len(corr))
    n[0] = 1
    eps = np.finfo(float).eps  # precision of float (difference between 1.0 and next smallest number, should be around 2.22e-16)
    t1 = np.divide(np.ones(np.shape(n)), n + eps)
    t2 = snrt + 2 * np.sqrt(snrt) * t1 + np.divide(np.ones(np.shape(n)), np.sqrt(n))
    t3 = snrt + 2 * np.sqrt(snrt) * t1 + 1
    thl = np.divide(t2, t3)

    return corr, thl


def plot_frc(corr, smoothed, thl, intersect, p_eff, p_unit='µm', thl_label='1-bit threshold', label='SR vs HR', filename_prefix='frc_plot'):
    """
    Plot the FRC curve.
    :param corr:        FRC curve as 1D numpy array.
    :param smoothed:    Smoothed FRC curve as 1D numpy array.
    :param thl:         THL curve as 1D numpy array.
    :param intersect:   X-coordinate where smoothed and thl intersect as int.
    :param p_eff:       Effective pixel size as float.
    :param p_unit:      Unit of p_eff as str. Default: 'µm'.
    :param thl_label:   Label of the THL curve as str. Default: '1-bit threshold'.
    :return:
    """
    plt.rcParams.update({'font.family': 'Times'})

    plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(0, 2, len(corr)), corr, label=label)
    plt.plot(np.linspace(0, 2, len(smoothed)), smoothed, 'r--', label='smoothed curve')
    x_thl = np.linspace(0, 2, len(thl))
    res_limit, res_limit_pix = calculate_resolution_limit(thl, p_eff, intersect)
    plt.plot(x_thl, thl, '--', label='{}'.format(thl_label))
    # plt.plot(x_thl[idx+1], thl[idx+1], 'go', label='intersect')
    plt.axvline(x_thl[intersect], linestyle='--', color='black', label="Resolution limit")
    plt.text(x_thl[intersect] + 0.02, thl[intersect] + 0.02, "{:.2f} {} ({:.2f} pixels)".format(res_limit, p_unit, res_limit_pix))
    plt.xlim(0, 1)
    plt.legend()
    plt.grid()
    plt.title("Fourier Ring Correlation", fontsize=18)
    plt.xlabel('Spatial Frequency / Nyquist', fontsize=14)
    plt.ylabel("Correlation", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # save as pdf
    plt.savefig(f'figures/{filename_prefix}.pdf', dpi=600, bbox_inches='tight')



if __name__ == "__main__":
    from PIL import Image
    from scipy import ndimage
    example_path = "C:/Users/aulho/OneDrive - Danmarks Tekniske Universitet/Billeder/august.jpg"

    img1 = Image.open(example_path).convert('L')  # Convert to grayscale

    # Downscale image to simulate a second image
    img2 = ndimage.gaussian_filter(np.array(img1), sigma=1)  # Example image 2
    img2 = ndimage.zoom(img2, 0.5)  # Example image 1

    # upscale img2 to match img1 size using nearest neighbor interpolation
    img2 = ndimage.zoom(img2, np.array(img1.size) / np.array(img2.shape), order=0)

    plt.figure()
    plt.imshow(img1, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.imshow(img2, cmap='gray')
    plt.title("Filtered Image")
    plt.axis('off')
    plt.show()

    p_eff = 2  # Effective pixel size in micrometers

    corr, thl = frc(img1, img2, thl_criterion='1bit')
    print("FRC shape: ", corr.shape)
    print("THL shape: ", thl.shape)

    smoothed = smooth(corr, 5)
    print("Smoothed FRC shape: ", smoothed.shape)

    intersect = find_intersect(smoothed, thl)
    print("Intersection index: ", intersect)

    plot_frc(corr, smoothed, thl, intersect[0], p_eff, p_unit='µm', thl_label='1-bit threshold')
