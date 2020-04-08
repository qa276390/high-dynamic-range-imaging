import numpy as np
import cv2
import random
import time


def expand_integer_grid(arr, n_classes):
    """
    Parameters
    ----------
    arr: 
        N dim array of size i_1, ..., i_N
    n_classes: 
        C

    Returns
    -------
    one hot: ndarray
        one-hot N+1 dim array of size i_1, ..., i_N, C
    

    """
    one_hot = np.zeros(arr.shape + (n_classes,))
    axes_ranges = [range(arr.shape[i]) for i in range(arr.ndim)]
    flat_grids = [_.ravel() for _ in np.meshgrid(*axes_ranges, indexing='ij')]
    one_hot[flat_grids + [arr.ravel()]] = 1
    assert((one_hot.sum(-1) == 1).all())
    assert(np.allclose(np.argmax(one_hot, -1), arr))
    return one_hot

def linearWeight(pixel_value):
    """ Linear weighting function based on pixel intensity that reduces the
    weight of pixel values that are near saturation.

    Parameters
    ----------
    pixel_value : np.uint8
        A pixel intensity value from 0 to 255

    Returns
    -------
    weight : np.float64
        The weight corresponding to the input pixel intensity

    """
    z_min, z_max = 0., 255.
    if pixel_value <= (z_min + z_max) / 2:
        return pixel_value - z_min
    return z_max - pixel_value


def sampleIntensities(images):
    """Randomly sample pixel intensities from the exposure stack.

    Parameters
    ----------
    images : list<numpy.ndarray>
        A list containing a stack of single-channel (i.e., grayscale)
        layers of an HDR exposure stack

    Returns
    -------
    intensity_values : numpy.array, dtype=np.uint8
        An array containing a uniformly sampled intensity value from each
        exposure layer (shape = num_intensities x num_images)

    """
    z_min, z_max = 0, 255
    num_intensities = z_max - z_min + 1
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    # Find the middle image to use as the source for pixel intensity locations
    mid_img = images[num_images // 2]

    for i in range(z_min, z_max + 1):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(num_images):
                intensity_values[i, j] = images[j][rows[idx], cols[idx]]
    return intensity_values


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    """Find the camera response curve for a single color channel

    Parameters
    ----------
    intensity_samples : numpy.ndarray
        Stack of single channel input values (n x P)

    log_exposures : numpy.ndarray
        Log exposure times (size == P)

    smoothing_lambda : float
        A constant value used to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.

    weighting_function : callable
        Function that computes a weight from a pixel intensity

    Returns
    -------
    numpy.ndarray, dtype=np.float64
        Return a vector g(z) where the element at index i is the log exposure
        of a pixel with intensity value z = i (e.g., g[0] is the log exposure
        of z=0, g[1] is the log exposure of z=1, etc.)
    """
    z_min, z_max = 0, 255
    INTENSITY_RANGE = 256  # difference between min and max possible pixel value for uint8
    n = intensity_samples.shape[0] # Numbers of pixels we picked
    P = len(log_exposures) # Numbers of images we take in different exposures

    # NxP + [(Zmax-1) - (Zmin + 1)] + 1 constraints; N + 256 columns
    mat_A = np.zeros((P * n + INTENSITY_RANGE - 2 + 1, INTENSITY_RANGE + n), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    # 1. Add data-fitting constraints:
    k = 0
    for i in range(n):
        for j in range(P):
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij)
            mat_A[k, z_ij] = w_ij
            mat_A[k, (INTENSITY_RANGE) + i] = -w_ij
            mat_b[k, 0] = w_ij * log_exposures[j]
            k += 1
    
    # 2. Add color curve centering constraint g(127)=0:
    mat_A[k, (z_max - z_min) // 2] = 1
    k += 1

    # 3. Add smoothing constraints:
    for z_k in range(z_min + 1, z_max):
        w_k = weighting_function(z_k)
        mat_A[k, z_k - 1] = w_k * smoothing_lambda
        mat_A[k, z_k    ] = -2 * w_k * smoothing_lambda
        mat_A[k, z_k + 1] = w_k * smoothing_lambda
        k += 1



    inv_A = np.linalg.pinv(mat_A)
    x = np.dot(inv_A, mat_b)

    g = x[0: INTENSITY_RANGE]
    return g[:, 0]


def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):
    """Calculate a radiance map for each pixel from the response curve.

    Parameters
    ----------
    images : list
        Collection containing a single color layer (i.e., grayscale)
        from each image in the exposure stack. (size == num_images)

    log_exposure_times : numpy.ndarray
        Array containing the log exposure times for each image in the
        exposure stack (size == num_images)

    response_curve : numpy.ndarray
        Least-squares fitted log exposure of each pixel value z

    weighting_function : callable
        Function that computes the weights

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        The image radiance map (in log space)
    """
    #rc_dict = dict( (k, v) for (k, v) in enumerate(response_curve))

    zmid = (255-0)/2
    imgarr = np.asarray(images) #(P photos, Length, Width)
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)

    num_images = len(images)
    W = zmid - np.absolute(imgarr-zmid)
    G = np.dot(expand_integer_grid(imgarr, 256),response_curve)
    
    lndT = log_exposure_times[:, np.newaxis, np.newaxis] * np.ones(imgarr.shape)
    img_rad_map[:, :] = np.sum((W * (G - lndT)), axis=0) / (np.sum(W, axis=0) + 1e-8)

    return img_rad_map

def globalToneMapping(image, Lwhite, alpha):
    """Global tone mapping using gamma correction
    ----------
    images : <numpy.ndarray>
        Image needed to be corrected
    Lwhite : floating number
        The number for constraint the highest value in hdr image
    alpha : floating number
        The number for correction. Higher value for brighter result; lower for darker
    Returns
    -------
    numpy.ndarray
        The resulting image after gamma correction
    """
    delta = 1e-9
    N = image.shape[0]*image.shape[1]*3
    #Lw = np.exp(image)
    Lw = image
    Lb = np.exp((1/N)*np.sum(np.log(delta+Lw)))
    Lm = (alpha/Lb)*Lw # linear 
    Ldf = Lm * (1 + (Lm / (Lwhite**2))) / (1 + Lm)
    image_corrected = Ldf
    image_corrected = ((image_corrected - image_corrected.min())*(255/(image_corrected.max()-image_corrected.min())))
    return image_corrected




def computeHDR(images, log_exposure_times, smoothing_lambda=100.):
    """Computational pipeline to produce the HDR images
    ----------
    images : list<numpy.ndarray>
        A list containing an exposure stack of images
    log_exposure_times : numpy.ndarray
        The log exposure times for each image in the exposure stack (in nature log)
    smoothing_lambda : np.int (Optional)
        A constant value to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.
    Returns
    -------
    numpy.ndarray
        The resulting HDR with intensities scaled to fit uint8 range
    """

    num_channels = images[0].shape[2]
    hdr_image = np.zeros(images[0].shape, dtype=np.float64)

    for channel in range(num_channels):
        print('channel:', channel)
        # Collect the current layer of each input image from the exposure stack
        layer_stack = [img[:, :, channel] for img in images]

        # Sample image intensities
        print('    sampling intensities...')
        otime = time.time()
        intensity_samples = sampleIntensities(layer_stack)
        print('    running time:', time.time()-otime)
        # Compute Response Curve
        print('    computing response curve...')
        otime = time.time()
        response_curve = computeResponseCurve(intensity_samples, log_exposure_times, smoothing_lambda, linearWeight)
        print('    running time:', time.time()-otime)
        
        # Build radiance map
        print('    building irradiance map using np...')
        otime = time.time()
        img_rad_map = computeRadianceMap(layer_stack, log_exposure_times, response_curve, linearWeight)
        print('    running time:', time.time()-otime)
        hdr_image[..., channel]  = img_rad_map

    return np.exp(hdr_image)
