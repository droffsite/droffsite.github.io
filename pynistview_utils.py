"""
pynistview_utils contains utility functions for processing data captured
by SEMPA. It is also used by the pyNISTview application.
"""

import imutils
import glob
import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d, find_peaks
from scipy.optimize import least_squares

from netCDF4 import Dataset

from colorspacious import cspace_convert

import cv2

from selelems import circle_array, diamond_array

sempa_file_suffix = 'sempa'
scale_multiplier = 10e8


def add_colorbar(img, ax, label='', cmap='gray'):
    # Add a colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(img.min(),
                                                         img.max()))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label(label)


def align_and_scale(intensity_1, intensity_2, m_1, m_2, m_3, m_4,
                    features=512, match_percent=0.15, filter=0.78):
    im_1 = rescale_to_8_bit(intensity_1)
    im_2 = rescale_to_8_bit(intensity_2)
    im_x = rescale_to_8_bit(m_1)
    im_y = rescale_to_8_bit(m_2)

    # im_1_filtered = np.where(im_1 > int(filter * np.max(im_1)), im_1, 0)
    # im_2_filtered = np.where(im_2 > int(filter * np.max(im_2)), im_2, 0)

    im_diff = im_2 - im_1

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(features)

    keypoints1, descriptors1 = orb.detectAndCompute(im_1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im_2, None)

    # Match features.
    # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches = list(matches)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = np.max([int(len(matches) * match_percent), 4])
    print(f'Found {len(matches)} matches; keeping {num_good_matches}.')
    matches = matches[:num_good_matches]

    # Draw top matches
    im_matches = cv2.drawMatches(im_1, keypoints1, im_2, keypoints2, matches,
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    im_keypoints = cv2.drawKeypoints(im_1, keypoints1, np.array([]))

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # print(f'{points1}\n{points2}')

    # lk_params = {'winSize': (19, 19), 'maxLevel': 2, 'criteria': (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03)};

    # p2, status, error = cv2.calcOpticalFlowPyrLK(im_1, im_2, points1, points2, **lk_params)
    # print(p2, status, error)
    # Find homography
    h, mask = cv2.findHomography(
        points1, points2, cv2.RANSAC, ransacReprojThreshold=2.0)
    # a_t = cv2.getAffineTransform(points1[:3], points2[:3])
    # h = cv2.warpAffine(points1, points2)
    # h, mask = cv2.findHomography(points1, np.float32(p2), cv2.RANSAC)
    # h = np.array([[9.91469067e-01, -3.06551887e-04,  0],
    #               [-1.66912051e-02,  9.97772576e-01,  0],
    #               [-8.31626757e-05,  1.28397011e-04,  1.00000000e+00]])

    # Use homography
    height, width = im_1.shape
    intensity_1_h = rescale_to(cv2.warpPerspective(im_1, h, (width, height)),
                               intensity_1)
    # intensity_1_h = rescale_to(cv2.warpAffine(
    #     im_1, a_t, (width, height)), im_1)
    m_1_h = rescale_to(cv2.warpPerspective(im_x, h, (width, height)), m_1)
    m_2_h = rescale_to(cv2.warpPerspective(im_y, h, (width, height)), m_2)
    plt.imshow(intensity_1_h, cmap='gray')

    # Figure out translation of homographies
#     first_x_y = (h[0, 2]).astype(int)
#     last_y_x = (np.where(
#         intensity_1_h[-1, h[0, 2].astype(int):] > min(intensity_1_h[-1]))[0][
#                     0] + h[0, 2]).astype(int)

    # Find the contour of the image
    cnts = cv2.findContours(rescale_to_8_bit(intensity_1_h), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    # print(f'Found {len(cnts)} contours with shape{np.asarray([0]).shape}:\n{cnts}')
    cnts = np.squeeze(cnts[0])
    # print(cnts)

    # Find the center point of the contour
    t_x, t_y = np.mean(cnts[:, 0]), np.mean(cnts[:, 1])
    # print(f'Found image center: ({t_x:.1f}, {t_y:.1f})')

    # Now measure the distance to each point in the contour from the center for quadrants
    # 1 through 4 (top right, top left, bottom left, bottom right, respectively)
    # q2 = np.squeeze([point for point in cnts if point[0] < t_x and point[1] < t_y])
    q1 = [point for point in cnts if point[0] > t_x and point[1] < t_y]
    q2 = [point for point in cnts if point[0] < t_x and point[1] < t_y]
    q3 = [point for point in cnts if point[0] < t_x and point[1] > t_y]
    q4 = [point for point in cnts if point[0] > t_x and point[1] > t_y]

    dists_q1 = [np.sqrt((pt[0] - t_x) ** 2 + (pt[1] - t_y) ** 2) for pt in q1]
    dists_q2 = [np.sqrt((pt[0] - t_x) ** 2 + (pt[1] - t_y) ** 2) for pt in q2]
    dists_q3 = [np.sqrt((pt[0] - t_x) ** 2 + (pt[1] - t_y) ** 2) for pt in q3]
    dists_q4 = [np.sqrt((pt[0] - t_x) ** 2 + (pt[1] - t_y) ** 2) for pt in q4]

    # The top left and bottom left points will be the points in the quadrants with the greatest
    # distances.
    top_left, bottom_left = q2[np.argmax(dists_q2)], q3[np.argmax(dists_q3)]
    top_right, bottom_right = q1[np.argmax(dists_q1)], q4[np.argmax(dists_q4)]
    print(
        f'Top left: {top_left}, Top right: {top_right}\nBottom left: {bottom_left}, Bottom right: {bottom_right}')

    # x_min = np.max((np.min(bottom_left[0]), np.max(top_left[0])))
    # y_min = top_left[1]
    # y_max = bottom_left[1]
    x_min, x_max = np.max((top_left[0], bottom_left[0])), np.min(
        (top_right[0], bottom_right[0]))
    y_min, y_max = np.max((top_left[1], top_right[1])), np.min(
        (bottom_right[1], bottom_right[1]))
    print(f'Cropping ({x_min}, {y_min}) to ({x_max}, {y_max})')

    # Crop the original images so they all cover the same area
    results_r = [img[y_min:y_max, x_min:x_max] for img in
                 (intensity_1_h, intensity_2, m_1_h, m_2_h, m_3, m_4)]
    results_h = (intensity_1_h, m_1_h, m_2_h)

    return results_r, results_h, h, im_matches, im_keypoints


def calculate_distance(x1, x2):
    """Calculate the distance between two points"""
    return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)


def clean_image(image, sigma=50, h=25):
    """Remove gradient and noise from image."""
    # Rescale to 0..255 for filters
    image_shifted = np.asarray(rescale_to_8_bit(image))

    image_denoised = cv2.fastNlMeansDenoising(np.asarray(
        rescale(image_shifted, 0, 255), dtype=np.uint8), None, h, 7, 21)
    image_denoised_blurred = gaussian_filter(image_denoised, sigma)

    return rescale_to(image_denoised, image), rescale_to(image_denoised_blurred, image)


def ciecam02_cmap():
    """Create a perceptually uniform colormap based on CIECAM02."""
    # Based on https://stackoverflow.com/questions/23712207/cyclic-colormap-without-visual-distortions-for-use-in-phase-angle-plots

    # first draw a circle in the cylindrical JCh color space.
    # First channel is lightness, second chroma, third hue in degrees.
    color_circle = np.ones((256, 3)) * 60
    color_circle[:, 1] = np.ones((256)) * 45
    color_circle[:, 2] = np.arange(0, 360, 360 / 256)
    color_circle_rgb = cspace_convert(color_circle, 'JCh', 'sRGB1')

    return mpl.colors.ListedColormap(color_circle_rgb)


def ciecam02_cmap_r():
    """Create a perceptually uniform colormap based on CIECAM02."""
    # Based on https://stackoverflow.com/questions/23712207/cyclic-colormap-without-visual-distortions-for-use-in-phase-angle-plots

    # first draw a circle in the cylindrical JCh color space.
    # First channel is lightness, second chroma, third hue in degrees.
    color_circle = np.ones((256, 3)) * 60
    color_circle[:, 1] = np.ones((256)) * 45
    color_circle[:, 2] = np.arange(360, 0, 360 / 256)
    color_circle_rgb = cspace_convert(color_circle, 'JCh', 'sRGB1')

    return mpl.colors.ListedColormap(color_circle_rgb)

def file_for(files, token):
    """Convenience method for extracting file locations"""

    return files[[i for i, item in enumerate(files) if token in item]][0]


def find_circular_contours(img, diff_axes_threshold=0.3, comparator='gt', show=False):
    """Return an array of circular contours found in the input image. Also return the full array of contours."""
    circle_contours = []

    # Work on the values above (default) or below the mean
    threshold = img.mean()
    # threshold = np.pi / 2
    ups = np.where(img > threshold, 255, 0) if comparator == 'gt' else \
        np.where(img < threshold, 255, 0)
    # ups = np.logical_and(0.9 * np.pi / 2 <= img, img <= 1.1 * np.pi / 2)

    # OpenCV requires unsigned ints
    ups_b = ups.astype(np.uint8)
    kernel = circle_array(3)
    ups = cv2.morphologyEx(ups_b, cv2.MORPH_CLOSE, kernel)
    xdim, ydim = ups.shape[1], ups.shape[0]

    if show:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(ups_b, cmap='gray')
        plt.subplot(122)
        plt.imshow(ups, cmap='gray')

    # Find all contours
    contours, _ = cv2.findContours(ups, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print(np.asarray(contours).shape)

    # Remove the contours that are on the edges
    contours_g = [contour for contour in contours
                  if 0 not in contour and xdim - 1 not in contour[:, :, 0]
                  and ydim - 1 not in contour[:, :, 1]
                  and len(np.squeeze(contour).shape) == 2]

    for contour in [c for c in contours_g if len(c) > 4]:
        # Fit the contour to an ellipse. This requires there be at least 5 points in the contour
        # (thus the if). Throw out anything where the major and minor axes are different by more
        # than the threshold
        width, height = cv2.fitEllipse(contour)[1]
        diff = np.abs(width - height)

        if diff < np.max([width, height]) * diff_axes_threshold:
            circle_contours.append(contour)

    return circle_contours[::-1], contours_g[::-1], contours[::-1]


def find_com(img_1, img_2, img_3=None):
    """Use centroids to find center of mass."""
    # Get the image dimensions
    dimx, dimy = img_1.shape[1], img_1.shape[0]

    # Calculate the 2D histogram to use for finding the CoM
    hist, _, _ = np.histogram2d(img_1.ravel(), img_2.ravel(), bins=dimx)

    # Get position in histogram. This will be coordinates.
    x_cm = np.average(range(1, dimx + 1),
                      weights=[np.sum(hist[:, x]) for x in range(dimx)])
    y_cm = np.average(range(1, dimy + 1),
                      weights=[np.sum(hist[y, :]) for y in range(dimy)])

    # Convert to original range
    x_cm = x_cm / dimx * np.ptp(img_1) + img_1.min()
    y_cm = y_cm / dimy * np.ptp(img_2) + img_2.min()

    return x_cm, y_cm


def find_group_contours(contours):
    """Find groupings of circular contours."""

    boxes, centers = get_bounding_boxes(contours)

    # Determine total number of contours
    count = len(centers)

    # Create a matrix of distances between contours
    distances = np.asarray([[calculate_distance(centers[i], centers[j])
                             for j in range(count)] for i in range(count)])

    groups = []
    loners = []
    indices = list(range(0, len(boxes)))

    while len(indices) > 0:
        # Take the first index in the list
        index = indices[0]
        neighbors = find_neighbors_for([index], boxes, distances)

        # Examine results and pare down the list of indices
        found = loners if len(neighbors) == 1 else groups

        found.append(neighbors)
        indices = [i for i in indices if i not in list(
            itertools.chain.from_iterable(found))]

    return groups, loners


def find_neighbors_for(indices, boxes, distances, neighbors=[], threshold=2):
    """Find nearest neighbors in to incoming indices."""

    # Add the indices to list of neighbors, eliminating duplicates
    neighbors = np.unique(np.concatenate((neighbors, indices))).astype(int)

    for index in indices:
        box = np.asarray(boxes[index])
        dists = distances[index]

        # Use threshold to determine what's considered close.
        d = np.max(box[2:]) * threshold

        # Find all boxes between 0 exclusive (ignore box being 0 from itself) and d,
        # eliminating duplicates
        new_indices = np.atleast_1d(np.squeeze(
            np.where(np.logical_and(0 < dists, dists <= d))))
        new_indices = [i for i in new_indices if i not in neighbors]

        # print(f'Looking at {index}, found {new_indices} with neighbors {neighbors}')

        # Recursively call self to follow leads
        neighbors = np.unique(np.concatenate((neighbors,
                                              find_neighbors_for(new_indices,
                                                                 boxes, distances, neighbors))))

    return neighbors.tolist()


def find_offsets(img_1, img_2, img_3=None):
    """Minimize the min to max spread of the images to get most consistent magnitude."""
    # Find a reasonable starting point
    # x0 = find_com(img_1, img_2, img_3)

    img3 = img_3 if img_3 is not None else np.zeros_like(img_1)

    x0 = (img_1.mean(), img_2.mean(), img3.mean())
    # bounds=(-4, 4)

    vals = (np.max(abs(img_1)), np.max(abs(img_2)), np.max(abs(img_3)))

    bounds = (np.multiply(-2, vals), np.multiply(2, vals))

    res = least_squares(ptp_magnitudes, x0, jac_magnitudes,
                        bounds=bounds, args=(img_1, img_2, img3))
    # res = least_squares(std_magnitudes, x0, method='trf', args=(img_1, img_2, img_3))

    return res.x, res.status, res.message


def jac_magnitudes(variables, x, y, z):
    X, Y, Z = x - variables[0], y - variables[1], z - variables[2]
    denominator = np.sqrt(X**2 + Y**2 + Z**2)

    return np.ptp(X / denominator), np.ptp(Y / denominator), np.ptp(Z / denominator)


def fit_image(image, mask=None):
    """Return a least-squares fitting of the input image. Allow specifying a mask to fit a subsection of the image."""
    if mask is None:
        mask = np.ones_like(image)

    # Get dimensions
    y_dim, x_dim = image.shape

    # Flatten the mask to eliminate masked off indices
    mask_flat = np.array(mask.flatten(), dtype=bool)

    # The linalg function needs flattened arrays of coordinates and values
    x, y = np.linspace(0, x_dim, x_dim), np.linspace(0, y_dim, y_dim)
    X, Y = np.meshgrid(x, y, copy=False)
    X, Y = X.flatten(), Y.flatten()

    # Set up arrays for linalg function
    A = np.array([X * 0 + 1, X, Y, X ** 2, X ** 2 * Y, X ** 2 * Y ** 2, Y ** 2,
                  X * Y ** 2, X * Y]).T
    B = image.flatten()

    coeff, _, _, _ = np.linalg.lstsq(A[mask_flat], B[mask_flat], rcond=None)

    image_fit = np.sum(coeff * A, axis=1).reshape(image.shape)
    image_fit_masked = image_fit * mask

    return image_fit, image_fit_masked


def get_bounding_boxes(contours):
    """Return the bounding boxes for the passed in contours"""
    boxes = [cv2.boundingRect(contour) for contour in contours]
    centers = [(x + int(x_ext / 2), y + int(y_ext / 2))
               for x, y, x_ext, y_ext in boxes]

    return boxes, centers


def get_file_key(file_path, key):
    """Return a simple string representation of a key in the image file."""

    # Valid values:
    # ['horizontal_min', 'horizontal_max', 'vertical_min', 'vertical_max', 'horizontal_full_scale',
    #  'vertical_full_scale', 'magnification', 'file_path', 'title', 'note', 'sample_id', 'start_time',
    # 'end_time', 'counter_gate_delay', 'counter_gate_width', 'adc_delay', 'period', 'detector',
    #  'channel_name', 'ix_counter_zero', 'mx_counter_zero', 'iy_counter_zero', 'my_counter_zero',
    #  'mx_sherman', 'my_sherman', 'accelerating_voltage', 'probe_current', 'num_overscan_columns',
    #  'scan_index', 'repeat_index', 'num_repeats', 'num_averages', 'horizontal_drift', 'vertical_drift',
    #  'run_uuid', 'image_data']

    file = Dataset(file_path, 'r')

    data = file.variables[key][...].data
    if data.shape != ():
        data = b''.join(data).decode()

    return data


def get_magnitudes(im_x, im_y, im_z=None):
    """Return magnitudes of combined images: low -> dark, high -> light."""
    # Handle 2d case.
    im_z = np.zeros_like(im_x) if im_z is None else im_z

    magnitudes = np.sqrt(im_x ** 2 + im_y ** 2 + im_z ** 2)

    return magnitudes.reshape(im_x.shape)


def get_phases(im_x, im_y, im_z=None):
    """Return phases of combined images in radians."""
    # First determine phi from X and Y normalized
    phis = np.arctan2(im_y / im_y.max(), im_x / im_x.max())

    # arctan2 works over the range -pi to pi; shift everything to 0 to 2pi
    # for color mapping
    phis = [[x if x >= 0 else x + 2 * np.pi for x in row] for row in phis]

    # Now Z. Set to pi/2 for 2D.
    thetas = np.ones_like(im_x) * np.pi / 2

    if im_z is not None:
        # Get the XY magnitudes, then arctan Z over that. Range 0 to pi.
        xy_magnitudes = get_magnitudes(im_x, im_y)
        thetas = -np.arctan2(im_z / im_z.max(),
                             xy_magnitudes / xy_magnitudes.max()) + np.pi/2

    return (np.asarray(phis), thetas)


def get_phi_diff(path, phis):
    """Determine the average angle of magnetization relative to the
       contour it circles."""

    # Determine the angle along the path from point to point. Need to
    # roll to connect end to beginning.
    path_diff = np.roll(path, -2) - path

    # Flip the sign of y for matplotlib (y=0 is at the top)
    path_angle = np.arctan2(-path_diff[:, 1], path_diff[:, 0])

    # Adjust range to 0..2 pi
    path_angle %= (2 * np.pi)

    # Collect the phis along the path and subtract them from the path angles
    phis_path = np.array([phis[point[1], point[0]] for point in path])
    phis_diff = phis_path - path_angle

    # Calculate alpha per JC's definition: deviation from domain wall normal traveling
    # from low M_z to high M_z. Alpha ranges from 0 to 2 pi. The path goes clockwise
    # around the contour. For now, assume the center of the contour is where high M_z
    # is, thus clocwise travel should be pi/2 while anticlockwise should be 3/2 pi.
    alphas = (phis_diff % (2 * np.pi)) + np.pi / 2

    # print(f'\N{greek small letter phi}: {np.min(phis_diff):.2f} to {np.max(phis_diff):.2f}; '\
    #       f'\N{greek small letter alpha}: {np.min(alphas):.2f} to {np.max(alphas):.2f}')

    return np.mean(phis_diff), np.std(phis_diff), np.mean(alphas), np.std(alphas)


def get_scale(file_path):
    """Determine the size of a single pixel in the input image."""
    file = Dataset(file_path, 'r')

    full_scale = np.abs(file.variables['vertical_full_scale'][...].data)
    magnification = np.abs(file.variables['magnification'][...].data)
    dim = file.variables['image_data'].shape[0]

    adjusted_scale = full_scale / magnification / dim

    return adjusted_scale, magnification


def image_data(file_path):
    """Read a file in NetCDF format and return the image data and axis."""
    file = Dataset(file_path, 'r')

    return file.variables['image_data'][...].data.T, str(
        file.variables['channel_name'][...].data[1], 'utf-8')


def import_files(name, runs, indir):
    # Read in image files. Return a dictionary of image data
    # {im1: [], im2: [], m1: [], m2: [], m3: [], m4: [], scale: scale}

    image_dict = {}

    print(f'Searching {name}* for runs {runs}...')

    # Get all the i* and m* files. Should end up with 8 of them.

    files = []

    for run in runs:
        full_name = name + f'{run:0>3}'
        file = indir + full_name

        for f in ['x', 'y', 'z']:
            files.extend(glob.glob(file + '*' + f + '*' + sempa_file_suffix))

    files = np.array(sorted(files), dtype="object")
    fnames = [f.split('/')[-1] for f in files]
    print(f'Found files:\n{fnames}')

    # ix, ix2, iy, iz
    # ix and iy are the same; similarly, ix2 and iz are the same. Ignore 2.
    # The two images will be used to align the data

    i_extensions = ['ix.', 'ix2']

    for i in range(len(i_extensions)):
        key = 'i{}'.format(i + 1)
        im, _ = image_data(file_for(files, i_extensions[i]))
        im_blurred = median_filter(im, 3)

        image_dict[key] = [im, im_blurred, im - im_blurred]

    # mx, mx2, my, mz
    # Stick with 1, 2, 3 for x, y, z; 4 will be the second x for data scaling
    extensions = ['mx.', 'my', 'mz', 'mx2']

    for i in range(len(extensions)):
        key = 'm{}'.format(i + 1)
        m, ax = image_data(file_for(files, extensions[i]))

        image_dict[key] = [m, ax]
        image_dict[key].extend(m.shape)
        image_dict[key].extend([m.min(), m.max()])

    scale, magnification = get_scale(files[0])
    image_dict['scale'] = scale
    image_dict['magnification'] = magnification

    return image_dict


def limit_to(image, sigma=3):
    """Restrict the image range to + or - sigma std deviations"""

    std = np.std(image)
    image_mean = np.mean(image)
    image_min, image_max = image_mean - sigma * std, image_mean + sigma + std

    image = np.where(image < image_min, image_min, image)
    image = np.where(image > image_max, image_max, image)

    return image


def mean_shift_filter(image, sp=1, sr=1):
    """Perform mean shift filtering."""
    # Convert to 8 bit color for the filter
    img = np.asarray(rescale_to_8_bit(image))
    img = cv2.cvtColor(median_filter(img, 5), cv2.COLOR_GRAY2BGR)

    # sp = spatial window, sr = color window
    img_shift = cv2.pyrMeanShiftFiltering(img, sp, sr)

    # Convert back to grayscale
    img_shift = cv2.cvtColor(img_shift, cv2.COLOR_BGR2GRAY)

    return rescale_to(img_shift, image)


def process_ltem_data(ltem_data_path):
    """Return data for .npy formatted LTEM data"""

    zfile = ltem_data_path[-1] == 'z'
    # Data format is array of complex numbers, real = x, imaginary = y
    ltem_data = np.load(ltem_data_path)
    ltem_data_xy = ltem_data if not zfile else (
        ltem_data['Bx'] + 1j * ltem_data['By'])#[::-1, :]
    xerror = 0 if not zfile else ltem_data['x'][1] - ltem_data['x'][0]
    yerror = 0 if not zfile else ltem_data['y'][1] - ltem_data['y'][0]

    ltem_data_z = np.real(
        np.sqrt(np.ones_like(ltem_data_xy) - ltem_data_xy ** 2))

    ltem_phases, _  = get_phases(
        np.real(ltem_data_xy), np.imag(ltem_data_xy))
    
    ltem_magnitudes = get_magnitudes(
        np.real(ltem_data_xy), np.imag(ltem_data_xy))

    ltem_circular_contours, ltem_contours_g, _ = find_circular_contours(
        ltem_magnitudes > np.average(ltem_magnitudes))

    ltem_alphas = np.asarray([get_phi_diff(np.squeeze(c), ltem_phases)
                              for c in ltem_circular_contours])

    return ltem_phases, ltem_magnitudes, ltem_circular_contours, ltem_contours_g, ltem_alphas, xerror, yerror


def ptp_magnitudes(variables, image_1, image_2, image_3=None):
    """Calculate the min to max spreads for magnitudes; used for optimizing the offset."""

    # Allow for 2D processing
    var_3 = variables[2] if image_3 is not None else 0
    arg_3 = image_3 - \
        variables[2] if image_3 is not None else np.zeros_like(image_1)

    return np.ptp(
        get_magnitudes(image_1 - variables[0], image_2 - variables[1], arg_3 - var_3))


def std_magnitudes(variables, image_1, image_2, image_3=None):
    """Calculate the min to max spreads for magnitudes; used for optimizing the offset."""

    # Allow for 2D processing
    var_3 = variables[2] if image_3 is not None else 0
    arg_3 = image_3 - \
        variables[2] if image_3 is not None else np.zeros_like(image_1)

    return np.std(
        get_magnitudes(image_1 - variables[0], image_2 - variables[1], arg_3 - var_3))


def remove_line_errors(image, lines, use_rows=True):
    """Remove line errors using either rows or columns."""
    image_base = image if use_rows else image.T

    image_mean = np.asarray(
        [np.mean(image_base[:lines, x]) for x in range(image.shape[1])]).T

    image_delined = image_base - np.tile(image_mean, (image.shape[0], 1))

    image_delined = image_delined if use_rows else image_delined.T

    return image_delined


def render_phases_and_magnitudes(phases, magnitudes):
    """Adjust the intensity of the contrast image according to phase magnitude."""
    # Set up the colormap
    cmap = ciecam02_cmap()

    # Use CIECAM02 color map, convert to sRGB1 (to facilitate intensity adjustment)
    im = cmap(phases / (2 * np.pi))
    im_srgb = im[:, :, :3]
    im_adjusted = np.zeros_like(im_srgb)

    # Apply phase intensity mask to the contrast: low intensity -> dark, high intensity -> light
    for i in range(3):
        im_adjusted[:, :, i] = np.multiply(
            im_srgb[:, :, i], rescale_to(magnitudes, [0.3, 1]))

    return im_adjusted


def rescale(data, to_min, to_max):
    """Rescale data to have to_min and to_max as min and max, respectively."""
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    factor = to_max - to_min

    return norm_data * factor + to_min


def rescale_to(source, target):
    """Rescale source image to the same range as target."""
    return rescale(source, np.min(target), np.max(target))


def rescale_to_8_bit(data):
    """Rescale data to 8 bit integers"""
    return np.asarray(rescale(data, 0, 255), dtype=np.uint8)


def save_file(file_path, im_image, img, axis_1, axis_2, scale,
              img_denoised_1=None, img_denoised_2=None,
              arrow_color=None, arrow_scale=None, figsize=(3, 3), dpi=300):
    """Save images to PNG files."""

    # First save intensity image
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(im_image, cmap='gray')
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.savefig(file_path + '_' + axis_1 + axis_2 + '_im.png')
    plt.close()

    # Now save magnetization images
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.savefig(file_path + '_' + axis_1 + axis_2 + '.png')

    ax.add_artist(ScaleBar(scale, box_alpha=0.8))
    plt.savefig(file_path + '_' + axis_1 + axis_2 + '_scale.png')

    if img_denoised_1 is not None:
        show_vector_plot(img_denoised_1, img_denoised_2, ax=ax,
                         color=arrow_color, scale=arrow_scale)
        plt.savefig(file_path + '_' + axis_1 + axis_2 + '_scale_arrows.png')

    plt.close()


def save_intensity_images(path, figsize=(3, 3), dpi=300):
    """Render and save all intensity images in path"""
    for file in (glob.iglob(path + '*ix*' + sempa_file_suffix)):
        im_image = image_data(file)

        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(im_image, cmap='gray')
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        plt.savefig(path + file + '_im.png')
        plt.close()


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def segment_image(image, segments=1, adaptive=False, sel=circle_array(1),
                  erode_iter=1, close_iter=7):
    """Segment image into a defined number of objects. Create positive and negative masks for those objects."""
    # Rescale to 0-255 for thresholding
    image_shift = np.asarray(rescale_to_8_bit(image))
    _, image_thresh = cv2.threshold(image_shift, image_shift.mean(),
                                    255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Alternative thresholding mechanism
    # m_1_thresh = cv2.adaptiveThreshold(m_1_shift, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 0)
    # m_1_shift = cv2.cvtColor(m_1_shift, cv2.COLOR_GRAY2BGR)

    # Perform morphological operations. First kill small noise, then close the image
    image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_ERODE, sel,
                                    iterations=erode_iter)
    image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, sel,
                                    iterations=close_iter)

    # Find the segment contours
    snakes = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_NONE)

    # Sort the results by size and keep the requested number
    snakes = imutils.grab_contours(snakes)
    snakes = sorted(snakes, key=cv2.contourArea, reverse=True)[:segments]

    # Handle possibly having fewer than specified number of segments
    found = min(segments, len(snakes))

    # Allocate for segment # of binary masks, positive and negative, each the size of the image
    masks = np.empty((found, 2, image.shape[1], image.shape[0]))

    for i in range(found):
        contour = snakes[i]
        mask = cv2.fillPoly(np.zeros_like(image), [contour], 1)
        masks[i, 0] = mask
        masks[i, 1] = np.ones_like(mask) - mask
        cv2.drawContours(image_shift, [contour], -1, (0, 255, 0), 1)

    # Draw the contours on the shifted input image
    #     for (i, c) in enumerate(snakes):
    #         ((x, y), _) = cv2.minEnclosingCircle(c)
    #         cv2.drawContours(image_shift, [c], -1, (0, 255, 0), 1)

    return image_thresh, image_shift, masks


def show_all_circles(img, contours, ax=None, show_numbers=True, reverse=False, alpha=1, color='black', show_axes=True):
    """Plot all circular contours."""

    if reverse:
        contours = contours[::-1]

    # Convert img to color and draw the contours
    c_img = cv2.cvtColor(np.zeros_like(
        img, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    cv2.drawContours(c_img, contours, -1, (255, 255, 255), 1)

    # Convert resultant image back to grayscale to produce a mask
    c_img_gray = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    masked_circles = np.ma.masked_where(c_img_gray == 0, c_img_gray)

    # Use the mask to produce an RBGA image (transparent background)
    c_img_rgba = cv2.cvtColor(c_img, cv2.COLOR_RGB2RGBA)
    c_img_rgba[:, :, 3] = masked_circles

    boxes, centers = get_bounding_boxes(contours)

    if ax is not None:
        cmap_to_show = cm.binary_r if color == 'black' else cm.binary
        ax.imshow(masked_circles, interpolation='none',
                  alpha=alpha, cmap=cmap_to_show, zorder=100)

        if show_numbers:
            index = 1

            for box in boxes:
                ax.text(centers[index - 1][0], centers[index - 1][1], index,
                        alpha=alpha, color=color,
                        horizontalalignment='center', verticalalignment='center')
                index += 1

        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    return masked_circles, c_img_rgba, boxes


def show_alphas(contours_g, circular_contours, phis, name, show_circles_separately=True, bins=20, ltem_data=None, show_title=True):
    all_alphas = np.asarray([get_phi_diff(np.squeeze(c), phis)
                            for c in contours_g])

    circle_alphas = np.asarray([get_phi_diff(np.squeeze(c), phis)
                                for c in circular_contours])

    # Define color maps
    colors = {'all': 'blue', 'circles': 'orange', 'ltem': 'green'}
    all_hist_type = 'step' if show_circles_separately else 'stepfilled'

    # Create a simulated phi distribution for LTEM if not provided
    p = [0., 0.1, 0.75, 0.1, 0., 0.01, 0.03, 0.01, 0]
    ltem_data = np.random.choice(
        np.arange(0, 2.25 * np.pi, np.pi / 4), len(all_alphas), p=p) \
        if ltem_data is None else ltem_data

    plt.figure(figsize=(10, 5))
    hist_all = plt.hist(all_alphas[:, 2], bins=bins, color=colors['all'],
                        alpha=0.5, histtype=all_hist_type, density=True, label='SEMPA')
    if show_circles_separately:
        hist_circle = plt.hist(circle_alphas[:, 2], bins=bins, color=colors['circles'],
                               alpha=0.5, density=True, label='SEMPA (circular)')
    hist_ltem = plt.hist(ltem_data, bins=bins, color=colors['ltem'], alpha=0.5,
                         density=True, label='LTEM')

    # Find the indices of the maxima for all and LTEM
    lbin_all = np.argmax(hist_all[0])
    lbin_ltem = np.argmax(hist_ltem[0])

    # Circular is a bit more complicated since it seems to have 2 big humps.
    # First, round all values to 4 places, select unique values, and sort
    # descending, keeping the first 2 value (top 2). Then find the indices
    # for those values.
    if show_circles_separately:
        tops_circle = sorted(
            np.unique(np.around(hist_circle[0], 4)), reverse=True)[:2]
        lbin_circle = [np.where(np.around(hist_circle[0], 4) == t)
                       for t in tops_circle]
    else:
        tops_circle = sorted(
            np.unique(np.around(hist_all[0], 0)), reverse=True)[:2]
        lbin_circle = [np.where(np.around(hist_all[0], 0) == t)
                       for t in tops_circle]

    # Now find the modes
    mode_all = (hist_all[1][lbin_all + 1] + hist_all[1][lbin_all]) / 2
    mode_ltem = (hist_ltem[1][lbin_ltem + 1] + hist_ltem[1][lbin_ltem]) / 2

    plt.axvline(mode_all, linestyle='dashed', linewidth=1, color=colors['all'])
    plt.axvline(mode_ltem, linestyle='dashed',
                linewidth=1, color=colors['ltem'])

    use_hist = hist_all if not show_circles_separately else hist_circle
    if show_circles_separately:
        mode_circle = (hist_circle[1][lbin_circle[0][0][-1] + 1] +
                       hist_circle[1][lbin_circle[0][0][0]]) / 2
        mode_circle_2 = (hist_circle[1][lbin_circle[1][0]
                                        [-1] + 1] + hist_circle[1][lbin_circle[1][0][-1]]) / 2
    # mode_circle = (use_hist[1][lbin_circle[0][0][-1] + 1] +
    #                use_hist[1][lbin_circle[0][0][0]]) / 2
    # mode_circle_2 = (use_hist[1][lbin_circle[1][0][-1] + 1] +
    #                  use_hist[1][lbin_circle[1][0][-1]]) / 2

        plt.axvline(mode_circle, linestyle='dashed',
                    linewidth=1, color=colors['circles'])
        plt.axvline(mode_circle_2, linestyle='dashed',
                    linewidth=1, color=colors['circles'])

    # Place text labels for the modes
    _, y_lim = plt.ylim()
    shift_x = 1.05
    t = plt.text(mode_all * shift_x, y_lim * 0.7, f'{mode_all:.2f}')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor=colors['all']))

    if show_circles_separately:
        t = plt.text(mode_circle * shift_x, y_lim * 0.9, f'{mode_circle:.2f}')
        t.set_bbox(dict(facecolor='white', alpha=0.8,
                   edgecolor=colors['circles']))
        t = plt.text(mode_circle_2 * shift_x, y_lim *
                     0.85, f'{mode_circle_2:.2f}')
        t.set_bbox(dict(facecolor='white', alpha=0.8,
                   edgecolor=colors['circles']))

    t = plt.text(mode_ltem * shift_x, y_lim * 0.7, f'{mode_ltem:.2f}')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor=colors['ltem']))

    plt.xlim(0, 2 * np.pi)

    tick_labels = ['0', r'$\dfrac{\pi}{4}$', r'$\dfrac{\pi}{2}$', r'$\dfrac{3\pi}{4}$',
                   r'$\pi$', r'$\dfrac{5\pi}{4}$', r'$\dfrac{3\pi}{2}$', r'$\dfrac{7\pi}{4}$', r'$2\pi$']
    plt.xticks(np.arange(0, 2.25 * np.pi, step=np.pi / 4), tick_labels)
    plt.yticks([])

    plt.xlabel(r'$\alpha$ [Radians]')
    if show_title:
        plt.title(rf'Average $\alpha$ for {name}')

    plt.legend()

    return plt, all_alphas, circle_alphas


def show_alphas_2(alphas_candidates, alphas_all, alphas_ltem, name=r'$\alpha$', show_all=True, show_title=True):
    plt.figure(figsize=((10, 5)))

    c_hist_arr, c_hist_bins, _ = plt.hist(alphas_candidates, alpha=0.7,
                                          density=True, zorder=10, label='Circular')
    if show_all:
        a_hist_arr, a_hist_bins, _ = plt.hist(alphas_all, histtype='step',
                                              density=True, zorder=2, alpha=0.6, label='All SEMPA')
    l_hist_arr, l_hist_bins, _ = plt.hist(alphas_ltem, density=True, zorder=1,
                                          alpha=0.3, label='LTEM')

    c_bin_width = c_hist_bins[1] - c_hist_bins[0]
    c_peak_indices = np.array(find_peaks(np.concatenate(
        ([0], c_hist_arr, [0])), distance=len(c_hist_arr) // 2)[0]) - 1
    c_max_bin_centers = [c_hist_bins[x] +
                         c_bin_width / 2 for x in c_peak_indices]

    if show_all:
        a_bin_width = a_hist_bins[1] - a_hist_bins[0]
        a_peak_indices = np.array(find_peaks(np.concatenate(
            ([0], a_hist_arr, [0])), distance=len(a_hist_arr) // 2)[0]) - 1
        a_max_bin_centers = [a_hist_bins[x] +
                             a_bin_width / 2 for x in a_peak_indices]

    l_bin_width = l_hist_bins[1] - l_hist_bins[0]
    l_peak_indices = np.array(find_peaks(np.concatenate(
        ([0], l_hist_arr, [0])), distance=len(l_hist_arr) // 2)[0]) - 1
    l_max_bin_centers = [l_hist_bins[x] +
                         l_bin_width / 2 for x in l_peak_indices]

    _, y_lim = plt.ylim()
    shift_x_pos = 0
    shift_x_neg = 0.35
    colors = {'circles': 'blue', 'all': 'orange', 'ltem': 'green'}
    c_y_adjust, a_y_adjust, l_y_adjust = 0.87, 0.7, 0.53
    box_alpha = 0.7
    zline, zbox = 20, 30
    new_line = '\n'

    for c in c_max_bin_centers:
        plt.axvline(c, color=colors['circles'],
                    linestyle='dotted', zorder=zline)
        t = plt.text(c + shift_x_pos, y_lim * c_y_adjust,
                     rf'{c:.2f}{new_line}$\pm${c_bin_width / 2:.2f}', zorder=zbox)
        t.set_bbox(dict(facecolor='white', alpha=box_alpha, zorder=20,
                   edgecolor=colors['circles']))

    if show_all:
        for a in a_max_bin_centers:
            plt.axvline(a, color=colors['all'],
                        linestyle='dotted', zorder=zline)
            t = plt.text(a - shift_x_neg, y_lim * a_y_adjust,
                         f'{a:.2f}{new_line}$\pm${a_bin_width / 2:.2f}', zorder=zbox)
            t.set_bbox(dict(facecolor='white', alpha=box_alpha,
                       zorder=20, edgecolor=colors['all']))

    for l in l_max_bin_centers:
        plt.axvline(
            l, color=colors['ltem'] if show_all else colors['all'], linestyle='dotted', zorder=zline)
        t = plt.text(l - shift_x_neg, y_lim * (l_y_adjust if show_all else a_y_adjust),
                     f'{l:.2f}{new_line}$\pm${l_bin_width / 2:.2f}', zorder=zbox)
        t.set_bbox(dict(facecolor='white', alpha=box_alpha, zorder=20,
                   edgecolor=colors['ltem'] if show_all else colors['all']))

    plt.xlabel(r'$\alpha$ [Radians]')
    plt.xlim((0, 2 * np.pi))
    tick_labels = ['0', r'$\dfrac{\pi}{4}$', r'$\dfrac{\pi}{2}$', r'$\dfrac{3\pi}{4}$',
                   r'$\pi$', r'$\dfrac{5\pi}{4}$', r'$\dfrac{3\pi}{2}$', r'$\dfrac{7\pi}{4}$', r'$2\pi$']
    plt.xticks(np.arange(0, 2.25 * np.pi, step=np.pi / 4), tick_labels)
    plt.yticks([])
    plt.legend()

    if show_title:
        plt.title = name


def show_circles(magnitudes, phis, thetas, contours, scale,
                 show_numbers=True, reverse=False, alpha=0.5, show='thetas',
                 normalize=True, show_both=False, phis_m=None,
                 candidate_cutoff=np.pi / 6, show_title=True, show_axes=True,
                 just_candidates=True):
    """Plot all circular contours."""
    masked_circles, _, boxes = show_all_circles(
        thetas, contours, reverse=reverse)
    count = len(boxes)
    print(f'Found {count} boxes.')

    num_columns = 3 if not show_both else 4
    column_width = 8 if not show_both else 6
    total_count = count if not show_both else 2 * count
    # num_rows = np.ceil(total_count / num_columns).astype(int)
    num_rows = 1
    yd, xd = magnitudes.shape

    extent_factor = 0.3
    index = 1
    plot_index = 1
    sm = cm.ScalarMappable(cmap=cm.coolwarm)

    # Track candidates. Cutoff of pi/6 allows for 3 sigma.
    candidates = {}
    all_contours = {}

    # fig = plt.figure(figsize=(num_columns * column_width, num_rows * column_width));
    fig = plt.figure()

    for box in boxes:
        xmin, ymin, width, height = box
        width_nm = int(width * scale * scale_multiplier)
        height_nm = int(height * scale * scale_multiplier)

        # Walk along the path of the contour and measure deviations from it
        path = np.squeeze(contours[index - 1])

        dev_avg, dev_std, alpha_avg, alpha_std = get_phi_diff(path, phis)
        # print(dev_avg, dev_std, alpha_avg, alpha_std)
        candidate = alpha_std <= candidate_cutoff

        all_contours[index] = [dev_avg, dev_std,
                               alpha_avg, alpha_std, width_nm, height_nm]
        if candidate:
            candidates[index] = all_contours[index]

        if candidate or not just_candidates:
            extent_x, extent_y = int(
                extent_factor * width), int(extent_factor * height)
            extent = np.min([extent_x, extent_y])
            xmin, ymin = np.max([xmin - extent, 0]), np.max([ymin - extent, 0])
            xmax = np.min([xmin + width + 2 * extent, xd])
            ymax = np.min([ymin + height + 2 * extent, yd])
            box_xd, box_yd = xmax - xmin, ymax - ymin

            m_subset = magnitudes[ymin:ymax, xmin:xmax]
            p_subset = phis[ymin:ymax, xmin:xmax]
            t_subset = thetas[ymin:ymax, xmin:xmax]

            im_to_show = p_subset if show_both == 'phis' else t_subset
            cmap_to_show = ciecam02_cmap() if show == 'phis' else cm.coolwarm_r

            # Create a pair of (x, y) coordinates
            x = np.linspace(1, box_xd - 2, box_xd, dtype=np.uint8)
            y = np.linspace(1, box_yd - 2, box_yd, dtype=np.uint8)
            X, Y = np.meshgrid(x, y)

            # Create vector lengths
            U = np.ones_like(m_subset) if normalize else m_subset * \
                np.cos(p_subset) * np.sin(t_subset)
            V = np.ones_like(m_subset) if normalize else m_subset * \
                np.sin(p_subset) * np.sin(t_subset)

            # index_to_show = index if not show_both else 2 * index - 1
            if plot_index % num_columns == 0:
                num_rows += 1
                n = len(fig.axes)

                for i in range(n):
                    fig.axes[i].change_geometry(num_rows, num_columns, i + 1)

            ax = plt.subplot(num_rows, num_columns, plot_index)
            ax.autoscale()
            ax.imshow(im_to_show, cmap=cmap_to_show)
            ax.imshow(masked_circles[ymin:ymax, xmin:xmax],
                      interpolation='none', alpha=alpha)
            ax.add_artist(ScaleBar(scale, box_alpha=0.8))

            title = f'({xmin}, {ymin}) to ({xmax}, {ymax}), ' \
                    f'{width_nm}x{height_nm} nm\n' \
                rf'$\alpha:\ {alpha_avg:.3f},\ \sigma_{{\alpha}}:\ {alpha_std:.2f}\quad $' \
                rf'$\phi_{{dev}}:\ {dev_avg:.3f},\ \sigma_{{dev}}:\ {dev_std:.2f}$'

            ax.set_title(title if show_title else '')

            colors = itertools.cycle(['red', 'yellow', 'green', 'blue'])
            num_pts = len(path)
            pts = [path[0], path[int(num_pts / 4)], path[int(num_pts / 2)],
                   path[int(3 * num_pts / 4)]]
            for point in pts:
                ax.scatter(point[0] - xmin, point[1] - ymin, c=next(colors))

            if show_numbers:
                c = 'black' if not candidate else 'blue'
                label = index if not candidate else fr'{index}$\circlearrowright$' \
                    if alpha_avg < np.pi else fr'{index}$\circlearrowleft$'

                t = ax.text(0, 0, label, fontdict={'fontsize': 20}, c=c,
                            horizontalalignment='left', verticalalignment='top')
                t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor=c))

            ax.quiver(X, Y, U, V, units='dots',
                      angles=p_subset * 180 / np.pi, pivot='mid')

            if show_axes:
                span = 5
                ax.set_xticks(x[::span])
                ax.set_xticklabels((x + xmin - 1).astype(int)[::span])
                ax.set_yticks(y[::span])
                ax.set_yticklabels((y + ymin - 1).astype(int)[::span])
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            if show_both:
                p_m_subset = phis_m[ymin:ymax, xmin:xmax]

                ax = plt.subplot(num_rows, num_columns, 2 * index)
                ax.autoscale()
                ax.imshow(p_m_subset, cmap=ciecam02_cmap())
                ax.imshow(masked_circles[ymin:ymax, xmin:xmax], interpolation='none',
                          alpha=alpha, cmap=cm.binary)
                ax.add_artist(ScaleBar(scale, box_alpha=0.8))

                if show_numbers:
                    t = ax.text(0, 0, index, fontdict={'fontsize': 20}, c='black',
                                horizontalalignment='left', verticalalignment='top')
                    t.set_bbox(
                        dict(facecolor='white', alpha=0.8, edgecolor='black'))

                ax.quiver(X, Y, U, V, units='dots',
                          angles=p_subset * 180 / np.pi, color='white')

                ax.set_title(title)
                span = 5
                ax.set_xticks(x[::span])
                ax.set_xticklabels((x + xmin - 1).astype(int)[::span])
                ax.set_yticks(y[::span])
                ax.set_yticklabels((y + ymin - 1).astype(int)[::span])

            # sm = cm.ScalarMappable(cmap=cm.coolwarm)
            # cbar = fig.colorbar(sm, ax=ax, shrink=0.3);
            # cbar.set_ticks([0, 0.5, 1]);
            # cbar.ax.set_yticklabels(['Down', f'X-Y', 'Up']);
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # cbar = plt.colorbar(sm, cax=cax)
            # cbar.set_ticks([0, 0.5, 1])
            # cbar.ax.set_yticklabels(['Down', f'X-Y', 'Up'])

            plot_index += 1 if not show_both else 2

        index += 1

    fig.set_size_inches(num_columns * column_width, num_rows * column_width)

    return candidates, all_contours


def show_contour_sizes(widths, heights, xerror, yerror):
    line = np.min((widths, heights)), np.max((widths, heights))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(widths, heights, xerror, yerror, fmt='.', ms=6)
    ax.plot(line, line, linestyle='dashed', alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('width [nm]')
    ax.set_ylabel('height [nm]')


def show_contours_overview(thetas_flattened, phis_flattened, contours_g, magnitudes_2d_flattened, axis_1, axis_2, axis_3):
    fig = plt.figure(figsize=(20, 20))

    ax1 = fig.add_subplot(221)
    ax1.imshow(thetas_flattened, cmap=cm.coolwarm_r)

    show_all_circles(thetas_flattened, contours_g, ax1, alpha=0.5)

    sm = cm.ScalarMappable(cmap=cm.coolwarm)
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.3)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.set_yticklabels(['Down', f'{axis_1}-{axis_2}', 'Up'])
    #cbar.set_label('Magnetization in Z');
    # ax1.imshow(masked_circles, interpolation='none')
    ax1.set_title(rf'M$_{axis_3}$ magnetization')

    ax2 = fig.add_subplot(222, polar=True)
    show_phase_colors_circle(
        ax2, add_dark_background=False, text_color='black')

    ax3 = fig.add_subplot(223)
    ax3.imshow(phis_flattened, cmap=ciecam02_cmap())
    show_all_circles(thetas_flattened, contours_g,
                     ax3, alpha=0.5, color='white')

    ax3.set_title(rf'M$_{axis_1}$-M$_{axis_2}$ plane')

    ax4 = fig.add_subplot(224)
    phis_flattened_rgba = render_phases_and_magnitudes(
        phis_flattened, magnitudes_2d_flattened)
    ax4.imshow(phis_flattened_rgba, cmap=ciecam02_cmap())
    show_all_circles(thetas_flattened, contours_g,
                     ax4, alpha=0.5, color='white')

    sm = cm.ScalarMappable(cmap=cm.binary_r)
    cbar = fig.colorbar(sm, ax=ax4, shrink=0.3)
    cbar.set_ticks([0, 1])
    cbar.ax.set_yticklabels([f'All {axis_3}', f'All {axis_1}-{axis_2}'])

    ax4.set_title(rf'M$_{axis_1}$-M$_{axis_2}$ plane with magnitude')


def show_groups(contours, magnitudes, phis, thetas, scale, normalize=True):
    """Plot groups of circular contours"""
    groups, _ = find_group_contours(contours)

    group_count = len(groups)

    # print(f'{group_count} groups:\n{groups}\n\n{len(loners)} loners:\n{loners}')
    contour_groups = [[np.concatenate(
        [np.squeeze(c) for c in [contours[i] for i in g]])] for g in groups]

    boxes_g = [get_bounding_boxes(cg) for cg in contour_groups]

    col_width = 12
    num_cols = 1
    num_rows = np.ceil(group_count / num_cols).astype(int)

    plt.figure(figsize=(num_cols * col_width, num_rows * col_width))
    masked_circles, _, _ = show_all_circles(thetas, contours)

    for index in range(group_count):
        box, group = boxes_g[index], groups[index]
        xmin, ymin, width, height = np.squeeze(box[0])
        xmax, ymax = xmin + width, ymin + height
        box_xd, box_yd = xmax - xmin, ymax - ymin

        _, centers = get_bounding_boxes([contours[i] for i in group])

        m_subset = magnitudes[ymin:ymax, xmin:xmax]
        p_subset = phis[ymin:ymax, xmin:xmax]
        t_subset = thetas[ymin:ymax, xmin:xmax]

        x = np.linspace(1, box_xd - 2, box_xd, dtype=np.uint8)
        y = np.linspace(1, box_yd - 2, box_yd, dtype=np.uint8)
        X, Y = np.meshgrid(x, y)

        # Create vector lengths
        U = np.ones_like(m_subset) if normalize else m_subset * \
            np.cos(p_subset) * np.sin(t_subset)
        V = np.ones_like(m_subset) if normalize else m_subset * \
            np.sin(p_subset) * np.sin(t_subset)

        ax = plt.subplot(num_rows, num_cols, index + 1)
        ax.imshow(thetas[ymin:ymax, xmin:xmax], cmap=cm.coolwarm_r, alpha=0.8)
        ax.imshow(masked_circles[ymin:ymax, xmin:xmax],
                  interpolation='none', alpha=0.5)
        ax.quiver(X, Y, U, V, units='dots',
                  angles=p_subset * 180 / np.pi, pivot='mid')

        for l_index in range(len(centers)):
            label = group[l_index] + 1
            t = ax.text(centers[l_index][0] - xmin, centers[l_index][1] - ymin, label,
                        alpha=1, color='black',
                        horizontalalignment='center', verticalalignment='center')
            t.set_bbox(dict(facecolor='white', alpha=0.65, edgecolor='black'))

        span = 5
        ax.set_xticks(x[::span])
        ax.set_xticklabels((x + xmin - 1).astype(int)[::span])
        ax.set_yticks(y[::span])
        ax.set_yticklabels((y + ymin - 1).astype(int)[::span])

        title = f'({xmin}, {ymin}) to ({xmax}, {ymax}), ' \
            f'{int(width * scale * scale_multiplier)}x{int(height * scale * scale_multiplier)} nm\n'
        ax.set_title(title)


def show_ltem_data(ltem_magnitudes, ltem_phases, ltem_contours, ltem_box_widths, ltem_xerror, ltem_yerror, name, use_cutoff=False):
    ydim, xdim = ltem_magnitudes.shape
    cutoff = np.min((xdim, ydim)) // 32
    phases_reduced = ltem_phases[cutoff:-cutoff,
                                 cutoff:-cutoff] if use_cutoff else ltem_phases
    phases_reduced = ltem_phases[512:1024,
                                 512:1024] if use_cutoff else ltem_phases
    magnitudes_reduced = ltem_magnitudes[cutoff:-cutoff,
                                         cutoff:-cutoff] if use_cutoff else ltem_magnitudes
    magnitudes_reduced = ltem_magnitudes[512:1024,
                                         512:1024] if use_cutoff else ltem_magnitudes
    magnitudes_norm = magnitudes_reduced / magnitudes_reduced.max()

    plt.figure(figsize=(16, 16))
    ax1 = plt.subplot(221)
    ax1.imshow(np.zeros_like(phases_reduced), cmap='gray')
    ax1.imshow(phases_reduced, alpha=magnitudes_norm, cmap=ciecam02_cmap(), origin='lower')
    # ax1.imshow(magnitudes_reduced, cmap=cm.binary_r, alpha=0.5)
    # show_all_circles(magnitudes_reduced, ltem_contours, ax1, color='white')
    #ax1.set_title(f'Phases for {ltem_data_name}');
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.add_artist(ScaleBar(1.987e-9))
    ax1.add_artist(ScaleBar(ltem_xerror))
    ax1.set_title('LTEM magnitudes CIECAM')

    ax2 = plt.subplot(223)
    # ax2.imshow(ltem_circular_contours)
    ax2.imshow(magnitudes_reduced, cmap='gray', origin='lower')
    # show_all_circles(magnitudes_reduced, ltem_contours, ax2, reverse=True, color='white')
    # show_all_circles(magnitudes_reduced,
    #                 ltem_circular_contours, ax2, alpha=0.5, show_numbers=False)
    #ax2.set_title(f'Magnitudes for {ltem_data_name}');
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('LTEM magnitudes')

    ax3 = plt.subplot(222, polar=True)
    show_phase_colors_circle(add_dark_background=False, ax=ax3)

    ax4 = plt.subplot(224)
    ax4.hist(ltem_box_widths, bins=25)
    ax4.set_xlabel('Pixels')
    ax4.set_yticks([])
    ax4.set_title(f'{name} contour widths')


def show_magnetization_components(magnitudes, m_1, m_2, m_3, axis_1, axis_2, axis_3):
    azim, elev = 15, 10

    xs, ys = np.linspace(1, magnitudes.shape[1], magnitudes.shape[1]), np.linspace(
        1, magnitudes.shape[0], magnitudes.shape[0])
    Xs, Ys = np.meshgrid(xs, ys)

    zs = m_3  # magnitudes * np.cos(thetas_flattened)

    fig = plt.figure(figsize=(30, 11))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.azim = azim
    ax1.elev = elev
    # ax.contour3D(Xs, Ys, zs, 100, cmap=cm.coolwarm, alpha=0.5);
    ax1.plot_surface(
        Xs, Ys, zs, cmap=cm.coolwarm, linewidth=0, alpha=0.3)
    ax1.plot_surface(Xs, Ys, np.zeros_like(Xs), alpha=0.5)

    ax1.set_xlabel(fr'M$_{axis_1}$')
    ax1.set_ylabel(fr'M$_{axis_2}$')
    ax1.set_zlabel(fr'M$_{axis_3}$')
    ax1.set_title(f'{axis_3} component of surface magnetization')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.azim = azim
    ax2.elev = elev
    # ax.contour3D(Xs, Ys, zs, 100, cmap=cm.coolwarm, alpha=0.5);
    ax2.plot_surface(
        Xs, Ys, m_1, cmap=cm.coolwarm, linewidth=0, alpha=0.3)
    ax2.plot_surface(Xs, Ys, np.zeros_like(Xs), alpha=0.5)

    ax2.set_xlabel(fr'M$_{axis_2}$')
    ax2.set_ylabel(fr'M$_{axis_3}$')
    ax2.set_zlabel(fr'M$_{axis_1}$')
    ax2.set_title(f'{axis_1} component of surface magnetization')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.azim = azim
    ax3.elev = elev
    # ax.contour3D(Xs, Ys, zs, 100, cmap=cm.coolwarm, alpha=0.5);
    ax3.plot_surface(
        Xs, Ys, m_2, cmap=cm.coolwarm, linewidth=0, alpha=0.3)
    ax3.plot_surface(Xs, Ys, np.zeros_like(Xs), alpha=0.5)

    ax3.set_xlabel(fr'M$_{axis_1}$')
    ax3.set_ylabel(fr'M$_{axis_3}$')
    ax3.set_zlabel(fr'M$_{axis_2}$')
    ax3.set_title(f'{axis_2} component of surface magnetization')


def show_magnitude_distribution(magnitudes):
    bins = 32

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)
    n, _, _ = ax.hist(magnitudes.flatten(), bins=bins, zorder=10)
    max_count = max(n)
    avg = (1 + 2 * np.squeeze(np.where(n == max_count))[()]) / (2 * bins)
    spline = UnivariateSpline(np.linspace(0, 1, bins), n - max_count / 2, s=0)
    plt.axvspan(*spline.roots(), facecolor='gray', alpha=0.4, zorder=0)
    ax.set_title(f'Magnitudes normalized (max count at {avg:.2f})')


def show_phase_colors_circle_old(ax=None, add_dark_background=True,
                                 text_color='white'):
    """Plot a ring of colors for a legend."""
    xs = np.arange(0, 2 * np.pi, 0.01)
    ys = np.ones_like(xs)

    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1, polar=True)

    fig = plt.gcf()
    dim = (fig.get_size_inches() * fig.dpi)[0]

    if add_dark_background:
        ax.scatter(0, 0, s=dim ** 2, marker='o', color='#3A404C')

    ax.scatter(xs, ys, c=xs, s=(dim / 8) ** 2, lw=0,
               cmap=ciecam02_cmap(), vmin=0, vmax=2 * np.pi)

    ax.set_yticks(())
    ax.tick_params(axis='x', colors=text_color)
    ax.set_anchor('W')


def show_phase_colors_circle(ax=None, add_dark_background=True,
                             text_color='white'):
    """Plot a ring of colors for a legend."""
   # Generate a figure with a polar projection
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    fig = plt.gcf()

    if add_dark_background:
        dark = '#3A404C'
        # fig.patch.set_facecolor(dark)
        ax.patch.set_facecolor(dark)
        text_color = dark
    else:
        fig.patch.set_facecolor('white')
        text_color = 'black'

    # Plot a color mesh on the polar plot with the color set by the angle

    n = 180  # the number of secants for the mesh
    t = np.linspace(0, 2 * np.pi, n)  # theta values
    # radius values; change 0.6 to 0 for full circle
    r = np.linspace(0.6, 1, 2)
    rg, tg = np.meshgrid(r, t)        # create r,theta meshgrid
    # c = tg                          # define color values as theta value
    # plot the colormesh on axis with colormap
    im = ax.pcolormesh(t, r, tg.T, cmap=ciecam02_cmap(), shading='auto')
    ax.set_yticklabels([])            # turn off radial tick labels (yticks)
    # cosmetic changes to tick labels
    ax.tick_params(axis='x', pad=15, labelsize=14, colors=text_color)
    # ax.spines['polar'].set_visible(False)    # turn off the axis spines.


def show_subplot(image, rows=1, cols=1, pos=1, title='', vmin=0, vmax=1,
                 ax=None, hide_axes=True):
    """Add a subplot with values normalized to 0..1."""
    if ax is None:
        ax = plt.subplot(rows, cols, pos)

    ax.imshow(image, vmin=vmin, vmax=vmax, cmap='gray')
    ax.grid(False)
    ax.set_title(title)

    if hide_axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    return ax


def show_subplot_raw(image, rows=1, cols=1, pos=1, title='', ax=None,
                     hide_axes=True):
    """Add a subplot without normalized values."""
    return show_subplot(image, rows, cols, pos, title,
                        vmin=np.min(image), vmax=np.max(image), ax=ax,
                        hide_axes=hide_axes)


def show_vector_plot(im_x, im_y, ax=None, color='white', scale=2, divisor=32):
    """Create a vector plot of the phases."""
    # Get dimensions
    yd, xd = im_x.shape

    X = np.linspace(xd / divisor, xd * (divisor - 1) /
                    divisor, divisor, dtype=np.uint8)
    Y = np.linspace(yd / divisor, yd * (divisor - 1) /
                    divisor, divisor, dtype=np.uint8)

    # Create a pair of (x, y) coordinates
    U, V = np.meshgrid(X, Y)
    x, y = U.ravel(), V.ravel()

    # Pull the values at those coordinates.
    Xs = convolve2d(im_x, np.ones((15, 15)), mode='same')[y, x]
    Ys = convolve2d(im_y, np.ones((15, 15)), mode='same')[y, x]

    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    # Show the plot
    ax.quiver(x, y, Xs, Ys, angles='uv', scale_units='dots', color=color,
              scale=scale)


def display_results(img_contrast_phase, img_denoised_1, img_flat_1,
                    img_denoised_2, img_flat_2, img_intensity,
                    img_scale, full_name, arrow_scale=4, arrow_color='black',
                    axis_1='x', axis_2='y'):
    """Plot the results."""
    fig = plt.figure(figsize=(20, 15), constrained_layout=True)

    gs = fig.add_gridspec(3, 3)
    gs.update(wspace=0.3, hspace=0.3)

    # Contrast image
    ax1 = plt.subplot(gs[:-1, :-1])
    ax1.imshow(img_contrast_phase)
    show_vector_plot(img_denoised_1, img_denoised_2, ax=ax1, color=arrow_color,
                     scale=arrow_scale)
    ax1.add_artist(ScaleBar(img_scale, box_alpha=0.8))
    ax1.set_title('Domains in the {}-{} plane for {}'.format(axis_1,
                                                             axis_2, full_name),
                  fontdict={'fontsize': 24})

    # Vector legend
    ax2 = plt.subplot(gs[:-1, -1], polar=True)
    show_phase_colors_circle(ax2)
    ax2.set_title('Magnetization angle', fontdict={'fontsize': 20})

    # Flattened intensity
    ax3 = plt.subplot(gs[-1, 0])
    ax3.imshow(img_intensity, cmap='gray')
    ax3.add_artist(ScaleBar(img_scale))
    ax3.set_title('Intensity flattened')

    # Flattened M1
    ax4 = plt.subplot(gs[-1, 1])
    ax4.imshow(img_flat_1, cmap='gray')
    ax4.add_artist(ScaleBar(img_scale))
    ax4.set_title('M{} flattened: {:.3f} to {:.3f}'.format(
        axis_1, img_flat_1.min(), img_flat_1.max()))

    # Add a colorbar
    add_colorbar(img_flat_1, ax4, r'$M_{rel}$')

    # Flattened M2
    ax5 = plt.subplot(gs[-1, -1])
    ax5.imshow(img_flat_2, cmap='gray')
    ax5.add_artist(ScaleBar(img_scale))
    ax5.set_title('M{} flattened: {:.3f} to {:.3f}'.format(
        axis_2, img_flat_2.min(), img_flat_2.max()))

    # Add a colorbar
    add_colorbar(img_flat_1, ax5, r'$M_{rel}$')

    # Turn off grids and axes except for the legend plot
    for ax in fig.get_axes():
        if len(ax.images) > 0:
            ax.grid(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
