"""Functions for pyNISTview."""

import imutils, glob, io

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d
from scipy.optimize import least_squares

from netCDF4 import Dataset

from colorspacious import cspace_convert

import cv2

from selelems import circle_array

sempa_file_suffix = 'sempa'


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
                    features=1024, match_percent=0.15):
    im_1 = rescale_to_8_bit(intensity_1)
    im_2 = rescale_to_8_bit(intensity_2)
    im_x = rescale_to_8_bit(m_1)
    im_y = rescale_to_8_bit(m_2)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(features)

    keypoints1, descriptors1 = orb.detectAndCompute(im_1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im_2, None)

    # Match features.
    # matcher = cv2.DescriptorMatcher_create(
    # cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * match_percent)
    matches = matches[:num_good_matches]

    # Draw top matches
    im_matches = cv2.drawMatches(im_1, keypoints1, im_2, keypoints2, matches,
                                 None)
    im_keypoints = cv2.drawKeypoints(im_1, keypoints1, np.array([]))
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # a_t = cv2.getAffineTransform(points1[:3], points2[:3])

    # Use homography
    height, width = im_1.shape
    intensity_1_h = rescale_to(cv2.warpPerspective(im_1, h, (width, height)),
                               intensity_1)
    m_1_h = rescale_to(cv2.warpPerspective(im_x, h, (width, height)), m_1)
    m_2_h = rescale_to(cv2.warpPerspective(im_y, h, (width, height)), m_2)

    # Figure out translation of homographies
#     first_x_y = (h[0, 2]).astype(int)
#     last_y_x = (np.where(
#         intensity_1_h[-1, h[0, 2].astype(int):] > min(intensity_1_h[-1]))[0][
#                     0] + h[0, 2]).astype(int)
    
    # Find the contour of the image
    cnts = np.squeeze(cv2.findContours(rescale_to_8_bit(intensity_1_h), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2])

    # Find the center point of the contour
    t_x, t_y = np.mean(cnts[:,0]), np.mean(cnts[:,1])
    
    # Now measure the distance to each point in the contour from the center for quadrants
    # 2 and 3 (top left and bottom left)
    q2 = np.squeeze([point for point in cnts if point[0] < t_x and point[1] < t_y])
    dists_q2 = [np.sqrt((pt[0] - t_x) ** 2 + (pt[1] - t_y) ** 2) for pt in q2]

    q3 = np.squeeze([point for point in cnts if point[0] < t_x and point[1] > t_y])
    dists_q3 = [np.sqrt((pt[0] - t_x) ** 2 + (pt[1] - t_y) ** 2) for pt in q3]

    # The top left and bottom left points will be the points in the quadrants with the greatest
    # distances.
    top_left, bottom_left = q2[np.argmax(dists_q2)], q3[np.argmax(dists_q3)]

    x_min = np.max((np.min(bottom_left[0]), np.max(top_left[0])))
    y_min = top_left[1]
    y_max = bottom_left[1]

    # Crop the original images so they all cover the same area
    results_r = [img[y_min:y_max, x_min:] for img in
                 (intensity_1_h, intensity_2, m_1_h, m_2_h, m_3, m_4)]
    results_h = (intensity_1_h, m_1_h, m_2_h)

    return results_r, results_h, h, im_matches, im_keypoints


def clean_image(image, sigma=50, h=25):
    """Remove gradient and noise from image."""
    # Rescale to 0..255 for filters
    image_shifted = np.asarray(rescale_to_8_bit(image))

    image_denoised = cv2.fastNlMeansDenoising(np.asarray(
        rescale(image_shifted, 0, 255), dtype=np.uint8), None, h, 7, 21)
    image_denoised_blurred = gaussian_filter(image_denoised, sigma)

    return rescale_to(image_denoised, image),rescale_to(image_denoised_blurred, image)


def create_ciecam02_cmap():
    """Create a perceptually uniform colormap based on CIECAM02."""
    # Based on https://stackoverflow.com/questions/23712207/cyclic-colormap-without-visual-distortions-for-use-in-phase-angle-plots

    # first draw a circle in the cylindrical JCh color space.
    # First channel is lightness, second chroma, third hue in degrees.
    color_circle = np.ones((256, 3)) * 60
    color_circle[:, 1] = np.ones((256)) * 45
    color_circle[:, 2] = np.arange(0, 360, 360 / 256)
    color_circle_rgb = cspace_convert(color_circle, 'JCh', 'sRGB1')

    return mpl.colors.ListedColormap(color_circle_rgb)


def file_for(files, token):
    """Convenience method for extracting file locations"""

    return files[[i for i, item in enumerate(files) if token in item]][0]


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


def find_offsets(img_1, img_2, img_3=None):
    """Minimize the min to max spread of the images to get most consistent magnitude."""
    # Find a reasonable starting point
    # x0 = find_com(img_1, img_2, img_3)

    img3 = img_3 if img_3 is not None else np.zeros_like(img_1)

    x0 =(img_1.mean(), img_2.mean(), img3.mean())
    # bounds=(-4, 4)
    
    vals = (np.max(abs(img_1)), np.max(abs(img_2)), np.max(abs(img_3)))

    bounds = (np.multiply(-1, vals), vals)

    res = least_squares(ptp_magnitudes, x0, bounds=bounds, args=(img_1, img_2, img_3))
    # res = least_squares(std_magnitudes, x0, method='trf', args=(img_1, img_2, img_3))

    return res.x, res.status, res.message


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
    # First handle X and Y normalized
    phis = np.arctan2(im_y / im_y.max(), im_x / im_x.max())

    # arctan2 works over the range -pi to pi; shift everything to 0 to 2pi
    # for color mapping
    phis = [[x if x >= 0 else x + 2 * np.pi for x in row] for row in phis]

    # Now Z. Set to pi/2 for 2D.
    thetas = np.ones_like(im_x) * np.pi / 2

    if im_z is not None:
        # Get the XY magnitudes, then arctan that over Z. Range 0 to pi.
        xy_magnitudes = get_magnitudes(im_x, im_y)
        thetas = np.arctan2(im_z / im_z.max(),
                            xy_magnitudes / xy_magnitudes.max()) + np.pi/2

    return (np.asarray(phis), thetas)


def get_scale(file_path):
    """Determine the size of a single pixel in the input image."""
    file = Dataset(file_path, 'r')

    full_scale = np.abs(file.variables['vertical_full_scale'][...].data)
    magnification = np.abs(file.variables['magnification'][...].data)
    dim = file.variables['image_data'].shape[0]

    adjusted_scale = full_scale / magnification / dim

    return adjusted_scale


def image_data(file_path):
    """Read a file in NetCDF format and return the image data and axis."""
    file = Dataset(file_path, 'r')

    return file.variables['image_data'][...].data.T, str(
        file.variables['channel_name'][...].data[1], 'utf-8')


def import_files(name, runs, indir):
    # Read in image files. Return a dictionary of image data
    # {im1: [], im2: [], m1: [], m2: [], m3: [], m4: [], scale: scale}
    
    image_dict = {}

    # Get all the i* and m* files. Should end up with 8 of them.

    files = []

    for run in runs:
        full_name = name + f'{run:0>3}'
        file = indir + full_name
        
        for f in ['x', 'y', 'z']:
            files.extend(glob.glob(file + '*' + f + '*' + sempa_file_suffix))
            
    files = np.array(sorted(files), dtype="object")

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

    image_dict['scale'] = get_scale(files[0])
    
    return image_dict


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


def ptp_magnitudes(variables, image_1, image_2, image_3=None):
    """Calculate the min to max spreads for magnitudes; used for optimizing the offset."""

    # Allow for 2D processing
    var_3 = variables[2] if image_3 is not None else 0
    arg_3 = image_3 - variables[2] if image_3 is not None else np.zeros_like(image_1)

    return np.ptp(
        get_magnitudes(image_1 - variables[0], image_2 - variables[1], arg_3 - var_3))


def std_magnitudes(variables, image_1, image_2, image_3=None):
    """Calculate the min to max spreads for magnitudes; used for optimizing the offset."""

    # Allow for 2D processing
    var_3 = variables[2] if image_3 is not None else 0
    arg_3 = image_3 - variables[2] if image_3 is not None else np.zeros_like(image_1)

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
    cmap = create_ciecam02_cmap()

    # Use CIECAM02 color map, convert to sRGB1 (to facilitate intensity adjustment)
    im = cmap(phases / (2 * np.pi))
    im_srgb = im[:, :, :3]
    im_adjusted = np.zeros_like(im_srgb)

    # Apply phase intensity mask to the contrast: low intensity -> dark, high intensity -> light
    for i in range(3):
        im_adjusted[:, :, i] = np.multiply(im_srgb[:, :, i], magnitudes)

    return im_adjusted


def rescale(data, to_min, to_max):
    """Rescale data to have to_min and to_max as min and max, respectively."""
    norm_data = (data - data.min()) / (data.max() - data.min())
    factor = to_max - to_min

    return norm_data * factor + to_min


def rescale_to(source, target):
    """Rescale source image to the same range as target."""
    return rescale(source, target.min(), target.max())


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

        plt.savefig(path + file + '_' + axis_1 + axis_2 + '_im.png')
        plt.close()
    

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
               cmap=create_ciecam02_cmap(), vmin=0, vmax=2 * np.pi)

    ax.set_yticks(())
    ax.tick_params(axis='x', colors=text_color)
    ax.set_anchor('W')


def show_phase_colors_circle(ax=None, add_dark_background=True,
                             text_color='white'):
    """Plot a ring of colors for a legend."""
   #Generate a figure with a polar projection
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    fig = plt.gcf()

    if add_dark_background:
        dark = '#3A404C'
        fig.patch.set_facecolor(dark)
        ax.patch.set_facecolor(dark)
        text_color = dark

    #Plot a color mesh on the polar plot
    #with the color set by the angle

    n = 180  #the number of secants for the mesh
    t = np.linspace(0, 2 * np.pi, n)  # theta values
    r = np.linspace(0.6, 1, 2)        # radius values; change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r, t)        # create a r,theta meshgrid
    # c = tg                          # define color values as theta value
    im = ax.pcolormesh(t, r, tg.T, cmap=create_ciecam02_cmap(), shading='auto')  # plot the colormesh on axis with colormap
    ax.set_yticklabels([])            # turn off radial tick labels (yticks)
    ax.tick_params(pad=15, labelsize=18, colors=text_color)      #cosmetic changes to tick labels
    ax.spines['polar'].set_visible(False)    #turn off the axis spine.


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


def show_vector_plot(im_x, im_y, ax=None, color='white', scale=2):
    """Create a vector plot of the phases."""
    # Get dimensions
    yd, xd = im_x.shape

    X = np.linspace(xd / 32, xd * 31/32, 32, dtype=np.uint8)
    Y = np.linspace(yd / 32, yd * 31/32, 32, dtype=np.uint8)

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
    # sm = cm.ScalarMappable(cmap='gray', norm=plt.Normalize(img_flat_1.min(),
    #                                                        img_flat_1.max()))
    # divider = make_axes_locatable(ax4)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    # cbar.set_label(r'$M_{rel}$')

    # Flattened M2
    ax5 = plt.subplot(gs[-1, -1])
    ax5.imshow(img_flat_2, cmap='gray')
    ax5.add_artist(ScaleBar(img_scale))
    ax5.set_title('M{} flattened: {:.3f} to {:.3f}'.format(
        axis_2, img_flat_2.min(), img_flat_2.max()))

    # Add a colorbar
    add_colorbar(img_flat_1, ax5, r'$M_{rel}$')
    # sm = cm.ScalarMappable(cmap='gray', norm=plt.Normalize(img_flat_2.min(),
    #                                                        img_flat_2.max()))
    # divider = make_axes_locatable(ax5)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    # cbar.set_label(r'$M_{rel}$')

    # Turn off grids and axes except for the legend plot
    for ax in fig.get_axes():
        if len(ax.images) > 0:
            ax.grid(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
