import os
import io
import imutils

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image

from scipy.ndimage import gaussian_filter, median_filter, rotate, zoom
from scipy.signal import convolve2d
from scipy.optimize import least_squares
import scipy.misc

from netCDF4 import Dataset

from colorspacious import cspace_convert

import cv2

from selelems import *


# Define convenience functions

def clean_image(image, sigma=50, h=25):
    """Remove gradient and noise from image"""

    # Rescale to 0..255 for filters
    image_shifted = np.asarray(rescale(image, 0, 255), dtype=np.uint8)

    image_denoised = cv2.fastNlMeansDenoising(np.asarray(rescale(image_shifted, 0, 255), dtype=np.uint8), None, h, 7, 21)
    image_denoised_blurred = gaussian_filter(image_denoised, sigma)

    return rescale_to(image_denoised, image), rescale_to(image_denoised_blurred, image)


def render_phases_and_magnitudes(phases, magnitudes):
    """Adjust the intensity of the contrast image according to phase magnitude"""

    # Set up the colormap
    cmap = create_ciecam02_cmap()

    # Use CIECAM02 color map, convert to sRGB1 (to facilitate intensity adjustment)
    im = cmap(magnitudes / (2 * np.pi))
    im_srgb = im[:, :, :3]
    im_adjusted = np.zeros_like(im_srgb)

    # Apply phase intensity mask to the contrast: low intensity -> dark, high intensity -> light
    for i in range(3):
        im_adjusted[:, :, i] = np.multiply(im_srgb[:, :, i], phases)

    return im_adjusted


def create_ciecam02_cmap():
    """Create a perceptually uniform colormap based on CIECAM02"""
    # Based on https://stackoverflow.com/questions/23712207/cyclic-colormap-without-visual-distortions-for-use-in-phase-angle-plots

    # first draw a circle in the cylindrical JCh color space.
    # First channel is lightness, second chroma, third hue in degrees.
    color_circle = np.ones((256, 3)) * 60
    color_circle[:, 1] = np.ones((256)) * 45
    color_circle[:, 2] = np.arange(0, 360, 360 / 256)
    color_circle_rgb = cspace_convert(color_circle, 'JCh', 'sRGB1')

    return mpl.colors.ListedColormap(color_circle_rgb)


def find_com(img_1, img_2):
    """Use centroids to find center of mass"""

    # Get the image dimensions
    dimx, dimy = img_1.shape[1], img_1.shape[0]

    # Calculate the 2D histogram to use for finding the CoM
    hist, _, _ = np.histogram2d(img_1.ravel(), img_2.ravel(), bins=dimx)

    # Get position in histogram. This will be coordinates.
    x_cm = np.average(range(1, dimx + 1), weights=[np.sum(hist[:, x]) for x in range(dimx)])
    y_cm = np.average(range(1, dimy + 1), weights=[np.sum(hist[y, :]) for y in range(dimy)])

    # Convert to original range
    x_cm = x_cm / dimx * np.ptp(img_1) + img_1.min()
    y_cm = y_cm / dimy * np.ptp(img_2) + img_2.min()

    return x_cm, y_cm


def find_offsets(img_1, img_2):
    """Minimize the min to max spread of the images to get most consistent magnitude"""

    # Find a reasonable starting point
    x0 = find_com(img_1, img_2)

    res = least_squares(ptp_magnitudes, x0, bounds=(-4, 4), args=(img_1, img_2))

    return res.x, res.status, res.message


def ptp_magnitudes(variables, image_1, image_2):
    """Function to be used for optimizing the offset"""

    return np.ptp(get_magnitudes(image_1 - variables[0], image_2 - variables[1]))


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
    A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    B = image.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A[mask_flat], B[mask_flat], rcond=None)

    image_fit = np.sum(coeff * A, axis=1).reshape(image.shape)
    image_fit_masked = image_fit * mask

    #return (np.sum(coeff * A, axis=1) * mask.flatten()).reshape(image.shape)

    return image_fit, image_fit_masked


def get_phases(im_x, im_y):
    """Return phases of combined images in radians"""

    phases = np.arctan2(im_y/im_y.max(), im_x/im_x.max())

    # arctan2 works over the range -pi to pi; shift everything to 0 to 2pi for color mapping
    phases = [[x if x >= 0 else x + 2 * np.pi for x in row] for row in phases]

    return np.asarray(phases)


def get_file_key(f, key):
    """Return a simple string representation of a key in the image file"""
    return b''.join(f.variables[key][...].data).decode()


def get_magnitudes(im_x, im_y):
    """Return magnitudes of combined images: low -> dark, high -> light"""

    # Use x and y values to determine magnitude

    magnitudes = np.sqrt(im_x**2 + im_y**2)
    #magnitudes /= np.max(magnitudes)

    return magnitudes.reshape(im_x.shape)


def get_scale(file_path):
    """Determine the size of a single pixel in the input image"""

    file = Dataset(file_path, 'r')

    full_scale = np.abs(file.variables['vertical_full_scale'][...].data)
    magnification = np.abs(file.variables['magnification'][...].data)
    dim = file.variables['image_data'].shape[0]

    adjusted_scale = full_scale / magnification / dim

    return adjusted_scale


def image_data(file_path):
    """Read a file in NetCDF format and return the image data and axis"""

    file = Dataset(file_path, 'r')

    return file.variables['image_data'][...].data.T, str(file.variables['channel_name'][...].data[1], 'utf-8')


def mean_shift_filter(image, sp=1, sr=1):
    """Perform mean shift filtering"""

    # Convert to 8 bit color for the filter
    img = np.asarray(rescale(image, 0, 255), dtype=np.uint8)
    img = cv2.cvtColor(median_filter(img, 5), cv2.COLOR_GRAY2BGR)

    # sp = spatial window, sr = color window
    img_shift = cv2.pyrMeanShiftFiltering(img, sp, sr)

    # Convert back to grayscale
    img_shift = cv2.cvtColor(img_shift, cv2.COLOR_BGR2GRAY)

    return rescale(img_shift, image.min(), image.max())


def segment_image(image, segments=1, adaptive=False, sel=circle_array(1), erode_iter=1, close_iter=7):
    """Segment image into a defined number of objects. Create positive and negative masks for those objects."""

    # Rescale to 0-255 for thresholding
    image_shift = np.asarray(rescale(image, 0, 255), dtype=np.uint8)
    ret, image_thresh = cv2.threshold(image_shift, image_shift.mean(), 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Alternative thresholding mechanism
    #m_1_thresh = cv2.adaptiveThreshold(m_1_shift, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 0)
    #m_1_shift = cv2.cvtColor(m_1_shift, cv2.COLOR_GRAY2BGR)

    # Perform morphological operations. First kill small noise, then close the image
    image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_ERODE, sel, iterations=erode_iter)
    image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, sel, iterations=close_iter)

    # Find the segment contours
    snakes = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort the results by size and keep the requested number
    snakes = imutils.grab_contours(snakes)
    snakes = sorted(snakes, key=cv2.contourArea, reverse = True)[:segments]

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


def remove_line_errors(image, lines, use_rows=True):
    """Remove line errors using either rows or columns"""

    image_base = image if use_rows else image.T

    image_mean = np.asarray([np.mean(image_base[:lines, x]) for x in range(image.shape[1])]).T

    image_delined = image_base - np.tile(image_mean, (image.shape[0], 1))

    image_delined = image_delined if use_rows else image_delined.T

    return image_delined


def rescale(data, to_min, to_max):
    """Rescale data to have to_min and to_max as min and max, respectively"""

    norm_data = (data - data.min()) / (data.max() - data.min())
    factor = to_max - to_min

    return norm_data * factor + to_min


def rescale_to(source, target):
    """Rescale source image to the same range as target"""

    return rescale(source, target.min(), target.max())


def save_file(file_path, img, axis_1, axis_2, scale):
    """Save the image to a PNG file"""
    fig = plt.figure();
    plt.imshow(img);
    ax = plt.gca();
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.savefig(file_path + '_' + axis_1 + axis_2 + '.png');

    ax.add_artist(ScaleBar(scale, box_alpha=0.8));
    plt.savefig(file_path + '_' + axis_1 + axis_2 + '_scale.png');

    plt.close();


def show_phase_colors_circle(ax=None, add_dark_background=True):
    """Plot a ring of colors for a legend"""

    xs = np.arange(0, 2 * np.pi, 0.01)
    ys = np.ones_like(xs)

    if ax == None:
        plt.figure();
        ax = plt.subplot(1, 1, 1, polar=True);

    fig = plt.gcf();
    dim = (fig.get_size_inches() * fig.dpi)[0];

    if add_dark_background:
        ax.scatter(0, 0, s=dim**2, marker='o', color='#3A404C');

    ax.scatter(xs, ys, c=xs, s=(dim/8)**2, cmap=create_ciecam02_cmap(),
               vmin=0, vmax=2 * np.pi);

    ax.set_yticks(());
    ax.set_anchor('W');


def show_subplot(image, rows=1, cols=1, pos=1, title='', vmin=0, vmax=1, ax=None, hide_axes=True):
    """Add a subplot with values normalized to 0..1"""

    if ax == None:
        ax = plt.subplot(rows, cols, pos)

    ax.imshow(image, vmin=vmin, vmax=vmax, cmap='gray')
    ax.grid(False)
    ax.set_title(title)

    if hide_axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    return ax


def show_subplot_raw(image, rows=1, cols=1, pos=1, title='', ax=None, hide_axes=True):
    """Add a subplot without normalized values"""

    return show_subplot(image, rows, cols, pos, title,
                        vmin=np.min(image), vmax=np.max(image), ax=ax, hide_axes=hide_axes)


def show_vector_plot(im_x, im_y, ax=None, color='white', scale=2):
    """Create a vector plot of the domains"""

    # Get dimensions
    yd, xd = im_x.shape

    X = np.linspace(xd / 32, xd, 32, dtype=np.uint8)
    Y = np.linspace(yd / 32, yd, 32, dtype=np.uint8)

    # Create a pair of (x, y) coordinates
    U, V = np.meshgrid(X, Y)
    x, y = U.ravel(), V.ravel()

    # Pull the values at those coordinates.
    Xs = convolve2d(im_x, np.ones((15, 15)), mode='same')[y, x]
    Ys = convolve2d(im_y, np.ones((15, 15)), mode='same')[y, x]

    if ax == None:
        plt.figure();
        ax = plt.subplot(1, 1, 1);

    # Show the plot
    ax.quiver(x, y, Xs, Ys, angles='uv',
              scale_units='dots', color=color, scale=scale);


def display_results(img_contrast_phase, img_denoised_1, img_flat_1, img_denoised_2, img_flat_2, img_intensity, img_scale, full_name, arrow_scale=4, arrow_color='black', axis_1='x', axis_2='y'):
    fig = plt.figure(figsize=(20, 15), constrained_layout=True);

    gs = fig.add_gridspec(3, 3)
    gs.update(wspace = 0.3, hspace = 0.3)

    # Contrast image
    ax1 = plt.subplot(gs[:-1, :-1]);
    ax1.imshow(img_contrast_phase);
    show_vector_plot(img_denoised_1, img_denoised_2, ax=ax1, color=arrow_color, scale=arrow_scale);
    ax1.add_artist(ScaleBar(img_scale, box_alpha=0.8));
    ax1.set_title('Domains in the {}-{} plane for {}'.format(axis_1, axis_2, full_name, fontdict={'fontsize': 24}));

    # Vector legend
    ax2 = plt.subplot(gs[:-1, -1], polar=True);
    show_phase_colors_circle(ax2);
    ax2.set_title('Magnetization angle', fontdict={'fontsize': 20});

    # Flattened intensity
    ax3 = plt.subplot(gs[-1, 0]);
    ax3.imshow(img_intensity, cmap='gray');
    ax3.add_artist(ScaleBar(img_scale));
    ax3.set_title('Intensity flattened');

    # Flattened M1
    ax4 = plt.subplot(gs[-1, 1]);
    ax4.imshow(img_flat_1, cmap='gray');
    ax4.add_artist(ScaleBar(img_scale));
    ax4.set_title('M{} flattened: {:.3f} to {:.3f}'.format(axis_1, img_flat_1.min(), img_flat_1.max()));

    # Add a colorbar
    sm = cm.ScalarMappable(cmap='gray', norm=plt.Normalize(img_flat_1.min(), img_flat_1.max()))
    divider = make_axes_locatable(ax4);
    cax = divider.append_axes('right', size='5%', pad=0.05);
    sm.set_array([]);
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical');
    cbar.set_label(r'$M_{rel}$')

    # Flattened M2
    ax5 = plt.subplot(gs[-1, -1]);
    ax5.imshow(img_flat_2, cmap='gray');
    ax5.add_artist(ScaleBar(img_scale));
    ax5.set_title('M{} flattened: {:.3f} to {:.3f}'.format(axis_2, img_flat_2.min(), img_flat_2.max()));

    # Add a colorbar
    sm = cm.ScalarMappable(cmap='gray', norm=plt.Normalize(img_flat_2.min(), img_flat_2.max()))
    divider = make_axes_locatable(ax5);
    cax = divider.append_axes('right', size='5%', pad=0.05);
    sm.set_array([]);
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical');
    cbar.set_label(r'$M_{rel}$')

    # Turn off grids and axes except for the legend plot
    for ax in fig.get_axes():
        if len(ax.images) > 0:
            ax.grid(False);
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)