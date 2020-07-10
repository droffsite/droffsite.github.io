import os
import io

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
import scipy.misc

from netCDF4 import Dataset

from colorspacious import cspace_convert

import cv2


# Define convenience functions

def clean_image(image, sigma=50, h=25):
    """Remove gradient and noise from image"""

    # Apply large sigma gaussian filter and subtract from the image to remove gradient
    image_g = gaussian_filter(image, sigma)
    image_nograd = image - image_g

    # Incoming images are floats centered around zero; shift right so the range is positive
    image_shifted = image_nograd + np.abs(image_nograd.min())

    # Convert to 0-255 float for CV2 denoising
    image_8U = np.uint8(image_shifted * 255 / image_shifted.max())
    image_denoised = cv2.fastNlMeansDenoising(image_8U, None, h, 7, 21)

    # Convert back to a float centered around zero
    image_denoised = (image_denoised * image_shifted.max() / 255) - np.abs(image_nograd.min())

    return image_nograd, image_denoised


def combine_phase_and_contrast(phase, contrast):
    """Adjust the intensity of the contrast image according to phase magnitude"""

    # Set up the colormap
    cmap = create_ciecam02_cmap()

    # Use CIECAM02 color map, convert to sRGB1 (to facilitate intensity adjustment)
    im = cmap(contrast / (2 * np.pi))
    im_srgb = im[:, :, :3]
    im_adjusted = np.zeros_like(im_srgb)

    # Apply phase intensity mask to the contrast: low intensity -> dark, high intensity -> light
    for i in range(3):
        im_adjusted[:, :, i] = np.multiply(im_srgb[:, :, i], phase)

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


def find_offsets(img_1, img_2):
    """Use centroids to find offsets"""

    # Get the image dimensions
    dimx, dimy = img_1.shape[1], img_1.shape[0]

    # Calculate the 2D histogram to use for finding the CoM
    hist, _, _ = np.histogram2d(img_2.ravel(), img_1.ravel(), bins=dimx)

    # Get position in histogram. This will be coordinates.
    x_cm = np.average(range(1, dimx + 1), weights=[np.sum(hist[:, x]) for x in range(dimx)])
    y_cm = np.average(range(1, dimy + 1), weights=[np.sum(hist[y, :]) for y in range(dimy)])

    # Convert to original range
    x_cm = x_cm / dimx * np.ptp(img_1) + img_1.min()
    y_cm = y_cm / dimy * np.ptp(img_2) + img_2.min()

    return x_cm, y_cm


def get_contrast(im_x, im_y):
    """Return combined x and y contrast"""

    contrast = np.array([np.arctan2(y, x) for x, y in zip(im_x/im_x.max(), im_y/im_y.max())])

    # arctan2 works over the range -pi to pi; shift everything to 0 to 2pi for color mapping
    contrast_adjusted = contrast.ravel()
    for i in range(len(contrast_adjusted)):
        value = contrast_adjusted[i]
        positive = value >= 0
        contrast_adjusted[i] = value if positive else value + 2 * np.pi

    return contrast_adjusted.reshape(contrast.shape)


def get_file_key(f, key):
    """Return a simple string representation of a key in the image file"""
    return b''.join(f.variables[key][...].data).decode()


def get_phase_intensities(im_x, im_y):
    """Return an image of intensities: low -> dark, high -> light"""

    # Use x and y values to determine magnitude

    phase_intensities = [np.sqrt(x**2 + y**2)
                         for x, y in zip(im_x.ravel(), im_y.ravel())]
    phase_intensities /= np.max(phase_intensities)

    return phase_intensities.reshape(im_x.shape)


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


def display_results(img_contrast_phase, img_denoised_1, img_flat_1, img_denoised_2, img_flat_2, img_intensity, scale, full_name, axis_1='x', axis_2='y'):
    fig = plt.figure(figsize=(20, 15), constrained_layout=True);

    gs = fig.add_gridspec(3, 3)
    gs.update(wspace = 0.3, hspace = 0.3)

    # Contrast image
    ax1 = plt.subplot(gs[:-1, :-1]);
    ax1.imshow(img_contrast_phase);
    show_vector_plot(img_denoised_1, img_denoised_2, ax=ax1, color='black', scale=1.5);
    ax1.add_artist(ScaleBar(scale, box_alpha=0.8));
    ax1.set_title('Domains in the {}-{} plane for {}'.format(axis_1, axis_2, full_name, fontdict={'fontsize': 24}));

    # Vector legend
    ax2 = plt.subplot(gs[:-1, -1], polar=True);
    show_phase_colors_circle(ax2);
    ax2.set_title('Magnetization angle', fontdict={'fontsize': 20});

    # Flattened intensity
    ax3 = plt.subplot(gs[-1, 0]);
    ax3.imshow(img_intensity, cmap='gray');
    ax3.add_artist(ScaleBar(scale));
    ax3.set_title('Intensity flattened');

    # Flattened M1
    ax4 = plt.subplot(gs[-1, 1]);
    ax4.imshow(img_flat_1, cmap='gray');
    ax4.add_artist(ScaleBar(scale));
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
    ax5.add_artist(ScaleBar(scale));
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