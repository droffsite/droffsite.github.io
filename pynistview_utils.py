"""
pynistview_utils contains utility functions for processing data captured
by SEMPA. It is also used by the pyNISTview application.
"""

from unittest import expectedFailure
import imutils
import glob
import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.projections import get_projection_class
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.gridspec as gs

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d, find_peaks
from scipy.optimize import least_squares

from netCDF4 import Dataset

from colorspacious import cspace_convert

import cv2

from selelems import circle_array, diamond_array

sempa_file_suffix = "sempa"
scale_multiplier = 1e9


def add_colorbar(img, ax, label="", cmap="gray"):
    """Add a colorbar

    Args:
        img (array): Image on which to add colorbar
        ax (Axes): Axes for image
        label (str, optional): Label. Defaults to ''.
        cmap (str, optional): Colormap. Defaults to 'gray'.
    """
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(img.min(), img.max()))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(label)


def adjust_magnetization_ranges(m_1_r_g, m_2_r_g, m_3_r_g, axis_1, axis_2, axis_3):
    """Center the magnetization ranges and fit as needed

    Args:
        m_1_r_g (list): m_1 image data
        m_2_r_g (list): m_2 image data
        m_3_r_g (list): m_3 image data
    """
    # Find averages for each image and shift the values
    m_1_average = (np.max(m_1_r_g) - np.abs(np.min(m_1_r_g))) / 2
    m_2_average = (np.max(m_2_r_g) - np.abs(np.min(m_2_r_g))) / 2
    m_3_average = (np.max(m_3_r_g) - np.abs(np.min(m_3_r_g))) / 2

    m_1_shifted, m_2_shifted, m_3_shifted = (
        m_1_r_g - m_1_average,
        m_2_r_g - m_2_average,
        m_3_r_g - m_3_average,
    )

    # Optimize the amount of shift necessary by minimizing least squares fitting
    (m_1_offset, m_2_offset, m_3_offset), status, message = find_offsets(m_1_shifted, m_2_shifted, m_3_shifted)

    m_1_offset_opt, m_2_offset_opt, m_3_offset_opt = (
        m_1_average + m_1_offset,
        m_2_average + m_2_offset,
        m_3_average + m_3_offset,
    )

    m_1_adjusted, m_2_adjusted, m_3_adjusted = (
        m_1_r_g - m_1_offset_opt,
        m_2_r_g - m_2_offset_opt,
        m_3_r_g - m_3_offset_opt,
    )

    m_1_adjusted = m_1_adjusted - fit_image(m_1_adjusted)[0]
    m_2_adjusted = m_2_adjusted - fit_image(m_2_adjusted)[0]
    m_3_adjusted = m_3_adjusted - fit_image(m_3_adjusted)[0]

    print(
        "M{} average: {:.4f}, M{} average: {:.4f}, M{} average: {:.4f}".format(
            axis_1, m_1_average, axis_2, m_2_average, axis_3, m_3_average
        )
    )
    print(
        "M{} offset: {:.4f}, M{} offset: {:.4f}, M{} offset: {:.4f}".format(
            axis_1, m_1_offset, axis_2, m_2_offset, axis_3, m_3_offset
        )
    )

    print(
        "M{} original range: {:.4f} to {:.4f}; offset range: {:.3f}, {:.4f} to {:.4f}".format(
            axis_1,
            m_1_r_g.min(),
            m_1_r_g.max(),
            np.ptp(m_1_adjusted),
            m_1_adjusted.min(),
            m_1_adjusted.max(),
        )
    )
    print(
        "M{} original range: {:.4f} to {:.4f}; offset range: {:.3f}, {:.4f} to {:.4f}".format(
            axis_2,
            m_2_r_g.min(),
            m_2_r_g.max(),
            np.ptp(m_2_adjusted),
            m_2_adjusted.min(),
            m_2_adjusted.max(),
        )
    )
    print(
        "M{} original range: {:.4f} to {:.4f}; offset range: {:.3f}, {:.4f} to {:.4f}".format(
            axis_3,
            m_3_r_g.min(),
            m_3_r_g.max(),
            np.ptp(m_3_adjusted),
            m_3_adjusted.min(),
            m_3_adjusted.max(),
        )
    )

    return m_1_adjusted, m_2_adjusted, m_3_adjusted


def align_and_scale(
    intensity_1,
    intensity_2,
    m_1,
    m_2,
    m_3,
    m_4,
    features=512,
    match_percent=0.15,
    filter=0.78,
):
    """Use intensity_1 and intensity_2 (which ideally are identical) to align m_1 - 4

    Args:
        intensity_1 (array): XY intensity image
        intensity_2 (array): XZ intensity image
        m_1 (array): XY m_x
        m_2 (array): XY m_y
        m_3 (array): XZ m_z
        m_4 (array): XZ m_x
        features (int, optional): Max number of features. Defaults to 512.
        match_percent (float, optional): Target percentage for "good" matches. Defaults to 0.15.
        filter (float, optional): Not used. Defaults to 0.78.
    """
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
    print(f"Found {len(matches)} matches; keeping {num_good_matches}.")
    matches = matches[:num_good_matches]

    # Draw top matches
    im_matches = cv2.drawMatches(
        im_1,
        keypoints1,
        im_2,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
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
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransacReprojThreshold=2.0)
    # a_t = cv2.getAffineTransform(points1[:3], points2[:3])
    # h = cv2.warpAffine(points1, points2)
    # h, mask = cv2.findHomography(points1, np.float32(p2), cv2.RANSAC)
    # h = np.array([[9.91469067e-01, -3.06551887e-04,  0],
    #               [-1.66912051e-02,  9.97772576e-01,  0],
    #               [-8.31626757e-05,  1.28397011e-04,  1.00000000e+00]])

    # Use homography
    height, width = im_1.shape
    intensity_1_h = rescale_to(cv2.warpPerspective(im_1, h, (width, height)), intensity_1)
    # intensity_1_h = rescale_to(cv2.warpAffine(
    #     im_1, a_t, (width, height)), im_1)
    m_1_h = rescale_to(cv2.warpPerspective(im_x, h, (width, height)), m_1)
    m_2_h = rescale_to(cv2.warpPerspective(im_y, h, (width, height)), m_2)
    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(131)
    ax1.imshow(intensity_1, cmap="gray")
    ax2 = plt.subplot(132)
    ax2.imshow(intensity_2, cmap="gray")
    ax3 = plt.subplot(133)
    ax3.imshow(intensity_1_h, cmap="gray")

    # Figure out translation of homographies
    #     first_x_y = (h[0, 2]).astype(int)
    #     last_y_x = (np.where(
    #         intensity_1_h[-1, h[0, 2].astype(int):] > min(intensity_1_h[-1]))[0][
    #                     0] + h[0, 2]).astype(int)

    # Find the contour of the image
    cnts = cv2.findContours(rescale_to_8_bit(intensity_1_h), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
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
    print(f"Top left: {top_left}, Top right: {top_right}\nBottom left: {bottom_left}, Bottom right: {bottom_right}")

    # x_min = np.max((np.min(bottom_left[0]), np.max(top_left[0])))
    # y_min = top_left[1]
    # y_max = bottom_left[1]
    x_min, x_max = np.max((top_left[0], bottom_left[0])), np.min((top_right[0], bottom_right[0]))
    y_min, y_max = np.max((top_left[1], top_right[1])), np.min((bottom_right[1], bottom_right[1]))
    print(f"Cropping ({x_min}, {y_min}) to ({x_max}, {y_max})")

    # Crop the original images so they all cover the same area
    results_r = [img[y_min:y_max, x_min:x_max] for img in (intensity_1_h, intensity_2, m_1_h, m_2_h, m_3, m_4)]
    results_h = (intensity_1_h, m_1_h, m_2_h)

    return results_r, results_h, h, im_matches, im_keypoints


def calculate_distance(x1, x2):
    """Calculate the distance between two points"""
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)


def calculate_winding_number(x, y, z, xy_s=None, contour=None, wider_contour=True):
    """Calculate the winding number"""

    # n = 1/4pi int(M dot (dM/dx cross dMdy) dxdy)
    # Requires that M be normalized at all points
    M = np.array(normalize_3d(x[xy_s], y[xy_s], z[xy_s]))

    dmdx, dmdy = np.gradient(M, axis=(2, 1))

    if contour is not None:
        _, _, width, height = cv2.boundingRect(contour)
        mask_scale = 1 / np.minimum(height, width)

        small_mask = scale_contour(contour, 1 - mask_scale)
        large_mask = scale_contour(contour, 1 + mask_scale)
        scaled_mask = np.append(contour, small_mask, axis=0)
        scaled_mask = np.append(scaled_mask, large_mask, axis=0)

        # mask = np.zeros_like(x)
        # cv2.drawContours(mask, [scaled_mask], 0, 255, 1)
        # reduced_mask = mask[xy_s]
        reduced_mask = get_contour_mask(scaled_mask if wider_contour else contour, x, xy_s)
        # plt.imshow(mask[xy_s], cmap='gray')

        masks = np.array([reduced_mask, reduced_mask, reduced_mask])

        # M = np.ma.masked_where(masks, M)
        dmdx = np.ma.masked_where(masks, dmdx)
        dmdx = np.ma.masked_where(masks, dmdx)

        # M[0] = np.ma.masked_where(reduced_mask, M[0])
        # M[1] = np.ma.masked_where(reduced_mask, M[1])
        # M[2] = np.ma.masked_where(reduced_mask, M[2])

        # dmdx = np.ma.masked_where(reduced_mask, dmdx)
        # dmdy = np.ma.masked_where(reduced_mask, dmdy)

    cross_dmdx_dmdy = np.cross(dmdx, dmdy, axis=0)
    dot_m_dmdx_dmdy = np.einsum("i...,i...", M, cross_dmdx_dmdy)

    return np.sum(dot_m_dmdx_dmdy) / (4 * np.pi)


def clean_image(image, sigma=50, h=25):
    """Remove gradient and noise from image."""
    # Rescale to 0..255 for filters
    image_shifted = np.asarray(rescale_to_8_bit(image))

    image_denoised = cv2.fastNlMeansDenoising(
        np.asarray(rescale(image_shifted, 0, 255), dtype=np.uint8), None, h, 7, 21
    )
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
    color_circle_rgb = cspace_convert(color_circle, "JCh", "sRGB1")

    return mpl.colors.ListedColormap(color_circle_rgb)


def ciecam02_cmap_r():
    """Create a perceptually uniform colormap based on CIECAM02."""
    # Based on https://stackoverflow.com/questions/23712207/cyclic-colormap-without-visual-distortions-for-use-in-phase-angle-plots

    # first draw a circle in the cylindrical JCh color space.
    # First channel is lightness, second chroma, third hue in degrees.
    color_circle = np.ones((256, 3)) * 60
    color_circle[:, 1] = np.ones((256)) * 45
    color_circle[:, 2] = np.arange(360, 0, 360 / 256)
    color_circle_rgb = cspace_convert(color_circle, "JCh", "sRGB1")

    return mpl.colors.ListedColormap(color_circle_rgb)


def file_for(files, token):
    """Convenience method for extracting file locations"""

    return files[[i for i, item in enumerate(files) if token in item]][0]


def find_circular_contours(img, diff_threshold=0.3, comparator="gt", show=False, ltem=False):
    """Return an array of circular contours found in the input image. Also return the full array of contours."""
    circle_contours = []

    # Work on the values above (default) or below the mean
    # threshold = img.mean()
    threshold = np.pi / 2 if not ltem else img.mean()
    ups = np.where(img > threshold, 255, 0) if comparator == "gt" else np.where(img < threshold, 255, 0)
    # ups = np.logical_and(0.9 * np.pi / 2 <= img, img <= 1.1 * np.pi / 2)

    # OpenCV requires unsigned ints
    ups_b = ups.astype(np.uint8)
    kernel = circle_array(3)
    ups = cv2.morphologyEx(ups_b, cv2.MORPH_CLOSE, kernel)
    xdim, ydim = ups.shape[1], ups.shape[0]

    if show:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(ups_b, cmap="gray")
        plt.subplot(122)
        plt.imshow(ups, cmap="gray")

    # Find all contours
    contours, _ = cv2.findContours(ups, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print(np.asarray(contours).shape)

    # Remove the contours that are on the edges
    contours_g = [
        contour
        for contour in contours
        if 0 not in contour
        and xdim - 1 not in contour[:, :, 0]
        and ydim - 1 not in contour[:, :, 1]
        and len(np.squeeze(contour).shape) == 2
    ]

    for contour in [c for c in contours_g if len(c) > 4]:
        # Fit the contour to an ellipse. This requires there be at least 5 points in the contour
        # (thus the if). Throw out anything where the major and minor axes are different by more
        # than the threshold. Also check the area, as other shapes would meet the above.
        width, height = cv2.fitEllipse(contour)[1]
        # avg_radius = (width + height) / 4
        # expected_length = 2 * avg_radius * np.pi
        # length = cv2.arcLength(contour, True)
        # area = cv2.contourArea(contour)
        # expected_area = np.pi * avg_radius * avg_radius

        diff_dims = np.abs(width - height)
        # diff_lengths = np.abs(expected_length - length)
        # diff_areas = np.abs(expected_area - area)

        if (
            diff_dims < np.max([width, height]) * diff_threshold
            and is_contour_circular(contour, diff_threshold)
            # and diff_areas < np.max([expected_area, area]) * diff_threshold
            # and diff_lengths < np.max([expected_length, length]) * diff_threshold
        ):
            circle_contours.append(contour)

        # approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    return circle_contours[::-1], contours_g[::-1], contours[::-1]


def find_com(img_1, img_2, img_3=None):
    """Use centroids to find center of mass."""
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


def find_group_contours(contours):
    """Find groupings of circular contours."""

    boxes, centers = get_bounding_boxes(contours)

    # Determine total number of contours
    count = len(centers)

    # Create a matrix of distances between contours
    distances = np.asarray([[calculate_distance(centers[i], centers[j]) for j in range(count)] for i in range(count)])

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
        indices = [i for i in indices if i not in list(itertools.chain.from_iterable(found))]

    return groups, loners


def find_minimum_in_contour(cnt, img, count=100):
    """Find the coordinates of the first of count minima in img that are inside cnt

    Args:
        cnt (List of points): The contour
        img (2d image array): The image
        count (int, optional): The number of minima to find. Defaults to 100.

    Returns:
        tuple: First minimum found in the contour
    """
    # Just examine the subset of img that's in the region of the contour
    boxes, _ = get_bounding_boxes([cnt])
    xmin, ymin, xext, yext = boxes[0]
    img_subset = img[ymin : ymin + yext, xmin : xmin + xext]

    # Find the sorted list of indices for minima, then check if they are in the contour
    # Need to shift the points since they're from a subset of img
    minima = np.array(np.unravel_index(np.argsort(img_subset.ravel())[:count], (img_subset.shape))).T
    in_check = [cv2.pointPolygonTest(cnt, (m[1] + xmin, m[0] + ymin), False) for m in minima]

    # Any indices in the contour will put a 1 in in_check; get the first one, then shift it
    raw_point = minima[in_check.index(1)]
    point = (raw_point[1] + xmin, raw_point[0] + ymin)

    return point


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
        new_indices = np.atleast_1d(np.squeeze(np.where(np.logical_and(0 < dists, dists <= d))))
        new_indices = [i for i in new_indices if i not in neighbors]

        # print(f'Looking at {index}, found {new_indices} with neighbors {neighbors}')

        # Recursively call self to follow leads
        neighbors = np.unique(
            np.concatenate(
                (
                    neighbors,
                    find_neighbors_for(new_indices, boxes, distances, neighbors),
                )
            )
        )

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

    res = least_squares(ptp_magnitudes, x0, jac_magnitudes, bounds=bounds, args=(img_1, img_2, img3))
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
    A = np.array(
        [
            X * 0 + 1,
            X,
            Y,
            X**2,
            X**2 * Y,
            X**2 * Y**2,
            Y**2,
            X * Y**2,
            X * Y,
        ]
    ).T
    B = image.flatten()

    coeff, _, _, _ = np.linalg.lstsq(A[mask_flat], B[mask_flat], rcond=None)

    image_fit = np.sum(coeff * A, axis=1).reshape(image.shape)
    image_fit_masked = image_fit * mask

    return image_fit, image_fit_masked


def get_bounding_boxes(contours):
    """Return the bounding boxes for the passed in contours"""
    boxes = [cv2.boundingRect(contour) for contour in contours]
    centers = [(x + int(x_ext / 2), y + int(y_ext / 2)) for x, y, x_ext, y_ext in boxes]

    return boxes, centers


def get_contour_mask(contour, img, xy_s=None):
    """Return an array corresponding to the contour

    Args:
        contour (list of points): points in the contour
        img (array): image array
        xy_s (range): subsection of the image
    """

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [contour], 0, 255, 1)

    return mask if xy_s is None else mask[xy_s]


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

    file = Dataset(file_path, "r")

    data = file.variables[key][...].data
    if data.shape != ():
        data = b"".join(data).decode()

    return data


def get_magnitudes(im_x, im_y, im_z=None):
    """Return magnitudes of combined images: low -> dark, high -> light."""
    # Handle 2d case.
    im_z = np.zeros_like(im_x) if im_z is None else im_z

    magnitudes = np.sqrt(im_x**2 + im_y**2 + im_z**2)

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
        thetas = -np.arctan2(im_z / im_z.max(), xy_magnitudes / xy_magnitudes.max()) + np.pi / 2

    return (np.asarray(phis), thetas)


def get_phi_diff(path, phis, thetas, mod=2 * np.pi, flip_y=True):
    """Determine the average angle of magnetization relative to the
    contour it circles."""
    # Flip the sign of y for matplotlib (y=0 is at the top)
    factor = -1 if flip_y else 1

    # Determine the angle along the path from point to point. Need to
    # roll to connect end to beginning.
    path_diff = np.roll(path, -2) - path
    path_angle = np.arctan2(factor * path_diff[:, 1], path_diff[:, 0])
    # print(f"Path angle: {path_angle}")

    # Collect the phis along the path and subtract them from the path angles
    phis_path = np.array([phis[point[1], point[0]] for point in path])
    phis_diff = phis_path - path_angle
    # print(f'Phis path: {phis_path}')
    # print(f'Phis diff: {phis_diff}')

    # Find the minimum theta inside the path for calculating alpha
    if thetas is not None:
        center = find_minimum_in_contour(path, thetas)
        center_diff = center - np.roll(path, -2)
        center_angle = np.arctan2(center_diff[:, 1], center_diff[:, 0])
           
        thetas_gradient = np.gradient(thetas)
        grad_angle = np.arctan2(*thetas_gradient)
        thetas_angle = np.array([grad_angle[point[1], point[0]] for point in path])
        # print(f'Thetas gradient path: {thetas_angle}')
 
    # Calculate alpha per JC's definition: deviation from domain wall normal traveling
    # from low M_z to high M_z. Alpha ranges from 0 to 2 pi. The path goes clockwise
    # around the contour. For now, assume the center of the contour is where high M_z
    # is, thus clockwise travel should be pi/2 while anticlockwise should be 3/2 pi.
    # Also: take a mod
    # alphas = (phis_path + center_angle) % mod
    # alphas = (phis_diff + np.pi / 2) % mod
    alphas = (phis_path + thetas_angle - np.pi) % mod if thetas is not None else (phis_diff + np.pi / 2) % mod
    
    # print(f'\N{greek small letter phi}: {np.min(phis_diff):.2f} to {np.max(phis_diff):.2f}; '\
    #       f'\N{greek small letter alpha}: {np.min(alphas):.2f} to {np.max(alphas):.2f}')

    return (
        np.mean(phis_diff),
        np.std(phis_diff, ddof=1),
        np.mean(alphas),
        np.std(alphas, ddof=1),
        phis_diff,
        alphas,
    )


def get_scale(file_path):
    """Determine the size of a single pixel in the input image."""
    file = Dataset(file_path, "r")

    full_scale = np.abs(file.variables["vertical_full_scale"][...].data)
    magnification = np.abs(file.variables["magnification"][...].data)
    dim = file.variables["image_data"].shape[0]

    adjusted_scale = full_scale / magnification / dim

    return adjusted_scale, magnification


def image_data(file_path):
    """Read a file in NetCDF format and return the image data and axis."""
    file = Dataset(file_path, "r")

    return file.variables["image_data"][...].data.T, str(file.variables["channel_name"][...].data[1], "utf-8")


def import_files(name, runs, indir):
    # Read in image files. Return a dictionary of image data
    # {im1: [], im2: [], m1: [], m2: [], m3: [], m4: [], scale: scale}

    image_dict = {}

    print(f"Searching {name}* for runs {runs}...")

    # Get all the i* and m* files. Should end up with 8 of them.

    files = []

    for run in runs:
        full_name = name + f"{run:0>3}"
        file = indir + full_name

        for f in ["x", "y", "z"]:
            files.extend(glob.glob(file + "*" + f + "*" + sempa_file_suffix))

    files = np.array(sorted(files), dtype="object")
    fnames = [f.split("/")[-1] for f in files]
    print(f"Found files:\n{fnames}")

    # ix, ix2, iy, iz
    # ix and iy are the same; similarly, ix2 and iz are the same. Ignore 2.
    # The two images will be used to align the data

    i_extensions = ["ix.", "ix2"]

    for i in range(len(i_extensions)):
        key = "i{}".format(i + 1)
        im, _ = image_data(file_for(files, i_extensions[i]))
        im_blurred = median_filter(im, 3)

        image_dict[key] = [im, im_blurred, im - im_blurred]

    # mx, mx2, my, mz
    # Stick with 1, 2, 3 for x, y, z; 4 will be the second x for data scaling
    extensions = ["mx.", "my", "mz", "mx2"]

    for i in range(len(extensions)):
        key = "m{}".format(i + 1)
        m, ax = image_data(file_for(files, extensions[i]))

        image_dict[key] = [m, ax]
        image_dict[key].extend(m.shape)
        image_dict[key].extend([m.min(), m.max()])

    scale, magnification = get_scale(files[0])
    image_dict["scale"] = scale
    image_dict["magnification"] = magnification

    return image_dict


def is_contour_circular(contour, threshold=0.2):
    """Determine if the area of the contour is circular

    Args:
        contour (contour): The contour to examine
        threshold (float): Acceptable eccentricity; 0 = exact circle

    Returns:
        boolean: Is it a circle (within threshold)?
    """
    width, height = cv2.fitEllipse(contour)[1]
    avg_radius = (width + height) / 4
    area = cv2.contourArea(contour)
    expected_area = np.pi * avg_radius * avg_radius

    diff_areas = np.abs(expected_area - area)

    return diff_areas < area * threshold


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


def normalize_3d(x, y, z):
    """Perform element-wise normalization of 3 arrays"""

    # First square the arrays, then perform element-wise sum
    x2, y2, z2 = x**2, y**2, z**2
    sum_xyz2 = np.sum((x2, y2, z2), axis=0)

    # Return the square root of each array divided by the sum array
    return (
        np.sqrt(np.abs(x2 / sum_xyz2)) * np.sign(x),
        np.sqrt(np.abs(y2 / sum_xyz2)) * np.sign(y),
        np.sqrt(np.abs(z2 / sum_xyz2)) * np.sign(z) * -1,
    )


def process_ltem_data(ltem_data_path):
    """Return data for .npy or .npz formatted LTEM data"""

    zfile = ltem_data_path[-1] == "z"
    # Data format is array of complex numbers, real = x, imaginary = y
    ltem_data = np.load(ltem_data_path)
    ltem_data_xy = ltem_data if not zfile else (ltem_data["Bx"] + 1j * ltem_data["By"])

    # Determine pixel size
    xerror = 0 if not zfile else (ltem_data["x"][1] - ltem_data["x"][0])  # * scale_multiplier
    yerror = 0 if not zfile else (ltem_data["y"][1] - ltem_data["y"][0])  # * scale_multiplier

    # Get phases and magnitudes
    ltem_phases, _ = get_phases(np.real(ltem_data_xy), np.imag(ltem_data_xy))

    ltem_magnitudes = get_magnitudes(np.real(ltem_data_xy), np.imag(ltem_data_xy))

    # ltem_circular_contours, ltem_contours_g, _ = find_circular_contours(
    #     ltem_magnitudes > np.average(ltem_magnitudes))

    # Find contours; throw out any that too short or too long; scale them up to move into
    # the domain wall; get rid of contours that go off the edge of the image
    ltem_height, ltem_width = ltem_magnitudes.shape
    size = ltem_width

    ltem_circular_contours, ltem_contours_g, _ = find_circular_contours(ltem_magnitudes, ltem=True)
    ltem_circular_contours = [c for c in ltem_circular_contours if len(c) > 0.05 * size and len(c) < 0.4 * size]
    ltem_circular_contours = [scale_contour(c, 1.3) for c in ltem_circular_contours]
    ltem_circular_contours = [c for c in ltem_circular_contours if np.min(c) >= 0 and np.max(c) < size]

    # ltem_alphas = np.asarray([get_phi_diff(np.squeeze(c), ltem_phases)
    #                           for c in ltem_circular_contours])

    return (
        ltem_phases,
        ltem_magnitudes,
        ltem_circular_contours,
        ltem_contours_g,
        xerror,
        yerror,
    )


def ptp_magnitudes(variables, image_1, image_2, image_3=None):
    """Calculate the min to max spreads for magnitudes; used for optimizing the offset."""

    # Allow for 2D processing
    var_3 = variables[2] if image_3 is not None else 0
    arg_3 = image_3 - variables[2] if image_3 is not None else np.zeros_like(image_1)

    return np.ptp(get_magnitudes(image_1 - variables[0], image_2 - variables[1], arg_3 - var_3))


def std_magnitudes(variables, image_1, image_2, image_3=None):
    """Calculate the min to max spreads for magnitudes; used for optimizing the offset."""

    # Allow for 2D processing
    var_3 = variables[2] if image_3 is not None else 0
    arg_3 = image_3 - variables[2] if image_3 is not None else np.zeros_like(image_1)

    return np.std(get_magnitudes(image_1 - variables[0], image_2 - variables[1], arg_3 - var_3))


def remove_line_errors(image, lines, use_rows=True):
    """Remove line errors using either rows or columns."""
    image_base = image if use_rows else image.T

    image_mean = np.asarray([np.mean(image_base[:lines, x]) for x in range(image.shape[1])]).T

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
            # im_srgb[:, :, i], rescale_to(magnitudes, [0.3, 1]))
            im_srgb[:, :, i],
            rescale_to(magnitudes, [0, 1]),
        )

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


def save_file(
    file_path,
    im_image,
    img,
    axis_1,
    axis_2,
    scale,
    img_denoised_1=None,
    img_denoised_2=None,
    arrow_color=None,
    arrow_scale=None,
    figsize=(3, 3),
    dpi=300,
):
    """Save images to PNG files."""

    # First save intensity image
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(im_image, cmap="gray")
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.savefig(file_path + "_" + axis_1 + axis_2 + "_im.png")
    plt.close()

    # Now save magnetization images
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.savefig(file_path + "_" + axis_1 + axis_2 + ".png")

    ax.add_artist(ScaleBar(scale, box_alpha=0.8))
    plt.savefig(file_path + "_" + axis_1 + axis_2 + "_scale.png")

    if img_denoised_1 is not None:
        show_vector_plot(img_denoised_1, img_denoised_2, ax=ax, color=arrow_color, scale=arrow_scale)
        plt.savefig(file_path + "_" + axis_1 + axis_2 + "_scale_arrows.png")

    plt.close()


def save_intensity_images(path, figsize=(3, 3), dpi=300):
    """Render and save all intensity images in path"""
    for file in glob.iglob(path + "*ix*" + sempa_file_suffix):
        im_image = image_data(file)

        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(im_image, cmap="gray")
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        plt.savefig(path + file + "_im.png")
        plt.close()


def scale_contour(cnt, scale, center_of=None):
    """Scale a contour.

    Args:
        cnt (list of points): Contour to be scaled
        scale (float): Scale factor
        center_of (image): Optional image with origin point

    Returns:
        list of points: The scaled contour
    """
    if center_of is None:
        # Find the centroid
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = find_minimum_in_contour(cnt, center_of)

    # Translate the points, scale, translate back
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    # Remove dupes
    # print(cnt_scaled)
    if scale < 1:
        cnt1 = [cnt_scaled[0]]
        # print(f'Start: {cnt1} from {cnt_scaled[0]} from shape {cnt_scaled.shape}')
        for i in range(1, len(cnt_scaled)):
            # print(f'{i}: {cnt_scaled[i]}')
            current = cnt_scaled[i]
            last = cnt1[-1]
            if not (current[0][0] == last[0][0] and current[0][1] == last[0][1]):
                cnt1.append(current)
                # print(cnt1)
    else:
        cnt1 = cnt_scaled

    return cnt1


def segment_image(image, segments=1, adaptive=False, sel=circle_array(1), erode_iter=1, close_iter=7):
    """Segment image into a defined number of objects. Create positive and negative masks for those objects."""
    # Rescale to 0-255 for thresholding
    image_shift = np.asarray(rescale_to_8_bit(image))
    _, image_thresh = cv2.threshold(image_shift, image_shift.mean(), 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Alternative thresholding mechanism
    # m_1_thresh = cv2.adaptiveThreshold(m_1_shift, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 0)
    # m_1_shift = cv2.cvtColor(m_1_shift, cv2.COLOR_GRAY2BGR)

    # Perform morphological operations. First kill small noise, then close the image
    image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_ERODE, sel, iterations=erode_iter)
    image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, sel, iterations=close_iter)

    # Find the segment contours
    snakes = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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


def show_all_circles(
    img,
    contours,
    phis=None,
    thetas=None,
    ax=None,
    show_numbers=True,
    reverse=False,
    alpha=1,
    color="black",
    show_axes=True,
    origin="upper",
):
    """Plot all circular contours."""

    if reverse:
        contours = contours[::-1]

    # Convert img to color and draw the contours
    c_img = cv2.cvtColor(np.zeros_like(img, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    cv2.drawContours(c_img, contours, -1, (255, 255, 255), 1)

    # Convert resultant image back to grayscale to produce a mask
    c_img_gray = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    masked_circles = np.ma.masked_where(c_img_gray == 0, c_img_gray)
    # plt.imshow(~masked_circles.mask)
    # print(~masked_circles.mask)

    # Use the mask to produce an RBGA image (transparent background)
    c_img_rgba = cv2.cvtColor(c_img, cv2.COLOR_RGB2RGBA)
    c_img_rgba[:, :, 3] = masked_circles

    boxes, centers = get_bounding_boxes(contours)

    if ax is not None:
        cmap_to_show = cm.binary_r if color == "black" else cm.binary

        if phis is None:
            ax.imshow(
                masked_circles,
                interpolation="none",
                alpha=alpha,
                cmap=cmap_to_show,
                zorder=100,
                origin=origin,
            )
        else:
            for index in range(len(boxes)):
                contour = contours[index]
                path = np.squeeze(contour)
                _, _, alpha_avg, _, _, _ = get_phi_diff(path, phis, thetas, origin == "upper")
                chirality = np.pi / 2 if alpha_avg < np.pi else 3 * np.pi / 2
                m = get_contour_mask(contour, phis)
                mask = np.ma.masked_where(m == 0, m) / 255 * chirality

                ax.imshow(
                    mask, interpolation="none", alpha=alpha, origin=origin, vmin=0, vmax=2 * np.pi, cmap="plasma_r"
                )

        if show_numbers:
            for index in range(len(boxes)):
                ax.text(
                    centers[index][0],
                    centers[index][1],
                    index + 1,
                    alpha=alpha,
                    color=color,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    return masked_circles, c_img_rgba, boxes


def show_alphas(
    contours_g,
    circular_contours,
    phis,
    thetas,
    name,
    show_circles_separately=True,
    bins=20,
    ltem_data=None,
    show_title=True,
):
    all_alphas = np.asarray([get_phi_diff(np.squeeze(c), phis, thetas) for c in contours_g])

    circle_alphas = np.asarray([get_phi_diff(np.squeeze(c), phis, thetas) for c in circular_contours])

    # Define color maps
    colors = {"all": "blue", "circles": "orange", "ltem": "green"}
    all_hist_type = "step" if show_circles_separately else "stepfilled"

    # Create a simulated phi distribution for LTEM if not provided
    p = [0.0, 0.1, 0.75, 0.1, 0.0, 0.01, 0.03, 0.01, 0]
    ltem_data = (
        np.random.choice(np.arange(0, 2.25 * np.pi, np.pi / 4), len(all_alphas), p=p)
        if ltem_data is None
        else ltem_data
    )

    plt.figure(figsize=(10, 5))
    hist_all = plt.hist(
        all_alphas[:, 2],
        bins=bins,
        color=colors["all"],
        alpha=0.5,
        histtype=all_hist_type,
        density=True,
        label="SEMPA",
    )
    if show_circles_separately:
        hist_circle = plt.hist(
            circle_alphas[:, 2],
            bins=bins,
            color=colors["circles"],
            alpha=0.5,
            density=True,
            label="SEMPA (circular)",
        )
    hist_ltem = plt.hist(
        ltem_data,
        bins=bins,
        color=colors["ltem"],
        alpha=0.5,
        density=True,
        label="LTEM",
    )

    # Find the indices of the maxima for all and LTEM
    lbin_all = np.argmax(hist_all[0])
    lbin_ltem = np.argmax(hist_ltem[0])

    # Circular is a bit more complicated since it seems to have 2 big humps.
    # First, round all values to 4 places, select unique values, and sort
    # descending, keeping the first 2 value (top 2). Then find the indices
    # for those values.
    if show_circles_separately:
        tops_circle = sorted(np.unique(np.around(hist_circle[0], 4)), reverse=True)[:2]
        lbin_circle = [np.where(np.around(hist_circle[0], 4) == t) for t in tops_circle]
    else:
        tops_circle = sorted(np.unique(np.around(hist_all[0], 0)), reverse=True)[:2]
        lbin_circle = [np.where(np.around(hist_all[0], 0) == t) for t in tops_circle]

    # Now find the modes
    mode_all = (hist_all[1][lbin_all + 1] + hist_all[1][lbin_all]) / 2
    mode_ltem = (hist_ltem[1][lbin_ltem + 1] + hist_ltem[1][lbin_ltem]) / 2

    plt.axvline(mode_all, linestyle="dashed", linewidth=1, color=colors["all"])
    plt.axvline(mode_ltem, linestyle="dashed", linewidth=1, color=colors["ltem"])

    use_hist = hist_all if not show_circles_separately else hist_circle
    if show_circles_separately:
        mode_circle = (hist_circle[1][lbin_circle[0][0][-1] + 1] + hist_circle[1][lbin_circle[0][0][0]]) / 2
        mode_circle_2 = (hist_circle[1][lbin_circle[1][0][-1] + 1] + hist_circle[1][lbin_circle[1][0][-1]]) / 2
        # mode_circle = (use_hist[1][lbin_circle[0][0][-1] + 1] +
        #                use_hist[1][lbin_circle[0][0][0]]) / 2
        # mode_circle_2 = (use_hist[1][lbin_circle[1][0][-1] + 1] +
        #                  use_hist[1][lbin_circle[1][0][-1]]) / 2

        plt.axvline(mode_circle, linestyle="dashed", linewidth=1, color=colors["circles"])
        plt.axvline(mode_circle_2, linestyle="dashed", linewidth=1, color=colors["circles"])

    # Place text labels for the modes
    _, y_lim = plt.ylim()
    shift_x = 1.05
    t = plt.text(mode_all * shift_x, y_lim * 0.7, f"{mode_all:.2f}")
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor=colors["all"]))

    if show_circles_separately:
        t = plt.text(mode_circle * shift_x, y_lim * 0.9, f"{mode_circle:.2f}")
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor=colors["circles"]))
        t = plt.text(mode_circle_2 * shift_x, y_lim * 0.85, f"{mode_circle_2:.2f}")
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor=colors["circles"]))

    t = plt.text(mode_ltem * shift_x, y_lim * 0.7, f"{mode_ltem:.2f}")
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor=colors["ltem"]))

    plt.xlim(0, 2 * np.pi)

    tick_labels = [
        "0",
        r"$\dfrac{\pi}{4}$",
        r"$\dfrac{\pi}{2}$",
        r"$\dfrac{3\pi}{4}$",
        r"$\pi$",
        r"$\dfrac{5\pi}{4}$",
        r"$\dfrac{3\pi}{2}$",
        r"$\dfrac{7\pi}{4}$",
        r"$2\pi$",
    ]
    plt.xticks(np.arange(0, 2.25 * np.pi, step=np.pi / 4), tick_labels)
    plt.yticks([])

    plt.xlabel(r"$\alpha$ [Radians]")
    if show_title:
        plt.title(rf"Average $\alpha$ for {name}")

    plt.legend()

    return plt, all_alphas, circle_alphas


def show_alphas_2(
    alphas_candidates,
    alphas_all,
    alphas_ltem,
    name=r"$\alpha$",
    show_all=True,
    show_title=True,
):
    plt.figure(figsize=((10, 5)))

    c_hist_arr, c_hist_bins, _ = plt.hist(alphas_candidates, alpha=0.7, bins=30, density=True, zorder=10, label="SEMPA")
    if show_all:
        a_hist_arr, a_hist_bins, _ = plt.hist(
            alphas_all,
            histtype="step",
            density=True,
            zorder=2,
            alpha=0.6,
            color="green",
            label="All SEMPA",
        )
    l_hist_arr, l_hist_bins, _ = plt.hist(alphas_ltem, density=True, zorder=1, bins=30, alpha=0.3, label="LTEM")

    c_bin_width = c_hist_bins[1] - c_hist_bins[0]
    c_peak_indices = np.array(find_peaks(np.concatenate(([0], c_hist_arr, [0])), distance=len(c_hist_arr) // 2)[0]) - 1
    c_max_bin_centers = [c_hist_bins[x] + c_bin_width / 2 for x in c_peak_indices]

    if show_all:
        a_bin_width = a_hist_bins[1] - a_hist_bins[0]
        a_peak_indices = (
            np.array(
                find_peaks(
                    np.concatenate(([0], a_hist_arr, [0])),
                    distance=len(a_hist_arr) // 2,
                )[0]
            )
            - 1
        )
        a_max_bin_centers = [a_hist_bins[x] + a_bin_width / 2 for x in a_peak_indices]

    l_bin_width = l_hist_bins[1] - l_hist_bins[0]
    l_peak_indices = np.array(find_peaks(np.concatenate(([0], l_hist_arr, [0])), distance=len(l_hist_arr) // 2)[0]) - 1
    l_max_bin_centers = [l_hist_bins[x] + l_bin_width / 2 for x in l_peak_indices]

    _, y_lim = plt.ylim()
    shift_x_pos = 0
    shift_x_neg = 0.35
    colors = {"circles": "blue", "all": "green", "ltem": "orange"}
    c_y_adjust, a_y_adjust, l_y_adjust = 0.87, 0.7, 0.53
    box_alpha = 0.7
    zline, zbox = 20, 30
    new_line = "\n"

    for c in c_max_bin_centers:
        plt.axvline(c, color=colors["circles"], linestyle="dotted", zorder=zline)
        t = plt.text(
            c + shift_x_pos,
            y_lim * c_y_adjust,
            rf"{c:.2f}{new_line}$\pm${c_bin_width / 2:.2f}",
            zorder=zbox,
        )
        t.set_bbox(
            dict(
                facecolor="white",
                alpha=box_alpha,
                zorder=20,
                edgecolor=colors["circles"],
            )
        )

    if show_all:
        for a in a_max_bin_centers:
            plt.axvline(a, color=colors["all"], linestyle="dotted", zorder=zline)
            t = plt.text(
                a - shift_x_neg,
                y_lim * a_y_adjust,
                f"{a:.2f}{new_line}$\pm${a_bin_width / 2:.2f}",
                zorder=zbox,
            )
            t.set_bbox(
                dict(
                    facecolor="white",
                    alpha=box_alpha,
                    zorder=20,
                    edgecolor=colors["all"],
                )
            )

    for l in l_max_bin_centers:
        plt.axvline(
            l,
            color=colors["ltem"] if show_all else colors["all"],
            linestyle="dotted",
            zorder=zline,
        )
        t = plt.text(
            l - shift_x_neg,
            y_lim * (l_y_adjust if show_all else a_y_adjust),
            f"{l:.2f}{new_line}$\pm${l_bin_width / 2:.2f}",
            zorder=zbox,
        )
        t.set_bbox(
            dict(
                facecolor="white",
                alpha=box_alpha,
                zorder=20,
                edgecolor=colors["ltem"] if show_all else colors["all"],
            )
        )

    plt.xlabel(r"$\alpha$ [Radians]")
    plt.xlim((0, 2 * np.pi))
    tick_labels = [
        "0",
        r"$\dfrac{\pi}{4}$",
        r"$\dfrac{\pi}{2}$",
        r"$\dfrac{3\pi}{4}$",
        r"$\pi$",
        r"$\dfrac{5\pi}{4}$",
        r"$\dfrac{3\pi}{2}$",
        r"$\dfrac{7\pi}{4}$",
        r"$2\pi$",
    ]
    plt.xticks(np.arange(0, 2.25 * np.pi, step=np.pi / 4), tick_labels)
    plt.yticks([])
    plt.legend()

    if show_title:
        plt.title = name


def show_alphas_3(features):
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))

    fig = plt.figure(figsize=(ncols * 6, nrows * 6));
    spec = gs.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    row, col = 0, 0

    for i in features.keys():
        c = features[i]
        alphas = c['alphas']
        alphas_1_3 = c['alphas_1_3']
        alphas_2_3 = c['alphas_2_3']
        alpha_avg = c['alpha_avg']
        alpha_avg_1_3 = c['alpha_avg_1_3']
        alpha_avg_2_3 = c['alpha_avg_2_3']

        length = len(alphas)
        xs = np.arange(length)
        xs_1 = np.linspace(0, length, len(alphas_1_3))
        xs_2 = np.linspace(0, length, len(alphas_2_3))

        ax = fig.add_subplot(spec[row, col])
        ax.scatter(xs, alphas, label='Wall');
        ax.scatter(xs_1, alphas_1_3, label='1/3');
        ax.scatter(xs_2, alphas_2_3, label='2/3');
        ax.axhline(np.pi, color="gray", ls="-", alpha=0.1)
        ax.axhline(
            alpha_avg,
            color="blue",
            ls="--",
            label=r"Avg $\alpha$",
            alpha=0.5
        )
        # ax.axhline(alpha_avg + mae, color="green", ls="-.", alpha=0.2)
        # ax.axhline(alpha_avg - mae, color="green", ls="-.", alpha=0.2)
        ax.axhline(np.pi, color="gray", ls="-", alpha=0.1)
        ax.axhline(
            alpha_avg_1_3,
            ls="--",
            color="orange",
            label=r"Avg $\alpha$ at $\frac{1}{3}$",
        )
        ax.axhline(
            alpha_avg_2_3,
            ls="--",
            color="green",
            label=r"Avg $\alpha$ at $\frac{2}{3}$",
        )

        ax.set_ylim((0, 2 * np.pi))
        ax.set_xticks([])
        ax.set_yticks([0, np.pi / 2, np.pi, 3 / 2 * np.pi, 2 * np.pi])
        ax.set_yticklabels(
            [
                "0",
                r"$\dfrac{\pi}{2}$",
                r"$\pi$",
                r"$\dfrac{3\pi}{2}$",
                r"$2\pi$",
            ]
        )
        ax.set_title(f'{i}: {alpha_avg_1_3:.2f}, {alpha_avg_2_3:.2f}, {alpha_avg:.2f}')
        ax.legend(ncol=2);

        col += 1
        if col == ncols:
            row += 1
            col = 0
            

def show_antiskyrmions(contours, antiskyrmions, magnitudes, phis, thetas, scale, m_1_adjusted, m_2_adjusted, m_3_adjusted):
    contours_to_use = [contours[i] for i in antiskyrmions.keys()]
    
    show_circles(
        magnitudes,
        phis,
        thetas,
        contours_to_use,
        scale,
        m_1_adjusted,
        m_2_adjusted,
        m_3_adjusted,
        normalize=False,
        show="thetas",
        show_numbers=True,
        show_axes=True,
        show_title=True,
        just_candidates=False,
        show_anyway=True,
        show_limit=100,
    )

    tick_labels = [
        "0",
        r"$\dfrac{\pi}{4}$",
        r"$\dfrac{\pi}{2}$",
        r"$\dfrac{3\pi}{4}$",
        r"$\pi$",
        r"$\dfrac{5\pi}{4}$",
        r"$\dfrac{3\pi}{2}$",
        r"$\dfrac{7\pi}{4}$",
        r"$2\pi$",
    ]
    ncols = 4
    nrows = int(np.ceil(len(antiskyrmions) / ncols))
    psize = 6
    row, col = 0, 0

    fig = plt.figure(figsize=(ncols * psize, nrows * psize))
    spec = gs.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    for i in antiskyrmions.keys():
        ants = antiskyrmions[i]

        ax = fig.add_subplot(spec[row, col])
        ax.autoscale()
        values, bins, bars = ax.hist(ants['alphas'], bins=16, alpha=0.8, color='gray', edgecolor='white')

        ax.axvline(np.pi, c='gray', alpha=0.5, zorder=10)
        
        fill_alpha = 0.2
        ax.fill_between(np.linspace(0, np.pi / 2, endpoint=True), max(values), color='red', zorder=0, alpha=fill_alpha)
        ax.fill_between(np.linspace(np.pi / 2, 3/2 * np.pi, endpoint=True), max(values), color='blue', zorder=0, alpha=fill_alpha)
        ax.fill_between(np.linspace(3/2 * np.pi, 2 * np.pi, endpoint=True), max(values), color='red', zorder=0, alpha=fill_alpha)
        
        # for j in range(len(values)):
        #     b = bins[j]
        #     c = 'blue' if np.pi / 2 < b < 3/2 * np.pi else 'red'
        #     bars[j].set_facecolor(c)
        
        c = 'black'
        t = ax.text(0, max(values), 'In', fontdict={"fontsize": 20}, c='red', horizontalalignment="left", verticalalignment="top")
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor=c))
        t = ax.text(np.pi, max(values), 'Out', fontdict={"fontsize": 20}, c='blue', horizontalalignment="center", verticalalignment="top", zorder=20)
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor=c))
        t = ax.text(1.9 * np.pi, max(values), 'In', fontdict={"fontsize": 20}, c='red', horizontalalignment="right", verticalalignment="top")
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor=c))

        ax.set_xticks(np.arange(0, 2.25 * np.pi, step=np.pi / 4), tick_labels)
        ax.set_title(f'{i}: n = {ants["winding_number"]:.2f}')
        
        col += 2
        if col == ncols:
            row += 1
            col = 0


def show_circles(
    magnitudes,
    phis,
    thetas,
    contours,
    scale,
    m_1=None,
    m_2=None,
    m_3=None,
    show_numbers=True,
    reverse=False,
    alpha=0.5,
    show="thetas",
    normalize=True,
    show_both=False,
    phis_m=None,
    candidate_cutoff=np.pi / 6,
    show_title=True,
    show_axes=True,
    just_candidates=True,
    origin="upper",
    show_limit=30,
    show_anyway=False,
    show_plot=True,
    show_alphas=True,
    alpha_limit=2 * np.pi,
):
    """Plot all circular contours."""
    _, _, boxes = show_all_circles(thetas, contours, reverse=reverse, origin=origin)
    count = len(boxes)
    print(f"Found {count} features.")

    yd, xd = magnitudes.shape

    extent_factor = 0.3

    # Track candidates. Cutoff of pi/6 allows for 3 sigma.
    candidates = {}
    all_contours = {}
    c_index, displayed = 0, 0
    cw, ccw = 0, 0
    # new_line = "\n"
    contours_to_plot = []

    while c_index < len(boxes) and displayed < show_limit:
        box = boxes[c_index]

        xmin, ymin, width, height = box
        width_nm = int(width * scale * scale_multiplier)
        height_nm = int(height * scale * scale_multiplier)

        # Select the area to evaluate
        extent_x, extent_y = int(extent_factor * width), int(extent_factor * height)
        extent = np.min([extent_x, extent_y])
        xmin, ymin = np.max([xmin - extent, 0]), np.max([ymin - extent, 0])
        xmax = np.min([xmin + width + 2 * extent, xd])
        ymax = np.min([ymin + height + 2 * extent, yd])
        box_xd, box_yd = xmax - xmin, ymax - ymin
        y_s, x_s = slice(ymin, ymax), slice(xmin, xmax)
        xy_s = (y_s, x_s)

        # Walk along the path of the contour and measure deviations from it
        path = contours[c_index]

        # Determine winding number
        if m_1 is not None:
            winding_number = calculate_winding_number(m_1, m_2, m_3, xy_s, path, wider_contour=False)
        else:
            winding_number = None

        # Scale the path to 1/3 and 2/3 for further alpha evaluation
        path_1_3 = np.squeeze(scale_contour(path, 1 / 3, thetas))
        path_2_3 = np.squeeze(scale_contour(path, 2 / 3, thetas))
        path = np.squeeze(path)

        dev_avg, dev_std, alpha_avg, alpha_std, _, alphas = get_phi_diff(path, phis, thetas, alpha_limit, origin == "upper")
        _, _, alpha_avg_1_3, alpha_std_1_3, _, alphas_1_3 = get_phi_diff(path_1_3, phis, thetas, alpha_limit, origin == "upper")
        _, _, alpha_avg_2_3, alpha_std_2_3, _, alphas_2_3 = get_phi_diff(path_2_3, phis, thetas, alpha_limit, origin == "upper")

        mae = np.abs(alphas - alpha_avg).mean()
        sem = alpha_std / np.sqrt(len(alphas))
        chirality = np.pi / 2 if alpha_avg < np.pi else 3 * np.pi / 2

        fake_news = np.ptp(alphas) >= np.pi if not show_anyway else False

        candidate = alpha_std <= candidate_cutoff and not fake_news and is_contour_circular(path)

        if thetas is not None:
            cx, cy = find_minimum_in_contour(path, thetas)
            cx -= xmin
            cy -= ymin
        else:
            cx, cy = int(box_xd / 2), int(box_yd / 2)

        if alpha_avg < np.pi:
            cw += 1
        else:
            ccw += 1

        all_contours[c_index] = {
            "path": path,
            "path_1_3": path_1_3,
            "path_2_3": path_2_3,
            "dev_avg": dev_avg,
            "dev_std": dev_std,
            "alpha_avg": alpha_avg,
            "alpha_std": alpha_std,
            "mae": mae,
            "sem": sem,
            "width_nm": width_nm,
            "height_nm": height_nm,
            "alphas": alphas,
            "alphas_1_3": alphas_1_3,
            "alpha_avg_1_3": alpha_avg_1_3,
            "alphas_2_3": alphas_2_3,
            "alpha_avg_2_3": alpha_avg_2_3,
            "winding_number": winding_number,
            "chirality": chirality,
            "cx": cx,
            "cy": cy,
            "xy_s": xy_s,
            "width": width,
            "height": height,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "box_xd": box_xd,
            "box_yd": box_yd,
            "candidate": candidate,
        }

        if candidate:
            candidates[c_index] = all_contours[c_index]

        if not fake_news and (candidate or not just_candidates):
            displayed += 1
            contours_to_plot.append(c_index)

        c_index += 1

    number_of_plots = np.min((show_limit, c_index))
    spurious = f" ({number_of_plots - cw - ccw} spurious contours)" if number_of_plots - cw - ccw > 0 else ""
    print(f"Found {cw} clockwise and {ccw} anticlockwise contours{spurious}")
    
    if show_plot:
        show_circles_plot(
            displayed,
            all_contours,
            contours_to_plot,
            magnitudes,
            phis,
            thetas,
            scale,
            show=show,
            normalize=normalize,
            origin=origin,
            show_title=show_title,
            show_numbers=show_numbers,
            show_axes=show_axes,
            show_alphas=show_alphas,
            alpha_limit=alpha_limit,
        )

    return candidates, all_contours


def show_circles_plot(
    displayed,
    all_contours,
    contours_to_plot,
    magnitudes,
    phis,
    thetas,
    scale,
    show="thetas",
    normalize=True,
    origin="upper",
    show_title=True,
    show_numbers=True,
    show_axes=True,
    show_alphas=True,
    alpha_limit=2 * np.pi,
):
    ncols = 4
    column_width = 6
    nrows = np.ceil(2 * displayed / ncols).astype(int)
    row, col = 0, 0

    fig = plt.figure(figsize=(ncols * column_width, nrows * column_width))
    spec = gs.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    for c_index in contours_to_plot:
        item = all_contours[c_index]
        path = item["path"]
        path_1_3 = item["path_1_3"]
        path_2_3 = item["path_2_3"]
        alphas = item["alphas"]
        alphas_1_3 = item["alphas_1_3"]
        alphas_2_3 = item["alphas_2_3"]
        dev_avg = item["dev_avg"]
        dev_std = item["dev_std"]
        alpha_avg = item["alpha_avg"]
        alpha_avg_1_3 = item["alpha_avg_1_3"]
        alpha_avg_2_3 = item["alpha_avg_2_3"]
        alpha_std = item["alpha_std"]
        candidate = item["candidate"]

        xy_s = item["xy_s"]
        xmin, xmax, ymin, ymax = item["xmin"], item["xmax"], item["ymin"], item["ymax"]
        width, height = item["width"], item["height"]
        box_xd, box_yd = item["box_xd"], item["box_yd"]
        winding_number = item["winding_number"]
        chirality = item["chirality"]
        cx, cy = item["cx"], item["cy"]
        width_nm, height_nm = item["width_nm"], item["height_nm"]

        # Set the underlying image and pick a color map
        m_subset = magnitudes[xy_s]
        p_subset = phis[xy_s]
        t_subset = thetas[xy_s]

        im_to_show = p_subset if show == "phis" else t_subset
        cmap_to_show = ciecam02_cmap() if show == "phis" else cm.coolwarm_r
        cmap_max = 2 * np.pi if show == "phis" else np.pi

        # Create a pair of (x, y) coordinates
        x = np.linspace(1, box_xd - 2, box_xd, dtype=np.uint8)
        y = np.linspace(1, box_yd - 2, box_yd, dtype=np.uint8)
        X, Y = np.meshgrid(x, y)

        # Create vector lengths
        U = np.ones_like(m_subset) if normalize else m_subset * np.cos(p_subset) * np.sin(t_subset)
        V = np.ones_like(m_subset) if normalize else m_subset * np.sin(p_subset) * np.sin(t_subset)

        # Create the axes
        ax = fig.add_subplot(spec[row, col])
        ax.autoscale()
        col += 1

        # Show the feature and path
        ax.imshow(im_to_show, cmap=cmap_to_show, vmin=0, vmax=cmap_max, origin=origin)
        # m = get_contour_mask(path, m_1, xy_s)
        m = get_contour_mask(path, np.zeros_like(phis), xy_s)
        mask = np.ma.masked_where(m == 0, m) / 255 * chirality
        ax.imshow(mask, interpolation="none", alpha=0.7, origin=origin, vmin=0, vmax=2 * np.pi, cmap="plasma_r")
        ax.scatter(path_1_3[:, 0] - xmin + 1, path_1_3[:, 1] - ymin, color="yellow", alpha=0.3)
        ax.scatter(path_2_3[:, 0] - xmin + 1, path_2_3[:, 1] - ymin, color="green", alpha=0.3)
        ax.add_artist(ScaleBar(scale, box_alpha=0.8))
        ax.scatter(cx, cy, color="black")

        title = (
            f"({xmin}, {ymin}) to ({xmax}, {ymax}), "
            f"{width_nm}x{height_nm} nm\n"
            rf"$\alpha:\ {alpha_avg:.2f},\ \sigma_{{\alpha}}:\ {alpha_std:.2f}\quad $"
            rf"$\phi_{{dev}}:\ {dev_avg:.2f},\ \sigma_{{dev}}:\ {dev_std:.2f}$"
        )
        if winding_number:
            title += rf"$\quad n: {winding_number:.2f}$"

        ax.set_title(title if show_title else "")

        # Place 4 dots to show the direction around the path
        colors = itertools.cycle(["red", "yellow", "green", "blue"])
        num_pts = len(path)
        pts = [
            path[0],
            path[int(num_pts / 4)],
            path[int(num_pts / 2)],
            path[int(3 * num_pts / 4)],
        ]
        for point in pts:
            ax.scatter(point[0] - xmin, point[1] - ymin, c=next(colors))

        if show_numbers:
            c = "blue" if candidate else "red"
            label = rf"{c_index + 1}$\circlearrowright$" if alpha_avg < np.pi else rf"{c_index + 1}$\circlearrowleft$"

            t = ax.text(
                0,
                0,
                label,
                fontdict={"fontsize": 20},
                c=c,
                horizontalalignment="left",
                verticalalignment="top",
            )
            t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor=c))

        # Show the vectors
        ax.quiver(X, Y, U, V, angles=p_subset * 180 / np.pi, pivot="mid", units="dots")

        # Turn axis labels on or off
        if show_axes:
            x_span, y_span = width // 5, height // 5
            ax.set_xticks(x[::x_span])
            ax.set_xticklabels((x + xmin - 1).astype(int)[::x_span])
            ax.set_yticks(y[::y_span])
            ax.set_yticklabels((y + ymin - 1).astype(int)[::y_span])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        #
        #
        # -----------------------------------
        #
        # Show alphas
        if show_alphas:
            num_alphas = len(alphas)
            xs = np.linspace(1, num_alphas + 1, num_alphas)
            xs_1 = np.linspace(1, num_alphas + 1, len(alphas_1_3))
            xs_2 = np.linspace(1, num_alphas + 1, len(alphas_2_3))

            ax = fig.add_subplot(spec[row, col])
            ax.autoscale()
            col += 1

            ax.plot(xs, alphas, ".", label=r"$\alpha$")
            ax.plot(xs_1, alphas_1_3, ".", label=r"$\alpha$ at $\frac{1}{3}$")
            ax.plot(xs_2, alphas_2_3, ".", label=r"$\alpha$ at $\frac{2}{3}$")
            ax.fill_between(xs, alphas + alpha_std, alphas - alpha_std, alpha=0.1)
            
            ax.axhline(np.pi, color="gray", ls="-", alpha=0.1)
            
            ax.axhline(alpha_avg, color="blue", ls="--", label=r"Avg $\alpha$", alpha=0.5)
            ax.axhline(
                alpha_avg_1_3,
                ls="--",
                color="yellow",
                label=r"Avg $\alpha$ at $\frac{1}{3}$",
            )
            ax.axhline(
                alpha_avg_2_3,
                ls="--",
                color="green",
                label=r"Avg $\alpha$ at $\frac{2}{3}$",
            )
            
            ax.set_ylim((0, alpha_limit))
            ax.set_xticks([])
            ax.set_yticks(np.linspace(0, alpha_limit, 5, endpoint=True))
            # ax.set_yticks([0, np.pi / 2, np.pi, 3 / 2 * np.pi, 2 * np.pi])
            ax.set_yticklabels(
                [
                    "0",
                    r"$\dfrac{\pi}{2}$",
                    r"$\pi$",
                    r"$\dfrac{3\pi}{2}$",
                    r"$2\pi$",
                ]
            )

            ax.set_title(
                rf"$\alpha$: {alphas.min():.2f} to {alphas.max():.2f}, {alpha_avg:.2f} ({alpha_avg_2_3:.2f}, {alpha_avg_1_3:.2f}) average"
            )
            ax.legend(ncol=3)

        if col == ncols:
            row += 1
            col = 0

        # if show_both:
        #     p_m_subset = phis_m[xy_s]

        #     ax = plt.subplot(num_rows, num_columns, 2 * index)
        #     ax.autoscale()
        #     ax.imshow(p_m_subset, cmap=ciecam02_cmap())
        #     ax.imshow(
        #         masked_circles[ymin:ymax, xmin:xmax],
        #         interpolation="none",
        #         alpha=alpha,
        #         cmap=cm.binary,
        #     )
        #     ax.add_artist(ScaleBar(scale, box_alpha=0.8))

        #     if show_numbers:
        #         t = ax.text(
        #             0,
        #             0,
        #             index,
        #             fontdict={"fontsize": 20},
        #             c="black",
        #             horizontalalignment="left",
        #             verticalalignment="top",
        #         )
        #         t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="black"))

        #     ax.quiver(
        #         X,
        #         Y,
        #         U,
        #         V,
        #         units="dots",
        #         angles=p_subset * 180 / np.pi,
        #         color="white",
        #     )

        #     ax.set_title(title)
        #     span = 5
        #     ax.set_xticks(x[::span])
        #     ax.set_xticklabels((x + xmin - 1).astype(int)[::span])
        #     ax.set_yticks(y[::span])
        #     ax.set_yticklabels((y + ymin - 1).astype(int)[::span])


def show_contour_sizes(
    widths,
    heights,
    xerror,
    yerror,
    ltem_widths=None,
    ltem_heights=None,
    ltem_xerror=None,
    ltem_yerror=None,
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    if len(widths) > 0:
        ax.errorbar(
            widths,
            heights,
            xerror,
            yerror,
            fmt=".",
            ms=6,
            label=f"SEMPA (avg: ({int(np.mean(widths))}, {int(np.mean(heights))}))",
        )

    if ltem_widths is not None:
        ax.errorbar(
            ltem_widths,
            ltem_heights,
            ltem_xerror,
            ltem_yerror,
            fmt=".",
            ms=6,
            label=f"LTEM (avg: ({int(np.mean(ltem_widths))}, {int(np.mean(ltem_heights))}))",
        )

    line = np.min((ax.get_xlim(), ax.get_ylim())), np.max((ax.get_xlim(), ax.get_ylim()))
    ax.plot(line, line, linestyle="dashed", alpha=0.3)

    ax.set_aspect("equal")
    ax.set_xlabel("width [nm]")
    ax.set_ylabel("height [nm]")
    if ltem_widths is not None:
        plt.legend()


def show_contours_overview(
    thetas_flattened,
    phis_flattened,
    contours_g,
    circular_contours,
    magnitudes_2d_flattened,
    axis_1,
    axis_2,
    axis_3,
    scale,
):
    fig = plt.figure(figsize=(20, 20))

    ax1 = fig.add_subplot(221)
    ax1.imshow(thetas_flattened, cmap=cm.coolwarm_r)

    show_all_circles(thetas_flattened, contours_g, phis=phis_flattened, thetas=thetas_flattened, ax=ax1, alpha=0.7)

    sm = cm.ScalarMappable(cmap=cm.coolwarm)
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.3)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.set_yticklabels(["Down", f"{axis_1}-{axis_2}", "Up"])
    ax1.add_artist(ScaleBar(scale, box_alpha=0.8))
    ax1.set_title(rf"M$_{axis_3}$ magnetization")

    # ax2 = fig.add_subplot(222, polar=True)
    # show_phase_colors_circle(ax2, add_dark_background=False, text_color="black")

    ax2 = fig.add_subplot(222)
    ax2.imshow(thetas_flattened, cmap=cm.coolwarm_r)

    show_all_circles(
        thetas_flattened, circular_contours, phis=phis_flattened, thetas=thetas_flattened, ax=ax2, alpha=0.7
    )

    sm = cm.ScalarMappable(cmap=cm.coolwarm)
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.3)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.set_yticklabels(["Down", f"{axis_1}-{axis_2}", "Up"])
    ax2.add_artist(ScaleBar(scale, box_alpha=0.8))
    ax2.set_title(rf"M$_{axis_3}$ magnetization (circular)")

    ax3 = fig.add_subplot(223)
    ax3.imshow(phis_flattened, cmap=ciecam02_cmap(), alpha=0.7)
    show_all_circles(
        thetas_flattened, contours_g, phis=phis_flattened, thetas=thetas_flattened, ax=ax3, alpha=0.7, color="white"
    )
    ax32 = inset_axes(ax3, width="15%", height="15%", loc=4, axes_class=get_projection_class("polar"))
    show_phase_colors_circle(ax32, add_dark_background=False, show_angles=False)
    ax3.add_artist(ScaleBar(scale, box_alpha=0.8))
    ax3.set_title(rf"M$_{axis_1}$-M$_{axis_2}$ plane")

    ax4 = fig.add_subplot(224)
    phis_flattened_rgba = render_phases_and_magnitudes(phis_flattened, magnitudes_2d_flattened)
    ax4.imshow(phis_flattened_rgba, cmap=ciecam02_cmap())
    show_all_circles(thetas_flattened, contours_g, ax=ax4, alpha=0.5, color="white")

    sm = cm.ScalarMappable(cmap=cm.binary_r)
    cbar = fig.colorbar(sm, ax=ax4, shrink=0.3)
    cbar.set_ticks([0, 1])
    cbar.ax.set_yticklabels([f"All {axis_3}", f"All {axis_1}-{axis_2}"])
    ax4.add_artist(ScaleBar(scale, box_alpha=0.8))
    ax4.set_title(rf"M$_{axis_1}$-M$_{axis_2}$ plane with magnitude")


def show_groups(contours, magnitudes, phis, thetas, scale, normalize=True, origin="upper"):
    """Plot groups of circular contours"""
    groups, loners = find_group_contours(contours)

    group_count = len(groups)

    print(f"{group_count} groups:\n{groups}\n\n{len(loners)} loners:\n{loners}")
    contour_groups = [[np.concatenate([np.squeeze(c) for c in [contours[i] for i in g]])] for g in groups]

    boxes_g = [get_bounding_boxes(cg) for cg in contour_groups]

    col_width = 12
    num_cols = 1
    num_rows = np.ceil(group_count / num_cols).astype(int)

    plt.figure(figsize=(num_cols * col_width, num_rows * col_width))

    for index in range(group_count):
        box, group = boxes_g[index], groups[index]
        xmin, ymin, width, height = np.squeeze(box[0])
        xmax, ymax = xmin + width, ymin + height
        box_xd, box_yd = xmax - xmin, ymax - ymin
        y_s, x_s = slice(ymin, ymax), slice(xmin, xmax)
        xy_s = (y_s, x_s)

        _, centers = get_bounding_boxes([contours[i] for i in group])

        m_subset = magnitudes[ymin:ymax, xmin:xmax]
        p_subset = phis[ymin:ymax, xmin:xmax]
        t_subset = thetas[ymin:ymax, xmin:xmax]

        x = np.linspace(1, box_xd - 2, box_xd, dtype=np.uint8)
        y = np.linspace(1, box_yd - 2, box_yd, dtype=np.uint8)
        X, Y = np.meshgrid(x, y)

        # Create vector lengths
        U = np.ones_like(m_subset) if normalize else m_subset * np.cos(p_subset) * np.sin(t_subset)
        V = np.ones_like(m_subset) if normalize else m_subset * np.sin(p_subset) * np.sin(t_subset)

        # Plot the background
        ax = plt.subplot(num_rows, num_cols, index + 1)
        ax.imshow(
            thetas[ymin:ymax, xmin:xmax],
            cmap=cm.coolwarm_r,
            alpha=0.6,
            vmin=0,
            vmax=np.pi,
        )

        for contour_index in group:
            contour = contours[contour_index]
            path = np.squeeze(contour)
            _, _, alpha_avg, _, _, _ = get_phi_diff(path, phis, thetas, origin == "upper")
            chirality = np.pi / 2 if alpha_avg < np.pi else 3 * np.pi / 2
            m = get_contour_mask(contour, magnitudes, xy_s)
            mask = np.ma.masked_where(m == 0, m) / 255 * chirality

            ax.imshow(mask, interpolation="none", alpha=0.7, origin=origin, vmin=0, vmax=2 * np.pi, cmap="plasma_r")

        ax.quiver(X, Y, U, V, units="dots", angles=p_subset * 180 / np.pi, pivot="mid")

        for l_index in range(len(centers)):
            label = group[l_index] + 1
            t = ax.text(
                centers[l_index][0] - xmin,
                centers[l_index][1] - ymin,
                label,
                alpha=1,
                color="black",
                horizontalalignment="center",
                verticalalignment="center",
            )
            t.set_bbox(dict(facecolor="white", alpha=0.65, edgecolor="black"))

        span = 5
        ax.set_xticks(x[::span])
        ax.set_xticklabels((x + xmin - 1).astype(int)[::span])
        ax.set_yticks(y[::span])
        ax.set_yticklabels((y + ymin - 1).astype(int)[::span])
        ax.add_artist(ScaleBar(scale, box_alpha=0.8, location="lower right"))

        title = (
            f"({xmin}, {ymin}) to ({xmax}, {ymax}), "
            f"{int(width * scale * scale_multiplier)}x{int(height * scale * scale_multiplier)} nm\n"
        )
        ax.set_title(title)


def show_ltem_data(
    ltem_magnitudes,
    ltem_phases,
    ltem_contours,
    ltem_box_widths,
    ltem_xerror,
    ltem_yerror,
    name,
    use_cutoff=False,
    show_angles=True,
):
    ydim, xdim = ltem_magnitudes.shape
    cutoff = np.min((xdim, ydim)) // 32
    ltem_contours_reduced = (
        [c for c in ltem_contours if np.min(c) >= 512 and np.max(c) < 1024] if use_cutoff else ltem_contours
    )
    phases_reduced = ltem_phases[cutoff:-cutoff, cutoff:-cutoff] if use_cutoff else ltem_phases
    phases_reduced = ltem_phases[512:1024, 512:1024] if use_cutoff else ltem_phases
    magnitudes_reduced = ltem_magnitudes[cutoff:-cutoff, cutoff:-cutoff] if use_cutoff else ltem_magnitudes
    magnitudes_reduced = ltem_magnitudes[512:1024, 512:1024] if use_cutoff else ltem_magnitudes
    magnitudes_norm = magnitudes_reduced / magnitudes_reduced.max()

    fig = plt.figure(figsize=(24, 8))
    ax1 = plt.subplot(131)
    ax1.imshow(np.zeros_like(phases_reduced), cmap="gray")
    ax1.imshow(phases_reduced, alpha=magnitudes_norm, cmap=ciecam02_cmap(), origin="lower")
    show_all_circles(
        np.zeros_like(magnitudes_reduced),
        ltem_contours_reduced,
        ax=ax1,
        color="white",
        origin="lower",
        show_numbers=False,
    )
    ax1.set_title(f"Phases for {name}")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.add_artist(ScaleBar(ltem_xerror))
    if show_angles:
        ax12 = inset_axes(ax1, width="15%", height="15%", loc=4, axes_class=get_projection_class("polar"))
        show_phase_colors_circle(ax12, add_dark_background=True, show_angles=False)

    ax2 = plt.subplot(132)
    ax2.imshow(magnitudes_reduced, cmap="gray", origin="lower")
    show_all_circles(
        np.zeros_like(magnitudes_reduced),
        ltem_contours_reduced,
        ax=ax2,
        color="yellow",
        origin="lower",
        show_numbers=False,
    )
    # show_all_circles(magnitudes_reduced,
    #                 ltem_circular_contours, ax2, alpha=0.5, show_numbers=False)
    # ax2.set_title(f'Magnitudes for {ltem_data_name}');
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f"Magnitudes for {name}")

    # ax3 = plt.subplot(222, polar=True)
    # show_phase_colors_circle(add_dark_background=False, ax=ax3)

    ax4 = plt.subplot(133)
    ax4.hist(ltem_box_widths, bins=25)
    ax4.set_xlabel("Pixels")
    ax4.set_yticks([])
    ax4.set_title(f"{name} contour widths")


def show_magnetization_components(magnitudes, m_1, m_2, m_3, axis_1, axis_2, axis_3):
    azim, elev = 15, 10

    xs, ys = np.linspace(1, magnitudes.shape[1], magnitudes.shape[1]), np.linspace(
        1, magnitudes.shape[0], magnitudes.shape[0]
    )
    Xs, Ys = np.meshgrid(xs, ys)

    zs = m_3  # magnitudes * np.cos(thetas_flattened)

    fig = plt.figure(figsize=(30, 11))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.azim = azim
    ax1.elev = elev
    # ax.contour3D(Xs, Ys, zs, 100, cmap=cm.coolwarm, alpha=0.5);
    ax1.plot_surface(Xs, Ys, zs, cmap=cm.coolwarm, linewidth=0, alpha=0.3)
    ax1.plot_surface(Xs, Ys, np.zeros_like(Xs), alpha=0.5)

    ax1.set_xlabel(rf"M$_{axis_1}$")
    ax1.set_ylabel(rf"M$_{axis_2}$")
    ax1.set_zlabel(rf"M$_{axis_3}$")
    ax1.set_title(f"{axis_3} component of surface magnetization")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.azim = azim
    ax2.elev = elev
    # ax.contour3D(Xs, Ys, zs, 100, cmap=cm.coolwarm, alpha=0.5);
    ax2.plot_surface(Xs, Ys, m_1, cmap=cm.coolwarm, linewidth=0, alpha=0.3)
    ax2.plot_surface(Xs, Ys, np.zeros_like(Xs), alpha=0.5)

    ax2.set_xlabel(rf"M$_{axis_2}$")
    ax2.set_ylabel(rf"M$_{axis_3}$")
    ax2.set_zlabel(rf"M$_{axis_1}$")
    ax2.set_title(f"{axis_1} component of surface magnetization")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.azim = azim
    ax3.elev = elev
    # ax.contour3D(Xs, Ys, zs, 100, cmap=cm.coolwarm, alpha=0.5);
    ax3.plot_surface(Xs, Ys, m_2, cmap=cm.coolwarm, linewidth=0, alpha=0.3)
    ax3.plot_surface(Xs, Ys, np.zeros_like(Xs), alpha=0.5)

    ax3.set_xlabel(rf"M$_{axis_1}$")
    ax3.set_ylabel(rf"M$_{axis_3}$")
    ax3.set_zlabel(rf"M$_{axis_2}$")
    ax3.set_title(f"{axis_2} component of surface magnetization")


def show_magnitude_distribution(magnitudes):
    bins = 32

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)
    n, _, _ = ax.hist(magnitudes.flatten(), bins=bins, zorder=10)
    max_count = max(n)
    avg = (1 + 2 * np.squeeze(np.where(n == max_count))[()]) / (2 * bins) * np.max(magnitudes)
    spline = UnivariateSpline(np.linspace(0, np.max(magnitudes), bins), n - max_count / 2, s=0)
    plt.axvspan(*spline.roots(), facecolor="gray", alpha=0.4, zorder=0)

    ax.set_xlim(0, np.max(magnitudes))
    ax.set_title(f"Magnitudes normalized (max count at {avg:.2f})")


def show_phase_colors_circle_old(ax=None, add_dark_background=True, text_color="white"):
    """Plot a ring of colors for a legend."""
    xs = np.arange(0, 2 * np.pi, 0.01)
    ys = np.ones_like(xs)

    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1, polar=True)

    fig = plt.gcf()
    dim = (fig.get_size_inches() * fig.dpi)[0]

    if add_dark_background:
        ax.scatter(0, 0, s=dim**2, marker="o", color="#3A404C")

    ax.scatter(
        xs,
        ys,
        c=xs,
        s=(dim / 8) ** 2,
        lw=0,
        cmap=ciecam02_cmap(),
        vmin=0,
        vmax=2 * np.pi,
    )

    ax.set_yticks(())
    ax.tick_params(axis="x", colors=text_color)
    ax.set_anchor("W")


def show_phase_colors_circle(ax=None, add_dark_background=True, text_color="white", show_angles=True):
    """Plot a ring of colors for a legend."""
    # Generate a figure with a polar projection
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    ax.patch.set_alpha(0)

    fig = plt.gcf()

    if add_dark_background:
        dark = "#3A404C"
        ax.patch.set_facecolor(dark)
        text_color = dark
    else:
        ax.patch.set_facecolor("white")
        text_color = "black"

    # Plot a color mesh on the polar plot with the color set by the angle

    n = 180  # the number of secants for the mesh
    t = np.linspace(0, 2 * np.pi, n)  # theta values
    # radius values; change 0.6 to 0 for full circle
    r = np.linspace(0.6, 1, 2)
    rg, tg = np.meshgrid(r, t)  # create r,theta meshgrid
    # c = tg                          # define color values as theta value
    # plot the colormesh on axis with colormap
    im = ax.pcolormesh(t, r, tg.T, cmap=ciecam02_cmap(), shading="auto")
    ax.set_yticklabels([])  # turn off radial tick labels (yticks)
    # cosmetic changes to tick labels
    if show_angles:
        ax.tick_params(axis="x", pad=15, labelsize=14, colors=text_color)
    else:
        ax.set_xticklabels([])
    # ax.spines['polar'].set_visible(False)    # turn off the axis spines.


def show_subplot(image, rows=1, cols=1, pos=1, title="", vmin=0, vmax=1, ax=None, hide_axes=True):
    """Add a subplot with values normalized to 0..1."""
    if ax is None:
        ax = plt.subplot(rows, cols, pos)

    ax.imshow(image, vmin=vmin, vmax=vmax, cmap="gray")
    ax.grid(False)
    ax.set_title(title)

    if hide_axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    return ax


def show_subplot_raw(image, rows=1, cols=1, pos=1, title="", ax=None, hide_axes=True):
    """Add a subplot without normalized values."""
    return show_subplot(
        image,
        rows,
        cols,
        pos,
        title,
        vmin=np.min(image),
        vmax=np.max(image),
        ax=ax,
        hide_axes=hide_axes,
    )


def show_vector_plot(im_x, im_y, ax=None, color="white", scale=2, divisor=32):
    """Create a vector plot of the phases."""
    # Get dimensions
    yd, xd = im_x.shape

    X = np.linspace(xd / divisor, xd * (divisor - 1) / divisor, divisor, dtype=np.uint8)
    Y = np.linspace(yd / divisor, yd * (divisor - 1) / divisor, divisor, dtype=np.uint8)

    # Create a pair of (x, y) coordinates
    U, V = np.meshgrid(X, Y)
    x, y = U.ravel(), V.ravel()

    # Pull the values at those coordinates.
    Xs = convolve2d(im_x, np.ones((15, 15)), mode="same")[y, x]
    Ys = convolve2d(im_y, np.ones((15, 15)), mode="same")[y, x]

    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    # Show the plot
    ax.quiver(x, y, Xs, Ys, angles="uv", scale_units="dots", color=color, scale=scale)


def show_winding_numbers(candidates, all):
    """Plot histogram of winding numbers

    Args:
        candidates (list): Winding numbers for candidate features
        all (list): Winding numbers for all features
    """
    bins = 8

    plt.figure(figsize=(12, 6))
    plt.hist(candidates, bins=bins, density=True, zorder=10, alpha=0.8, label="Candidates")
    plt.hist(all, bins=bins, density=True, zorder=1, histtype="step", label="All")
    plt.xlim(1.1 * min(candidates + all), 1.1 * max(candidates + all))
    plt.legend()
    plt.xlabel("Winding number")
    plt.yticks([])


def display_results(
    img_contrast_phase,
    img_denoised_1,
    img_flat_1,
    img_denoised_2,
    img_flat_2,
    img_intensity,
    img_scale,
    full_name,
    arrow_scale=4,
    arrow_color="black",
    axis_1="x",
    axis_2="y",
):
    """Plot the results."""
    fig = plt.figure(figsize=(20, 15), constrained_layout=True)

    gs = fig.add_gridspec(3, 3)
    gs.update(wspace=0.3, hspace=0.3)

    # Contrast image
    ax1 = plt.subplot(gs[:-1, :-1])
    ax1.imshow(img_contrast_phase)
    show_vector_plot(img_denoised_1, img_denoised_2, ax=ax1, color=arrow_color, scale=arrow_scale)
    ax1.add_artist(ScaleBar(img_scale, box_alpha=0.8))
    ax1.set_title(
        "Domains in the {}-{} plane for {}".format(axis_1, axis_2, full_name),
        fontdict={"fontsize": 24},
    )

    # Vector legend
    ax2 = plt.subplot(gs[:-1, -1], polar=True)
    show_phase_colors_circle(ax2)
    ax2.set_title("Magnetization angle", fontdict={"fontsize": 20})

    # Flattened intensity
    ax3 = plt.subplot(gs[-1, 0])
    ax3.imshow(img_intensity, cmap="gray")
    ax3.add_artist(ScaleBar(img_scale))
    ax3.set_title("Intensity flattened")

    # Flattened M1
    ax4 = plt.subplot(gs[-1, 1])
    ax4.imshow(img_flat_1, cmap="gray")
    ax4.add_artist(ScaleBar(img_scale))
    ax4.set_title("M{} flattened: {:.3f} to {:.3f}".format(axis_1, img_flat_1.min(), img_flat_1.max()))

    # Add a colorbar
    add_colorbar(img_flat_1, ax4, r"$M_{rel}$")

    # Flattened M2
    ax5 = plt.subplot(gs[-1, -1])
    ax5.imshow(img_flat_2, cmap="gray")
    ax5.add_artist(ScaleBar(img_scale))
    ax5.set_title("M{} flattened: {:.3f} to {:.3f}".format(axis_2, img_flat_2.min(), img_flat_2.max()))

    # Add a colorbar
    add_colorbar(img_flat_1, ax5, r"$M_{rel}$")

    # Turn off grids and axes except for the legend plot
    for ax in fig.get_axes():
        if len(ax.images) > 0:
            ax.grid(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
