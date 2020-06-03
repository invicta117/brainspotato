import math
import os

import numpy as np
import pydicom
import scipy.ndimage as ndimage
from scipy import ndimage as ndimage
from scipy.misc import imresize
from skimage import measure, morphology, segmentation
from skimage.filters import roberts, sobel
from skimage.measure import label, perimeter, regionprops
from skimage.morphology import (ball, binary_closing, binary_erosion, closing,
                                dilation, disk, erosion, reconstruction,
                                remove_small_objects)
from skimage.segmentation import clear_border


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(
            slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    resize_slices = []
    for each_slice in scans:
        try:
            slice = np.array(each_slice.pixel_array)
        except ValueError:
            print("Found Corrupt file: ")
        else:
            resize_slices.append(slice)
    image = np.stack(resize_slices)
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def seperate_skull(image):

    marker_internal, marker_external, marker_watershed = generate_markers(
        image)

    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    lungfilter = np.bitwise_or(marker_internal, outline)

    lungfilter = ndimage.morphology.binary_closing(
        lungfilter, structure=np.ones((5, 5)), iterations=3)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((512, 512)))

    return segmented


def generate_markers(image):
    # Creation of the internal Marker
    marker_internal = image < 60
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 1:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-1]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed
