"""
Utility functions: geometry, padding, intersection, tile/chunk logic.
"""

import numpy as np
from typing import Tuple
import cv2
from skimage.morphology import disk
from src.tiff_stitching.config import TILE_SIZE, OVERLAP, STRIDE, CHUNK_SIZE

BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1)
Pad = Tuple[int, int, int, int]  # (left, top, right, bottom)
Shape = Tuple[int, int, int]  # (C, H, W)
Edge = Tuple[bool, bool, bool, bool]  # (left, top, right, bottom)


def labels_to_polygons(labels: np.ndarray, convex_hull=False, smooth=None):
    """
    Convert labels to polygons, optimizing the process by using OpenCV's findContours.
    This function assumes that the labels are in the format of a 2D array where each pixel's value corresponds to its label.
    The function will return a list of polygons, where each polygon is represented as a list of points.
    The points are in the format (x, y), where x is the column index and y is the row index.
    The function will ignore the background label (0) and only return polygons for non-zero labels.
    Process is optimized for large images by cropping the ROI
    """
    polygons = []
    centroids_x = []
    centroids_y = []
    edges = []

    unique_labels = np.unique(labels)
    image_height, image_width = labels.shape

    for label in unique_labels:
        if label == 0:
            continue  # Ignore background label

        # Get bbox for this label
        where = np.where(labels == label)
        y_min, y_max = np.min(where[0]), np.max(where[0])
        x_min, x_max = np.min(where[1]), np.max(where[1])

        # Check if label touches image edges
        edge = (
            x_min == 0,  # left
            y_min == 0,  # top
            x_max == image_width - 1,  # right
            y_max == image_height - 1,  # bottom
        )
        edges.append(edge)

        # Crop to ROI
        mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        mask[labels[y_min : y_max + 1, x_min : x_max + 1] == label] = 255

        # Apply smoothing if specified
        if smooth:
            kernel = disk(smooth)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if convex_hull:
                contour = cv2.convexHull(contour)

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            centroid_x = int(round(moments["m10"] / moments["m00"]))
            centroid_y = int(round(moments["m01"] / moments["m00"]))
            centroids_x.append(centroid_x)
            centroids_y.append(centroid_y)

            polygon = contour.reshape(-1, 2)
            # Ensure the polygon is closed
            if np.array_equal(polygon[0], polygon[-1]) is False:
                polygon = np.vstack((polygon, polygon[0]))
            # Offset polygon coordinates to match original image
            polygon[:, 0] += x_min
            polygon[:, 1] += y_min
            polygons.append(polygon)

    return polygons, edges


def intersects(bbox: BBox, chunk: BBox) -> bool:
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = chunk
    return not (x1 <= cx0 or cx1 <= x0 or y1 <= cy0 or cy1 <= y0)


def pad_array(arr: np.ndarray, pad: Pad, constant: int = 0) -> np.ndarray:
    left, top, right, bottom = pad
    # arr shape could be (C,H,W) or (H,W)
    if arr.ndim == 3:
        c, h, w = arr.shape
        pad_width = ((0, 0), (top, bottom), (left, right))
    else:
        h, w = arr.shape
        pad_width = ((top, bottom), (left, right))
    return np.pad(arr, pad_width, mode="constant", constant_values=constant)


def needed_by_future_chunks(
    tile_bbox: BBox, current_chunk: BBox, image_size: Tuple[int, int]
) -> bool:
    x0, y0, x1, y1 = tile_bbox
    cx0, cy0, cx1, cy1 = current_chunk
    H, W = image_size
    # if tile extends below or right of current chunk, it may still be needed
    return (y1 > cy1) or (x1 > cx1)
