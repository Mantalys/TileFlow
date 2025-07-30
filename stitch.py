import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import sys
from skimage.morphology import disk


def stitching(
    chunk_1: np.array,
    chunk_2: np.array,
    coord_chunk_1: tuple,
    coord_chunk_2: tuple,
    overlap: int,
    chunk_size: int,
    tile_size: int,
) -> np.array:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(chunk_1, cmap="viridis")
    plt.subplot(1, 2, 2)
    plt.imshow(chunk_2, cmap="viridis")
    plt.show()
    x10, y10, x11, y11 = coord_chunk_1
    x20, y20, x21, y21 = coord_chunk_2
    h1, w1 = chunk_1.shape
    h2, w2 = chunk_2.shape

    unique_labels1 = np.unique(chunk_1)
    unique_labels1 = unique_labels1[
        unique_labels1 != 0
    ]  # optionnel : ignorer l'arrière-plan

    polys1 = []
    centros1 = []
    valid_labels_1 = set()
    for label in unique_labels1:
        poly1, centro1 = process_mask(
            src=chunk_1,
            label=label,
            smooth=1,
            convex_hull=False,
            offset=np.array([0, 0]),
            x_offset=0,
            y_offset=0,
            return_centroid=True,
        )
        x, y = centro1
        if x < w1 - overlap * tile_size:
            polys1.append(poly1)
            centros1.append(centro1)
            valid_labels_1.add(label)
    print(
        f"nombre de poly dans chunk_1 {len(polys1)}) et nb de centroides : {len(centros1)}"
    )

    max_label_1 = np.max(chunk_1)
    chunk_2_relabel = np.where(chunk_2 != 0, chunk_2 + max_label_1 + 1, 0)
    unique_labels2 = np.unique(chunk_2_relabel)
    unique_labels2 = unique_labels2[
        unique_labels2 != 0
    ]  # optionnel : ignorer l'arrière-plan

    polys2 = []
    centros2 = []
    valid_labels_2 = set()
    for label in unique_labels2:
        poly2, centro2 = process_mask(
            src=chunk_2_relabel,
            label=label,
            smooth=1,
            convex_hull=False,
            offset=np.array([0, 0]),
            x_offset=0,
            y_offset=0,
            return_centroid=True,
        )
        if centro2 is None:
            continue
        x, y = centro2
        if x + (w1 - overlap * tile_size) > w1:
            polys2.append(poly2)
            centros2.append(centro2)
            valid_labels_2.add(label)
    print(
        f"nombre de poly dans chunk_2 {len(polys2)}) et nb de centroides : {len(centros2)}"
    )

    max_label_1 = np.max(chunk_1)
    chunk_2_relabel = np.where(chunk_2 != 0, chunk_2 + max_label_1 + 1, 0)
    image_full = np.zeros((x21, y21 + y20), dtype=np.uint16)
    image_full[:, : w1 - overlap * tile_size] = chunk_1[:, : w1 - overlap * tile_size]
    image_full[:, w1 - overlap * tile_size :] = chunk_2_relabel[
        :, overlap * tile_size :
    ]

    valid_labels = valid_labels_1.union(valid_labels_2)
    for label in np.unique(image_full):
        if label != 0 and label not in valid_labels:
            image_full[image_full == label] = 0

    # Étape 4 : randomiser seulement les labels restants
    image_full = randomize_labels(image_full)

    plt.figure()
    plt.imshow(image_full, cmap="viridis")
    plt.show()

    return image_full


def process_mask(
    src,
    label,
    smooth,
    convex_hull,
    offset,
    x_offset=0,
    y_offset=0,
    return_centroid=False,
):
    mask = np.zeros(src.shape, dtype=np.uint8)
    mask[src == label] = 255
    kernel = disk(smooth)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    if convex_hull:
        contour = cv2.convexHull(contour)
    polygon = contour.reshape(-1, 2)
    if not np.array_equal(polygon[0], polygon[-1]):
        polygon = np.vstack((polygon, polygon[0]))
    polygon_out = (polygon + offset).tolist()
    centroid = None
    if return_centroid:
        moments = cv2.moments(contour)
        m00 = moments["m00"] if moments["m00"] != 0 else sys.float_info.epsilon
        centroid = (
            int(round(moments["m10"] / m00)) + x_offset,
            int(round(moments["m01"] / m00)) + y_offset,
        )
    return polygon_out, centroid


def randomize_labels(segmentation):
    """
    Prend une image de segmentation (2D ou 3D) avec des labels, et assigne
    aléatoirement de nouveaux IDs aux labels existants.

    Args:
        segmentation (np.ndarray): Image de segmentation avec des entiers comme labels.

    Returns:
        np.ndarray: Nouvelle image de segmentation avec labels réassignés aléatoirement.
    """
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels != 0]  # Exclure l’arrière-plan (0)

    shuffled_labels = unique_labels.copy()
    np.random.shuffle(shuffled_labels)

    label_mapping = {old: new for old, new in zip(unique_labels, shuffled_labels)}

    new_segmentation = np.copy(segmentation)
    for old_label, new_label in label_mapping.items():
        new_segmentation[segmentation == old_label] = new_label

    return new_segmentation
