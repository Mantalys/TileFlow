import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import sys
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from typing import Tuple


# create pydantic Chunk class Chunk:
class Chunk(BaseModel):
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    position: int  #  gives the position of the chunk in the image, 0 for first chunk, 1 for second chunk, etc.

    @property
    def width(self) -> int:
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    def chunk_image(self, image) -> np.array:
        return image[self.y_start : self.y_end, self.x_start : self.x_end]

    def get_valid_xmax(self, overlap: int) -> int:
        """
        Returns the valid x coordinate for the chunk, considering the overlap.
        """
        return self.x_start + self.width - overlap

    def get_valid_xmin(self, overlap: int) -> int:
        """
        Returns the valid x coordinate for the chunk, considering the overlap.
        """
        return self.x_start + overlap


class ChunkData(BaseModel):
    polygons: list
    centroids: list
    valid_labels: set


def stitching_list(chunk_list_output, chunk_grid, overlap, tile_size):
    """
    Stitch multiple chunks together based on their coordinates and overlap.
    """
    line, row = chunk_grid
    label_max = 0
    height_reconstructed, width_reconstructed = 0, 0
    height_reconstructed = chunk_list_output[0][0].height
    width_reconstructed = (chunk_list_output[-1][0].x_end)
    total_cells=0
    x_lines=[]
    #x_lines.append(width_reconstructed//2)
    print(f"reconstru width {width_reconstructed}")

    reconstructed = np.zeros(
        (
            height_reconstructed,
            width_reconstructed,
        ),
        dtype=np.uint16,
    )
    for chunk in range(0, len(chunk_list_output)):

            print(f"Process Chunk {chunk}")
            unique_labels1 = np.unique(chunk_list_output[chunk][1])
            chunk_relabel = np.where(
                chunk_list_output[chunk][1] != 0, chunk_list_output[chunk][1] + label_max, 0
            )
            unique_labels2 = np.unique(chunk_relabel)
            chunk_1_data = ChunkData(polygons=[], centroids=[], valid_labels=set())
            offset=0
            for label in unique_labels2:
                if label == 0:
                    continue
                polygon, centroid = process_mask(
                    chunk_relabel,
                    label,
                    smooth=0,
                    convex_hull=False,
                    offset=np.array([0, 0]),
                    x_offset=chunk_list_output[chunk][0].x_start,
                    y_offset=0,
                    return_centroid=True,
                )
                if centroid is None:
                    continue

                x, y = centroid
                # check if chunk is on the left (no neighbor chunk)
                # could be precomputed
                if chunk_list_output[chunk][0].position == 0:
                    offset=0
                    if x <= chunk_list_output[chunk][0].get_valid_xmax(overlap * tile_size):
                        chunk_1_data.polygons.append(polygon)
                        chunk_1_data.centroids.append(centroid)
                        chunk_1_data.valid_labels.add(label)
                        x_offset=0

                elif chunk_list_output[chunk][0].position == row-1:
                    offset = chunk_list_output[chunk][0].x_start
                    if x>=chunk_list_output[chunk][0].get_valid_xmin(overlap*tile_size):
                        chunk_1_data.polygons.append(polygon)
                        chunk_1_data.centroids.append(centroid)
                        chunk_1_data.valid_labels.add(label)

                
                else :
                    offset = chunk_list_output[chunk][0].x_start
                    if (chunk_list_output[chunk][0].get_valid_xmin(overlap*tile_size)<=x) and x<=chunk_list_output[chunk][0].get_valid_xmax(overlap*tile_size):
                        chunk_1_data.polygons.append(polygon)
                        chunk_1_data.centroids.append(centroid)
                        chunk_1_data.valid_labels.add(label)
            label_max = np.max(chunk_relabel)

            print(f"Chunk {chunk} unique labels: {len((chunk_1_data.valid_labels)) - 1}")
            total_cells+=len((chunk_1_data.valid_labels)) - 1
            print(f"Nb labels avant reconstruction : {total_cells}")
            # Recalibrage des labels chunk_2
            draw_polygons_in_mask(
                    reconstructed, chunk_1_data.polygons, list(chunk_1_data.valid_labels),x_offset=offset)
            if chunk != len(chunk_list_output)-1:
                x_lines.append((chunk,chunk_list_output[chunk][0].x_start,chunk_list_output[chunk][0].x_end))
            else:
                x_lines.append((chunk,chunk_list_output[chunk][0].x_start,chunk_list_output[chunk][0].get_valid_xmax(overlap*tile_size)+overlap*tile_size-1))#-1 is just for plot if we don't do this the figure would be enlarge and there would be an empty area

    reconstructed = randomize_labels(reconstructed)
    plt.figure()
    plt.imshow(reconstructed, cmap="viridis")
    for i,start,end in x_lines:
        if i==len(x_lines)-1:
            plt.axvline(x=start, color="g")#début chunk en vert
            plt.text(start + 5, 10, f"Début Chunk {i}", color="g", rotation=90, va='bottom', fontsize=8)
            plt.axvline(x=end, color="r",label=f"Chunk_{i}_end")#fin chunk en rouge
            plt.text(end + 5, 10, f"Fin Chunk {i+1}", color="r", rotation=90, va='bottom', fontsize=8)
        else:
            plt.axvline(x=start, color="g",label=f"Chunk_{i}_start")#début chunk en vert
            plt.axvline(x=end, color="r",label=f"Chunk_{i}_end")#fin chunk en rouge
    plt.axis("on")
    plt.title("Image reconstruite")
    plt.show()

    return reconstructed


def stitching(
    chunk_1: np.array,
    chunk_2: np.array,
    coord_chunk_1: Chunk,
    coord_chunk_2: Chunk,
    overlap: int,
    chunk_size: int,
    tile_size: int,
) -> np.array:
    # y10, x10, y11, x11 = coord_chunk_1
    # y20, x20, y21, x21 = coord_chunk_2
    # h1, w1 = chunk_1.shape
    # h2, w2 = chunk_2.shape
    overlap_cols = overlap * tile_size

    chunk_1_valid_x = coord_chunk_1.get_valid_xmax(overlap_cols)
    print(f"Valid x for chunk 1: {chunk_1_valid_x}")

    chunk_2_valid_x = coord_chunk_2.get_valid_xmin(overlap_cols)
    print(f"Valid x for chunk 2: {chunk_2_valid_x}")

    # Étendre les chunks temporairement pour récupérer les polygones complets
    # chunk_1_ext = np.concatenate((chunk_1, chunk_2[:, :overlap_cols]), axis=1)
    # chunk_2_ext = np.concatenate((chunk_1[:, -overlap_cols:], chunk_2), axis=1)

    # Extraction depuis chunk_1 étendu
    unique_labels1 = np.unique(chunk_1)
    # unique_labels1 = unique_labels1[unique_labels1 != 0]

    chunk_1_data = ChunkData(polygons=[], centroids=[], valid_labels=set())
    for label in unique_labels1:
        if label == 0:
            continue
        polygon, centroid = process_mask(
            chunk_1,
            label,
            smooth=0,
            convex_hull=False,
            offset=np.array([0, 0]),
            x_offset=0,
            y_offset=0,
            return_centroid=True,
        )
        if centroid is None:
            continue

        x, y = centroid
        if x < chunk_1_valid_x:  # centroïde dans la zone non-overlap de chunk_1
            chunk_1_data.polygons.append(polygon)
            chunk_1_data.centroids.append(centroid)
            chunk_1_data.valid_labels.add(label)

    # Recalibrage des labels chunk_2
    max_label_1 = np.max(chunk_1)
    # chunk_2_ext_relabel = np.where(chunk_2 != 0, chunk_2 + max_label_1 + 1, 0)
    chunk_2_relabel = np.where(chunk_2 != 0, chunk_2 + max_label_1, 0)

    unique_labels2 = np.unique(chunk_2_relabel)
    # unique_labels2 = unique_labels2[unique_labels2 != 0]

    chunk_2_data = ChunkData(polygons=[], centroids=[], valid_labels=set())
    for label in unique_labels2:
        if label == 0:
            continue
        polygon, centroid = process_mask(
            chunk_2_relabel,
            label,
            smooth=0,
            convex_hull=False,
            offset=np.array([0, 0]),
            x_offset=0,
            y_offset=0,
            return_centroid=True,
        )
        if centroid is None:
            continue
        x, y = centroid
        # x = (
        # x + coord_chunk_1.width - overlap_cols
        # )  # Ajustement de l'offset pour chunk_2
        if x >= overlap * tile_size:  # centroïde dans la zone non-overlap de chunk_2
            chunk_2_data.polygons.append(polygon)
            chunk_2_data.centroids.append(centroid)
            chunk_2_data.valid_labels.add(label)

    # Reconstruction finale (sans overlap doublé)
    # h_full = x21
    # w_full = w1 + w2 - overlap_cols
    print(
        f"Reconstruction dimensions: height={coord_chunk_1.height}, width={coord_chunk_1.width + coord_chunk_2.width - overlap_cols * 2}"
    )
    reconstructed = np.zeros(
        (
            coord_chunk_1.height,
            coord_chunk_1.width + coord_chunk_2.width - overlap_cols * 2,
        ),
        dtype=np.uint16,
    )
    draw_polygons_in_mask(
        reconstructed, chunk_1_data.polygons, list(chunk_1_data.valid_labels)
    )
    draw_polygons_in_mask(
        reconstructed,
        chunk_2_data.polygons,
        list(chunk_2_data.valid_labels),
        x_offset=coord_chunk_1.width - overlap_cols * 2,
    )

    reconstructed = randomize_labels(reconstructed)
    x_line_mid = chunk_1_valid_x

    x_line_c1_ext = x_line_mid + overlap_cols
    x_line_c2_ext = x_line_mid - overlap_cols
    print(x_line_mid, x_line_c1_ext, x_line_c2_ext)

    # Tracer une ligne verte verticale sur toute la hauteur de l'image
    # (0, 255, 0) = vert en BGR

    # Affichage
    plt.figure()
    plt.imshow(reconstructed, cmap="viridis")
    plt.axvline(x=x_line_c1_ext, color="g", label="c1_extended")
    plt.axvline(x=x_line_c2_ext, color="g", label="c2_extended")
    plt.axvline(x=x_line_mid, color="r", label="milieu_image")

    plt.title("Image reconstruite")
    plt.show()

    return reconstructed


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

    if smooth:
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


def draw_polygons_in_mask(image, polygons, labels, x_offset=0, y_offset=0):
    """
    Dessine des polygones dans `image` avec les `labels` correspondants.
    """
    for poly, label in zip(polygons, labels):
        pts = np.array(poly, dtype=np.int32)
        pts[:, 0] += x_offset
        pts[:, 1] += y_offset
        cv2.fillPoly(image, [pts], int(label))
