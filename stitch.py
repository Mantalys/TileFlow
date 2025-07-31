import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import sys
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel


# create pydantic Chunk class Chunk:
class Chunk(BaseModel):
    x_start: int
    y_start: int
    x_end: int
    y_end: int

    @property
    def width(self) -> int:
        return self.x_end - self.x_start
    
    @property
    def height(self) -> int:
        return self.y_end - self.y_start
    
    def chunk_image(self, image) -> np.array:
        return image[self.x_start:self.x_end, self.y_start:self.y_end]
    
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
        polygon, centroid = process_mask(chunk_1, label, smooth=1, convex_hull=False,
                                      offset=np.array([0, 0]), x_offset=0, y_offset=0,
                                      return_centroid=True)
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
    chunk_2_relabel = np.where(chunk_2 != 0, chunk_2 + max_label_1 + 1, 0)

    unique_labels2 = np.unique(chunk_2_relabel)
    # unique_labels2 = unique_labels2[unique_labels2 != 0]

    chunk_2_data = ChunkData(polygons=[], centroids=[], valid_labels=set())
    for label in unique_labels2:
        if label == 0:
            continue
        polygon, centroid = process_mask(chunk_2_relabel, label, smooth=1, convex_hull=False,
                                      offset=np.array([0, 0]), x_offset=0, y_offset=0,
                                      return_centroid=True)
        if centroid is None:
            continue
        x, y = centroid
        x = x + coord_chunk_1.width - overlap_cols  # Ajustement de l'offset pour chunk_2
        if x >= chunk_2_valid_x:  # centroïde dans la zone non-overlap de chunk_2
            chunk_2_data.polygons.append(polygon)
            chunk_2_data.centroids.append(centroid)
            chunk_2_data.valid_labels.add(label)

    # Reconstruction finale (sans overlap doublé)
    # h_full = x21
    # w_full = w1 + w2 - overlap_cols
    print(f"Reconstruction dimensions: height={coord_chunk_1.height}, width={coord_chunk_1.width + coord_chunk_2.width - overlap_cols * 2}")
    reconstructed = np.zeros((1400, 1868), dtype=np.uint16)
    draw_polygons_in_mask(reconstructed, chunk_1_data.polygons, list(chunk_1_data.valid_labels))
    draw_polygons_in_mask(reconstructed, chunk_2_data.polygons, list(chunk_2_data.valid_labels), x_offset=coord_chunk_1.width - overlap_cols)

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
    plt.axvline(x=x_line_c1_ext, color='g', label='c1_extended')
    plt.axvline(x=x_line_c2_ext, color='g', label='c2_extended')
    plt.axvline(x=x_line_mid, color='r', label='milieu_image')

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


def draw_polygons_in_mask(image, polygons, labels,x_offset=0,y_offset=0):
    """
    Dessine des polygones dans `image` avec les `labels` correspondants.
    """
    for poly, label in zip(polygons, labels):
        pts = np.array(poly, dtype=np.int32)
        pts[:, 0] += x_offset
        pts[:, 1] += y_offset
        cv2.fillPoly(image, [pts], int(label))
