import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from src.tiff_stitching.utils import (
    BBox, Edge
)


class ChunkShape:
    def __init__(self, context: BBox, core: BBox):
        self.context = context
        self.core = core

    @property
    def width(self) -> int:
        return self.context[2] - self.context[0]

    @property
    def height(self) -> int:
        return self.context[3] - self.context[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)
    
    @property
    def core_shape(self) -> Tuple[int, int]:
        return (self.core[3] - self.core[1], self.core[2] - self.core[0])
    
    def is_inside(self, x: float, y: float) -> bool:
        """
        Check if the point (x, y) is within the core bounding box of the chunk.
        """
        return (
            self.core[0] <= x < self.core[2] and
            self.core[1] <= y < self.core[3]
        )


class Chunk2D:
    def __init__(self, shape: ChunkShape, edges: Edge, position: Tuple[int, int]):
        self.shape = shape
        self.edges = edges
        self.position = position
        self.array: np.ndarray = None

    def set_array(self, array: np.ndarray):
        """
        Sets the array for the chunk, the shape must match the chunk shape.
        """
        if array.shape != self.shape.shape:
            raise ValueError(
                f"Array shape {array.shape} does not match chunk shape {self.shape.shape}"
            )
        self.array = array


    def get_slice(self) -> Tuple[slice, slice]:
        """
        Returns the slice for the chunk based on its context.
        This is useful for indexing into a larger array.
        """
        return (
            slice(self.shape.context[1], self.shape.context[3]),
            slice(self.shape.context[0], self.shape.context[2]),
        )


class ChunkData:
    def __init__(self):
        self.polygons = []
        self.centroids = []
        self.valid_labels = set()


def extract_polygons(
        chunks: List[Chunk2D],
    ):
    """
    Extracts polygons and centroids from a list of chunks.
    """
    reconstructed = np.zeros(
        (
            1400,
            1868,
        ),
        dtype=np.uint16,
    )
    return reconstructed



def stitching_list(
        chunk_list_output,
        chunk_grid,
        overlap
    ):
    """
    Stitch multiple chunks together based on their coordinates and overlap.
    """
    row_max, col_max = chunk_grid
    label_max = 0
    height_reconstructed, width_reconstructed = 0, 0
    height_reconstructed = chunk_list_output[0][0].height
    width_reconstructed = chunk_list_output[-1][0].x_end
    total_cells = 0
    total_centros = 0
    total_centros1 = 0
    x_lines = []
    liste_total = []
    # x_lines.append(width_reconstructed//2)
    print(f"reconstru width {width_reconstructed}")

    reconstructed = np.zeros(
        (
            height_reconstructed,
            width_reconstructed,
        ),
        dtype=np.uint16,
    )
    for chunk in range(0, len(chunk_list_output)):
        row, col = chunk_list_output[chunk][0].position
        print(f"Process Chunk {chunk}, row: {row}, col: {col}")
        unique_labels1 = np.unique(chunk_list_output[chunk][1])
        chunk_relabel = np.where(
            chunk_list_output[chunk][1] != 0, chunk_list_output[chunk][1] + label_max, 0
        )
        unique_labels2 = np.unique(chunk_relabel)
        chunk_1_data = ChunkData(polygons=[], centroids=[], valid_labels=set())
        offset = 0
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
            if col == 0 and not col ==col_max - 1:
                offset = 0
                if x < chunk_list_output[chunk][0].get_valid_xmax(overlap):
                    chunk_1_data.polygons.append(polygon)
                    chunk_1_data.centroids.append(centroid)
                    chunk_1_data.valid_labels.add(label)
                    x_offset = 0

            
            elif col == col_max - 1 and not col == 0:
                offset = chunk_list_output[chunk][0].x_start
                if x >= chunk_list_output[chunk][0].get_valid_xmin(overlap):
                    chunk_1_data.polygons.append(polygon)
                    chunk_1_data.centroids.append(centroid)
                    chunk_1_data.valid_labels.add(label)

            elif col == 0 and col == col_max - 1:
                
                if (
                    chunk_list_output[chunk][0].get_valid_xmin(0) <= x
                    and x < chunk_list_output[chunk][0].get_valid_xmax(0)
                ):
                    chunk_1_data.polygons.append(polygon)
                    chunk_1_data.centroids.append(centroid)
                    chunk_1_data.valid_labels.add(label)
                print("Chunk is the only one ")
                
            else:
                offset = chunk_list_output[chunk][0].x_start
                if (
                    chunk_list_output[chunk][0].get_valid_xmin(overlap) <= x
                ) and x < chunk_list_output[chunk][0].get_valid_xmax(overlap):
                    chunk_1_data.polygons.append(polygon)
                    chunk_1_data.centroids.append(centroid)
                    chunk_1_data.valid_labels.add(label)
        label_max = np.max(chunk_relabel)
        print(f"Chunk {chunk} unique labels: {len((chunk_1_data.valid_labels))}")
        total_cells += len((chunk_1_data.valid_labels))
        print(f"Nb labels avant reconstruction : {total_cells}")
        total_centros += len((chunk_1_data.centroids))
        print(f"Nb centroides avant reconstruction : {total_centros}")
        total_centros1 += len(set(chunk_1_data.centroids))
        print(f"Nb centroides avant reconstruction : {total_centros1}")
        liste_total += chunk_1_data.centroids
        print(
            f"longueur du set centro si != de totalcentro on compte en double {len(set(liste_total))}"
        )

        draw_polygons_in_mask(
            reconstructed,
            chunk_1_data.polygons,
            list(chunk_1_data.valid_labels),
            x_offset=offset,
        )
        if chunk != len(chunk_list_output) - 1:
            x_lines.append(
                (
                    col,
                    chunk_list_output[chunk][0].x_start,
                    chunk_list_output[chunk][0].x_end,
                )
            )
        else:
            x_lines.append(
                (
                    col,
                    chunk_list_output[chunk][0].x_start,
                    chunk_list_output[chunk][0].get_valid_xmax(overlap) + overlap - 1,
                )
            )  # -1 is just for plot if we don't do this the figure would be enlarge and there would be an empty area

    reconstructed = randomize_labels(reconstructed)
    plt.figure()
    plt.imshow(reconstructed, cmap="viridis")
    for i, start, end in x_lines:
        if i == len(x_lines) - 1:
            plt.axvline(x=start, color="g")  # début chunk en vert
            plt.text(
                start - 5,
                0,
                f"Début Chunk {i}",
                color="g",
                rotation=30,
                va="bottom",
                fontsize=10,
            )
            plt.axvline(
                x=end - 5, color="r", label=f"Chunk_{i}_end"
            )  # fin chunk en rouge
            plt.text(
                end + 5,
                0,
                f"Fin Chunk {i}",
                color="r",
                rotation=30,
                va="bottom",
                fontsize=10,
            )
        else:
            plt.axvline(
                x=start, color="g", label=f"Chunk_{i}_start"
            )  # début chunk en vert
            plt.text(
                start - 5,
                0,
                f"Début Chunk {i}",
                color="g",
                rotation=30,
                va="bottom",
                fontsize=10,
            )
            plt.text(
                end - 5,
                0,
                f"Fin Chunk {i}",
                color="r",
                rotation=30,
                va="bottom",
                fontsize=10,
            )
            plt.axvline(
                x=end + 2, color="r", label=f"Chunk_{i}_end"
            )  # fin chunk en rouge
    plt.axis("on")
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
