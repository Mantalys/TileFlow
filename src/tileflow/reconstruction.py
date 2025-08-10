from tileflow.core import (
    RegionImage,
)
from typing import List, Tuple
from tileflow.core import Image2D, new_image2d


def reconstruct(regions: List[RegionImage]) -> List[Image2D]:
    """
    Reconstructs a full image from a list of chunks.

    Args:
        regions (List[RegionImage]): List of regions to reconstruct the image from.

    Returns:
        np.ndarray: The reconstructed image.
    """
    last_region = regions[-1]
    if not last_region.region_spec.position.edges.right:
        raise ValueError(
            "Last region must have a right edge to determine full image size."
        )
    if not last_region.region_spec.position.edges.bottom:
        raise ValueError(
            "Last region must have a bottom edge to determine full image size."
        )

    width_reconstructed = last_region.region_spec.geometry.core.x1
    height_reconstructed = last_region.region_spec.geometry.core.y1
    reconstructed = [
        new_image2d(
            (height_reconstructed, width_reconstructed),
            dtype=rdata.dtype,
        )
        for rdata in last_region.image_data
    ]

    for region in regions:
        if region.image_data is None:
            continue
        core_bbox = region.region_spec.geometry.core
        core_image = region.only_core_image()
        for i in range(len(core_image)):
            reconstructed[i][
                core_bbox.y0 : core_bbox.y1, core_bbox.x0 : core_bbox.x1
            ] = core_image[i]

    return reconstructed
    label_max = 0
    reconstructed = np.zeros(
        (
            height_reconstructed,
            width_reconstructed,
        ),
        dtype=np.uint16,
    )
    print(f"Reconstructed image size: {reconstructed.shape}")

    for chunk in chunks:
        if chunk.array is None:
            continue
        unique_labels1 = np.unique(chunk.array)
        chunk_relabel = np.where(chunk.array != 0, (chunk.array) + label_max, 0)
        unique_labels2 = np.unique(chunk_relabel)
        for label in unique_labels2:
            if label == 0:
                continue
            polygon, centroid = process_mask(
                chunk_relabel,
                label,
                smooth=0,
                convex_hull=False,
                offset=np.array([0, 0]),
                x_offset=chunk.shape.context[0],
                y_offset=chunk.shape.context[1],
                return_centroid=True,
            )
            if centroid is None:
                continue
            x, y = centroid
            if chunk.shape.is_inside(x, y):
                chunk_data = ChunkData()
                chunk_data.polygons.append(polygon)
                chunk_data.centroids.append(centroid)
                chunk_data.valid_labels.add(label)

                # Draw the polygon in the reconstructed image
                draw_polygons_in_mask(
                    reconstructed,
                    chunk_data.polygons,
                    list(chunk_data.valid_labels),
                    x_offset=chunk.shape.context[0],
                    y_offset=chunk.shape.context[1],
                )
        label_max += np.max(chunk.array)
