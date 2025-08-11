from tileflow.core import RegionImage, Image2D, new_image2d
from typing import List


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
        raise ValueError("Last region must have a right edge to determine full image size.")
    if not last_region.region_spec.position.edges.bottom:
        raise ValueError("Last region must have a bottom edge to determine full image size.")

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
            reconstructed[i][core_bbox.y0 : core_bbox.y1, core_bbox.x0 : core_bbox.x1] = core_image[
                i
            ]

    return reconstructed
