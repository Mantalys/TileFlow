"""
Debug visualization for tile, core, and chunk layouts.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import OVERLAP, CHUNK_SIZE
from utils import build_tiles


def visualize_layout(H: int, W: int) -> None:
    """
    Plot WSI boundary, tile bboxes (blue), core bboxes (green), and chunks (red).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")

    # Draw tiles and cores
    tiles = build_tiles(H, W)
    for t in tiles:
        x0, y0, x1, y1 = t["bbox"]
        ax.add_patch(
            Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="blue", linewidth=0.5
            )
        )
        cx0, cy0, cx1, cy1 = t["core_bbox"]
        ax.add_patch(
            Rectangle(
                (cx0, cy0),
                cx1 - cx0,
                cy1 - cy0,
                fill=False,
                edgecolor="green",
                linewidth=0.8,
            )
        )

    # Draw chunks
    for cy in range(0, H, CHUNK_SIZE - OVERLAP):
        for cx in range(0, W, CHUNK_SIZE - OVERLAP):
            ax.add_patch(
                Rectangle(
                    (cx, cy),
                    CHUNK_SIZE,
                    CHUNK_SIZE,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.0,
                )
            )

    ax.set_title("WSI Layout: Tiles (blue), Cores (green), Chunks (red)")
    plt.show()
