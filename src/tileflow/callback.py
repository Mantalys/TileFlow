from tileflow.core import RegionImage
from typing import List


class TileFlowCallback:
    def on_tile_processed(self, region: RegionImage, tile_index: int, total_tiles: int):
        """
        Callback function to be called after each tile is processed.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to handle tile processing."
        )

    def on_chunk_processed(self, chunk_index, chunk_output):
        """
        Callback function to be called after each chunk is processed.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to handle chunk processing."
        )

    def on_processing_complete(self, regions: List[RegionImage]):
        """
        Callback function to be called when the processing is complete.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to handle completion of processing."
        )


class ProgressCallback(TileFlowCallback):
    def on_tile_processed(self, region: RegionImage, tile_index: int, total_tiles: int):
        print(
            f"Processed tile {tile_index}/{total_tiles} ({(tile_index / total_tiles) * 100:.2f}%)"
        )

    def on_chunk_processed(
        self, region: RegionImage, chunk_index: int, total_chunks: int
    ):
        print(
            f"Processed chunk {chunk_index}/{total_chunks} ({(chunk_index / total_chunks) * 100:.2f}%)"
        )
