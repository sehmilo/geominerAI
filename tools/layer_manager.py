"""
In-memory Layer Manager for GeoMinerAI.
Manages all uploaded and generated layers during a session.
Layers can be vector (GeoDataFrame), raster (numpy), EE images, documents, or images.
"""

import uuid
import json
from datetime import datetime

import geopandas as gpd
import numpy as np
from shapely.geometry import shape, mapping


class LayerManager:
    """Session-scoped layer store. All layers live in memory."""

    def __init__(self):
        self._layers = {}  # id -> layer dict
        self._order = []   # ordered list of layer ids (bottom to top)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_layer(self, layer: dict) -> str:
        """
        Add a layer to the manager. Returns the layer ID.

        Expected layer dict keys:
            type: 'vector' | 'raster' | 'ee_image' | 'image' | 'document' | 'tabular' | 'drawn'
            name: display name
            data: the actual data object
            source: 'upload' | 'generated' | 'drawn'
            visible: bool (default True)
            ... additional type-specific keys
        """
        layer_id = str(uuid.uuid4())[:8]
        layer.setdefault("id", layer_id)
        layer.setdefault("visible", True)
        layer.setdefault("created_at", datetime.now().isoformat())
        layer.setdefault("source", "unknown")

        self._layers[layer_id] = layer
        self._order.append(layer_id)
        return layer_id

    def remove_layer(self, layer_id: str) -> bool:
        """Remove a layer by ID."""
        if layer_id in self._layers:
            del self._layers[layer_id]
            self._order.remove(layer_id)
            return True
        return False

    def get_layer(self, layer_id: str) -> dict:
        """Get a layer by ID."""
        return self._layers.get(layer_id)

    def get_layer_by_name(self, name: str) -> dict:
        """Get first layer matching a name (case-insensitive)."""
        name_lower = name.lower()
        for lid in self._order:
            layer = self._layers[lid]
            if layer.get("name", "").lower() == name_lower:
                return layer
        return None

    def list_layers(self) -> list:
        """Return ordered list of layer summaries (for UI display)."""
        summaries = []
        for lid in self._order:
            layer = self._layers[lid]
            summary = {
                "id": lid,
                "name": layer.get("name", "Unnamed"),
                "type": layer.get("type", "unknown"),
                "visible": layer.get("visible", True),
                "source": layer.get("source", "unknown"),
                "created_at": layer.get("created_at", ""),
            }

            # Add type-specific metadata
            if layer["type"] == "vector" and isinstance(layer.get("data"), gpd.GeoDataFrame):
                gdf = layer["data"]
                summary["feature_count"] = len(gdf)
                summary["geometry_type"] = gdf.geom_type.unique().tolist() if len(gdf) > 0 else []
                summary["crs"] = str(gdf.crs) if gdf.crs else "Unknown"
            elif layer["type"] == "raster":
                summary["shape"] = list(layer["data"].shape) if hasattr(layer.get("data"), "shape") else None
                summary["crs"] = layer.get("crs", "Unknown")
            elif layer["type"] == "ee_image":
                summary["ee"] = True
            elif layer["type"] == "tabular":
                summary["row_count"] = len(layer["data"]) if hasattr(layer.get("data"), "__len__") else 0
            elif layer["type"] == "document":
                text = layer.get("data", "")
                summary["char_count"] = len(text) if isinstance(text, str) else 0

            summaries.append(summary)

        return summaries

    def toggle_visibility(self, layer_id: str) -> bool:
        """Toggle layer visibility. Returns new visibility state."""
        if layer_id in self._layers:
            self._layers[layer_id]["visible"] = not self._layers[layer_id]["visible"]
            return self._layers[layer_id]["visible"]
        return False

    def reorder(self, layer_ids: list):
        """Set new layer order."""
        valid = [lid for lid in layer_ids if lid in self._layers]
        # Append any layers not in the new order at the end
        for lid in self._order:
            if lid not in valid:
                valid.append(lid)
        self._order = valid

    def move_layer_up(self, layer_id: str):
        """Move layer one position up (toward top)."""
        if layer_id in self._order:
            idx = self._order.index(layer_id)
            if idx < len(self._order) - 1:
                self._order[idx], self._order[idx + 1] = self._order[idx + 1], self._order[idx]

    def move_layer_down(self, layer_id: str):
        """Move layer one position down (toward bottom)."""
        if layer_id in self._order:
            idx = self._order.index(layer_id)
            if idx > 0:
                self._order[idx], self._order[idx - 1] = self._order[idx - 1], self._order[idx]

    def rename_layer(self, layer_id: str, new_name: str):
        """Rename a layer."""
        if layer_id in self._layers:
            self._layers[layer_id]["name"] = new_name

    def clear_all(self):
        """Remove all layers."""
        self._layers.clear()
        self._order.clear()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_visible_layers(self) -> list:
        """Return only visible layers in order."""
        return [self._layers[lid] for lid in self._order if self._layers[lid].get("visible", True)]

    def get_vector_layers(self) -> list:
        """Return all vector layers."""
        return [
            self._layers[lid]
            for lid in self._order
            if self._layers[lid].get("type") in ("vector", "drawn")
        ]

    def get_raster_layers(self) -> list:
        """Return all raster layers."""
        return [
            self._layers[lid]
            for lid in self._order
            if self._layers[lid].get("type") == "raster"
        ]

    def get_ee_layers(self) -> list:
        """Return all Earth Engine layers."""
        return [
            self._layers[lid]
            for lid in self._order
            if self._layers[lid].get("type") == "ee_image"
        ]

    def get_all_bounds(self) -> tuple:
        """Get combined bounds of all visible vector layers. Returns (minx, miny, maxx, maxy) or None."""
        all_bounds = []
        for layer in self.get_visible_layers():
            if layer["type"] in ("vector", "drawn") and isinstance(layer.get("data"), gpd.GeoDataFrame):
                gdf = layer["data"]
                if len(gdf) > 0 and gdf.crs:
                    gdf_4326 = gdf.to_crs("EPSG:4326") if str(gdf.crs) != "EPSG:4326" else gdf
                    all_bounds.append(gdf_4326.total_bounds)

        if not all_bounds:
            return None

        bounds = np.array(all_bounds)
        return (bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max())

    # ------------------------------------------------------------------
    # Serialization (for layer list display)
    # ------------------------------------------------------------------

    def layer_names(self) -> list:
        """Return list of (id, name) tuples."""
        return [(lid, self._layers[lid].get("name", "Unnamed")) for lid in self._order]

    def count(self) -> int:
        return len(self._layers)
