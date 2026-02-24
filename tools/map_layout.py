"""
Map Layout & Export for GeoMinerAI.
Provides data extraction and export capabilities from the map display.
Supports exporting layers to various formats and generating print-ready layouts.
"""

import io
import json
import base64
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd


# ---------------------------------------------------------------------------
# Export layer data
# ---------------------------------------------------------------------------

def export_layer_to_geojson(layer: dict) -> str:
    """Export a vector layer to GeoJSON string."""
    if layer["type"] not in ("vector", "drawn"):
        raise ValueError(f"Cannot export {layer['type']} layer as GeoJSON")

    gdf = layer["data"]
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Layer data is not a GeoDataFrame")

    return gdf.to_json()


def export_layer_to_csv(layer: dict) -> str:
    """Export a layer to CSV string."""
    if layer["type"] == "tabular":
        df = layer["data"]
    elif layer["type"] in ("vector", "drawn"):
        gdf = layer["data"]
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        if "geometry" in gdf.columns:
            df["longitude"] = gdf.geometry.x if all(gdf.geometry.geom_type == "Point") else None
            df["latitude"] = gdf.geometry.y if all(gdf.geometry.geom_type == "Point") else None
    else:
        raise ValueError(f"Cannot export {layer['type']} layer as CSV")

    return df.to_csv(index=False)


def export_layer_to_shapefile_zip(layer: dict) -> bytes:
    """Export a vector layer to a zipped shapefile (bytes)."""
    if layer["type"] not in ("vector", "drawn"):
        raise ValueError(f"Cannot export {layer['type']} layer as Shapefile")

    gdf = layer["data"]
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        import zipfile
        shp_path = os.path.join(tmpdir, "export.shp")
        gdf.to_file(shp_path)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmpdir):
                fpath = os.path.join(tmpdir, fname)
                if os.path.isfile(fpath):
                    zf.write(fpath, fname)

        return zip_buffer.getvalue()


def export_layer_to_kml(layer: dict) -> str:
    """Export a vector layer to KML string."""
    if layer["type"] not in ("vector", "drawn"):
        raise ValueError(f"Cannot export {layer['type']} layer as KML")

    gdf = layer["data"]
    if gdf.crs and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
        gdf.to_file(tmp.name, driver="KML")
        tmp.seek(0)
        with open(tmp.name, "r") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Data extraction from map display
# ---------------------------------------------------------------------------

def extract_data_at_point(layers: list, lon: float, lat: float) -> list:
    """
    Extract data values from all layers at a given point.

    Args:
        layers: list of layer dicts
        lon: longitude
        lat: latitude

    Returns:
        list of dicts with layer name and extracted values
    """
    from shapely.geometry import Point
    point = Point(lon, lat)
    results = []

    for layer in layers:
        if layer["type"] in ("vector", "drawn"):
            gdf = layer["data"]
            if not isinstance(gdf, gpd.GeoDataFrame) or len(gdf) == 0:
                continue
            if gdf.crs and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            # Point-in-polygon or nearest feature
            containing = gdf[gdf.contains(point)]
            if len(containing) > 0:
                row = containing.iloc[0]
                attrs = {col: str(row[col]) for col in containing.columns if col != "geometry"}
                results.append({
                    "layer": layer["name"],
                    "type": "contains",
                    "attributes": attrs,
                })
            else:
                # Find nearest feature
                distances = gdf.geometry.distance(point)
                nearest_idx = distances.idxmin()
                nearest = gdf.loc[nearest_idx]
                attrs = {col: str(nearest[col]) for col in gdf.columns if col != "geometry"}
                results.append({
                    "layer": layer["name"],
                    "type": "nearest",
                    "distance_deg": float(distances.min()),
                    "attributes": attrs,
                })

        elif layer["type"] == "raster":
            data = layer.get("data")
            bounds = layer.get("bounds")
            if data is not None and bounds is not None:
                minx, miny, maxx, maxy = bounds
                if minx <= lon <= maxx and miny <= lat <= maxy:
                    # Sample raster at point
                    h, w = data.shape[-2], data.shape[-1]
                    col = int((lon - minx) / (maxx - minx) * w)
                    row = int((maxy - lat) / (maxy - miny) * h)
                    col = max(0, min(w - 1, col))
                    row = max(0, min(h - 1, row))
                    values = data[:, row, col] if data.ndim == 3 else data[row, col]
                    results.append({
                        "layer": layer["name"],
                        "type": "raster_sample",
                        "values": values.tolist() if hasattr(values, "tolist") else [float(values)],
                    })

    return results


def extract_data_in_polygon(layers: list, polygon_geojson: dict) -> list:
    """
    Extract/clip data from all layers within a polygon.

    Args:
        layers: list of layer dicts
        polygon_geojson: GeoJSON geometry dict of the polygon

    Returns:
        list of clipped layer data
    """
    from shapely.geometry import shape as shapely_shape
    poly = shapely_shape(polygon_geojson)
    clip_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    results = []

    for layer in layers:
        if layer["type"] in ("vector", "drawn"):
            gdf = layer["data"]
            if not isinstance(gdf, gpd.GeoDataFrame) or len(gdf) == 0:
                continue
            if gdf.crs and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            clipped = gpd.clip(gdf, clip_gdf)
            if len(clipped) > 0:
                results.append({
                    "layer": layer["name"],
                    "type": "vector_clip",
                    "feature_count": len(clipped),
                    "data": clipped,
                })

    return results


# ---------------------------------------------------------------------------
# Layout metadata (for print/export layout)
# ---------------------------------------------------------------------------

def generate_layout_metadata(
    layers: list,
    title: str = "GeoMinerAI Map",
    author: str = "GeoMinerAI",
    scale: str = "",
    notes: str = "",
) -> dict:
    """Generate metadata for a map layout export."""
    visible_layers = [l for l in layers if l.get("visible", True)]

    return {
        "title": title,
        "author": author,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scale": scale,
        "notes": notes,
        "layer_count": len(visible_layers),
        "layers": [
            {
                "name": l.get("name", "Unnamed"),
                "type": l.get("type", "unknown"),
                "source": l.get("source", "unknown"),
            }
            for l in visible_layers
        ],
    }


# ---------------------------------------------------------------------------
# Download helpers for Streamlit
# ---------------------------------------------------------------------------

def get_download_data(layer: dict, fmt: str = "geojson") -> tuple:
    """
    Prepare layer data for Streamlit download button.

    Returns:
        (data_bytes, filename, mime_type)
    """
    name = layer.get("name", "export").replace(" ", "_")

    if fmt == "geojson":
        data = export_layer_to_geojson(layer).encode("utf-8")
        return data, f"{name}.geojson", "application/geo+json"
    elif fmt == "csv":
        data = export_layer_to_csv(layer).encode("utf-8")
        return data, f"{name}.csv", "text/csv"
    elif fmt == "shapefile":
        data = export_layer_to_shapefile_zip(layer)
        return data, f"{name}.zip", "application/zip"
    elif fmt == "kml":
        data = export_layer_to_kml(layer).encode("utf-8")
        return data, f"{name}.kml", "application/vnd.google-earth.kml+xml"
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
