"""
Geoprocessing engine for GeoMinerAI.
Provides buffer, clip, hotspot analysis, spatial joins, and ML classification.
All operations work on in-memory GeoDataFrames or Earth Engine objects.
"""

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import ee


# ---------------------------------------------------------------------------
# Vector geoprocessing
# ---------------------------------------------------------------------------

def buffer_layer(gdf: gpd.GeoDataFrame, distance_m: float) -> gpd.GeoDataFrame:
    """Buffer all geometries by distance in meters."""
    if gdf.crs and gdf.crs.is_geographic:
        projected = gdf.to_crs(epsg=3857)
        projected["geometry"] = projected.geometry.buffer(distance_m)
        return projected.to_crs(gdf.crs)
    else:
        result = gdf.copy()
        result["geometry"] = result.geometry.buffer(distance_m)
        return result


def clip_layer(
    gdf: gpd.GeoDataFrame, clip_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Clip gdf by the geometry of clip_gdf."""
    return gpd.clip(gdf, clip_gdf)


def spatial_join(
    left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, how: str = "inner"
) -> gpd.GeoDataFrame:
    """Spatial join between two GeoDataFrames."""
    return gpd.sjoin(left, right, how=how, predicate="intersects")


def dissolve_layer(gdf: gpd.GeoDataFrame, by: str = None) -> gpd.GeoDataFrame:
    """Dissolve geometries, optionally grouping by a column."""
    if by and by in gdf.columns:
        return gdf.dissolve(by=by).reset_index()
    return gpd.GeoDataFrame(
        geometry=[unary_union(gdf.geometry)], crs=gdf.crs
    )


def centroid_layer(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return centroids of all features."""
    result = gdf.copy()
    result["geometry"] = result.geometry.centroid
    return result


def convex_hull(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return convex hull of the entire dataset."""
    hull = unary_union(gdf.geometry).convex_hull
    return gpd.GeoDataFrame(geometry=[hull], crs=gdf.crs)


# ---------------------------------------------------------------------------
# Hotspot / kernel density analysis
# ---------------------------------------------------------------------------

def kernel_density_hotspot(
    gdf: gpd.GeoDataFrame,
    bandwidth_m: float = 500,
    grid_size: int = 100,
) -> dict:
    """
    Simple kernel density estimation for point data.
    Returns a raster-like grid as a numpy array with bounds.
    """
    if gdf.crs and gdf.crs.is_geographic:
        pts = gdf.to_crs(epsg=3857)
    else:
        pts = gdf.copy()

    coords = np.array([(g.x, g.y) for g in pts.geometry if g is not None])
    if len(coords) == 0:
        raise ValueError("No valid point geometries found.")

    xmin, ymin, xmax, ymax = pts.total_bounds
    pad = bandwidth_m * 2
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad

    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(xi, yi)

    density = np.zeros_like(xx)
    for cx, cy in coords:
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        density += np.exp(-dist_sq / (2 * bandwidth_m ** 2))

    # Normalize to 0-1
    if density.max() > 0:
        density = density / density.max()

    return {
        "type": "raster",
        "name": "hotspot_density",
        "data": density[np.newaxis, :, :],  # (1, H, W)
        "bounds": (xmin, ymin, xmax, ymax),
        "crs": "EPSG:3857",
        "grid_size": grid_size,
        "source": "generated",
    }


# ---------------------------------------------------------------------------
# Earth Engine based geoprocessing
# ---------------------------------------------------------------------------

def ee_clip_image(image: ee.Image, geometry: ee.Geometry) -> ee.Image:
    """Clip an Earth Engine image to a geometry."""
    return image.clip(geometry)


def ee_buffer_geometry(geometry: ee.Geometry, distance_m: float) -> ee.Geometry:
    """Buffer an Earth Engine geometry."""
    return geometry.buffer(distance_m)


def ee_zonal_stats(
    image: ee.Image, zones: ee.FeatureCollection, scale: int = 30
) -> dict:
    """Calculate zonal statistics for an image within feature zones."""
    stats = image.reduceRegions(
        collection=zones,
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.minMax(), sharedInputs=True
        ).combine(ee.Reducer.stdDev(), sharedInputs=True),
        scale=scale,
    )
    return stats.getInfo()


# ---------------------------------------------------------------------------
# ML classification (wraps existing tools)
# ---------------------------------------------------------------------------

def classify_alteration_zones(aoi_geojson: dict = None) -> dict:
    """
    Run the full mineral exploration classification pipeline.
    Returns the classified ee.Image and metadata.
    """
    from tools.indices import add_spectral_indices
    from tools.pca import apply_pca
    from tools.classifier import classify_zones

    if aoi_geojson:
        aoi = ee.Geometry(aoi_geojson)
    else:
        # Default: Jos Plateau, Nigeria
        aoi = ee.Geometry.Polygon([
            [8.982, 10.645],
            [9.024, 10.645],
            [9.024, 10.562],
            [8.982, 10.562],
            [8.982, 10.645],
        ])

    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate("2024-01-01", "2024-05-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .median()
        .clip(aoi)
    )

    with_indices = add_spectral_indices(image)

    # PCA on all bands then SWIR subset
    all_bands = ["B2", "B3", "B4", "B8", "B11", "B12", "IOI", "CMI", "Fe_Ratio", "AlOH_Index"]
    pca_full = apply_pca(with_indices, all_bands, aoi, comps=3)
    swir_bands = ["B11", "B12"]
    pca_swir = apply_pca(with_indices, swir_bands, aoi, comps=2)
    pca_swir = pca_swir.select(["PC1", "PC2"]).rename(["SWIR_PC1", "SWIR_PC2"])

    stacked = with_indices.addBands(pca_full).addBands(pca_swir)
    classified = classify_zones(stacked)

    return {
        "type": "ee_image",
        "name": "alteration_classification",
        "data": classified,
        "geometry": aoi,
        "vis_params": {"min": 0, "max": 1, "palette": ["white", "red"]},
        "source": "generated",
    }


# ---------------------------------------------------------------------------
# Tool dispatcher - maps NLP-extracted intent to geoprocessing function
# ---------------------------------------------------------------------------

GEOPROCESSING_TOOLS = {
    "buffer": {
        "func": buffer_layer,
        "description": "Create buffer zones around features",
        "params": ["layer", "distance_m"],
    },
    "clip": {
        "func": clip_layer,
        "description": "Clip a layer by another layer's extent",
        "params": ["layer", "clip_layer"],
    },
    "dissolve": {
        "func": dissolve_layer,
        "description": "Dissolve geometries into a single feature",
        "params": ["layer", "by"],
    },
    "centroid": {
        "func": centroid_layer,
        "description": "Calculate centroids of features",
        "params": ["layer"],
    },
    "convex_hull": {
        "func": convex_hull,
        "description": "Generate convex hull around all features",
        "params": ["layer"],
    },
    "spatial_join": {
        "func": spatial_join,
        "description": "Join two layers based on spatial relationship",
        "params": ["left_layer", "right_layer"],
    },
    "hotspot": {
        "func": kernel_density_hotspot,
        "description": "Kernel density hotspot analysis for point data",
        "params": ["layer", "bandwidth_m"],
    },
    "classify": {
        "func": classify_alteration_zones,
        "description": "Run alteration zone classification (ML pipeline)",
        "params": ["aoi_geojson"],
    },
}


def list_tools() -> list:
    """Return list of available geoprocessing tools with descriptions."""
    return [
        {"name": k, "description": v["description"], "params": v["params"]}
        for k, v in GEOPROCESSING_TOOLS.items()
    ]
