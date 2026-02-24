"""
File/Knowledge Base Loader for GeoMinerAI.
Supports: PDF, images, raster (GeoTIFF), Word, CSV, shapefiles, GeoJSON, KML.
All data is loaded into memory as layers for the session.
"""

import io
import json
import tempfile
import zipfile
import os

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from PIL import Image


SUPPORTED_EXTENSIONS = {
    "csv": "tabular",
    "xlsx": "tabular",
    "xls": "tabular",
    "geojson": "vector",
    "json": "vector",
    "kml": "vector",
    "kmz": "vector",
    "shp": "vector",
    "zip": "vector",       # zipped shapefile
    "gpkg": "vector",
    "tif": "raster",
    "tiff": "raster",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "bmp": "image",
    "pdf": "document",
    "docx": "document",
    "doc": "document",
    "txt": "document",
}


def get_file_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return SUPPORTED_EXTENSIONS.get(ext, "unknown")


def load_csv(file_bytes: bytes, filename: str) -> dict:
    """Load CSV/Excel into a GeoDataFrame if lat/lon columns exist, else DataFrame."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    # Try to detect coordinate columns
    lat_cols = [c for c in df.columns if c.lower() in ("lat", "latitude", "y")]
    lon_cols = [c for c in df.columns if c.lower() in ("lon", "longitude", "lng", "x")]

    if lat_cols and lon_cols:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_cols[0]], df[lat_cols[0]]),
            crs="EPSG:4326",
        )
        return {
            "type": "vector",
            "name": filename,
            "data": gdf,
            "source": "upload",
        }

    return {
        "type": "tabular",
        "name": filename,
        "data": df,
        "source": "upload",
    }


def load_vector(file_bytes: bytes, filename: str) -> dict:
    """Load vector geodata (GeoJSON, KML, Shapefile zip, GPKG)."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "geojson" or ext == "json":
        gdf = gpd.read_file(io.BytesIO(file_bytes), driver="GeoJSON")
    elif ext == "kml":
        # fiona needs KML driver enabled
        gdf = gpd.read_file(io.BytesIO(file_bytes), driver="KML")
    elif ext == "kmz":
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            kml_name = [n for n in zf.namelist() if n.endswith(".kml")][0]
            kml_bytes = zf.read(kml_name)
            gdf = gpd.read_file(io.BytesIO(kml_bytes), driver="KML")
    elif ext == "zip":
        # Zipped shapefile
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                zf.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files:
                raise ValueError("No .shp file found in the zip archive.")
            gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
    elif ext == "gpkg":
        gdf = gpd.read_file(io.BytesIO(file_bytes), driver="GPKG")
    else:
        gdf = gpd.read_file(io.BytesIO(file_bytes))

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    return {
        "type": "vector",
        "name": filename,
        "data": gdf,
        "source": "upload",
    }


def load_raster(file_bytes: bytes, filename: str) -> dict:
    """Load a raster file (GeoTIFF) into memory."""
    memfile = MemoryFile(file_bytes)
    dataset = memfile.open()
    band_data = dataset.read()
    bounds = dataset.bounds
    crs = dataset.crs
    transform = dataset.transform

    return {
        "type": "raster",
        "name": filename,
        "data": band_data,
        "bounds": bounds,
        "crs": str(crs) if crs else "EPSG:4326",
        "transform": transform,
        "width": dataset.width,
        "height": dataset.height,
        "count": dataset.count,
        "meta": dataset.meta,
        "source": "upload",
        "_memfile": memfile,
    }


def load_image(file_bytes: bytes, filename: str) -> dict:
    """Load an image file (PNG, JPG, BMP)."""
    img = Image.open(io.BytesIO(file_bytes))
    return {
        "type": "image",
        "name": filename,
        "data": img,
        "size": img.size,
        "source": "upload",
    }


def load_document(file_bytes: bytes, filename: str) -> dict:
    """Load a document (PDF, DOCX, TXT) and extract text content."""
    ext = filename.rsplit(".", 1)[-1].lower()
    text = ""

    if ext == "txt":
        text = file_bytes.decode("utf-8", errors="replace")
    elif ext == "pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            text = "[PDF support requires PyPDF2. Install with: pip install PyPDF2]"
    elif ext in ("docx", "doc"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            text = "[DOCX support requires python-docx. Install with: pip install python-docx]"

    return {
        "type": "document",
        "name": filename,
        "data": text,
        "source": "upload",
    }


def load_file(file_bytes: bytes, filename: str) -> dict:
    """Route file to appropriate loader based on extension."""
    ftype = get_file_type(filename)

    if ftype == "tabular":
        return load_csv(file_bytes, filename)
    elif ftype == "vector":
        return load_vector(file_bytes, filename)
    elif ftype == "raster":
        return load_raster(file_bytes, filename)
    elif ftype == "image":
        return load_image(file_bytes, filename)
    elif ftype == "document":
        return load_document(file_bytes, filename)
    else:
        raise ValueError(
            f"Unsupported file type: {filename}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )
