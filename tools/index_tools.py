import ee
import pandas as pd

def extract_indices(csv_path: str, image: ee.Image) -> str:
    # Load CSV with lat/lon
    df = pd.read_csv(csv_path)
    features = []
    
    for _, row in df.iterrows():
        point = ee.Geometry.Point([row['lon'], row['lat']])
        feat = ee.Feature(point).set('id', row['pit_id'])
        features.append(feat)

    fc = ee.FeatureCollection(features)

    # Sample values from the image at pit locations
    sampled = image.sampleRegions(
        collection=fc,
        scale=10,
        geometries=True
    )

    # Export sampled data to CSV format as a string (simulated return for now)
    url = sampled.getDownloadURL('CSV')  # or export to Drive in production

    return f"Spectral indices extracted and available at: {url}"

