import ee

def sample_pits(final_img, csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    feats = [ee.Feature(ee.Geometry.Point([row.lon,row.lat]), {'pit_id':str(row.pit_id)}) for _,row in df.iterrows()]
    fc = ee.FeatureCollection(feats)
    sampled = final_img.sampleRegions(fc, scale=10, geometries=True)
    task = ee.batch.Export.table.toDrive(sampled, 'tulu_indices', folder='outputs', fileNamePrefix='tulu_indices', fileFormat='CSV')
    task.start()
    return "Export started"
