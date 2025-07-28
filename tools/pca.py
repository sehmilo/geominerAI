import ee

def apply_pca(image, bands, region, comps=3):
    img = image.select(bands)
    mean_dict = img.reduceRegion(ee.Reducer.mean(), region, 10)
    means = ee.Image.constant(mean_dict.values(bands)).rename(bands)
    centered = img.subtract(means)
    cov = centered.toArray().reduceRegion(ee.Reducer.centeredCovariance(), region, 10)
    eigens = ee.Array(cov.get('array')).eigen()
    vectors = ee.Image.constant(eigens.slice(1,0,comps))
    arr = centered.toArray().toArray(1)
    proj = vectors.matrixMultiply(arr).arrayProject([0]).arrayFlatten([['PC'+str(i+1) for i in range(comps)]])
    return proj
