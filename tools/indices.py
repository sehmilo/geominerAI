# tools/indices.py

import ee

def add_spectral_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndvi_mask = ndvi.lt(0.3)  # mask dense vegetation

    ioi = image.expression('(b4 - b2) / (b4 + b2)', {
        'b4': image.select('B4'), 'b2': image.select('B2')
    }).rename('IOI')

    cmi = image.expression('(b11 - b12) / (b11 + b12)', {
        'b11': image.select('B11'), 'b12': image.select('B12')
    }).rename('CMI')

    fe_ratio = image.expression('b4 / b3', {
        'b4': image.select('B4'), 'b3': image.select('B3')
    }).rename('Fe_Ratio')

    aloh_index = image.expression('(b11 - b8) / (b11 + b8)', {
        'b11': image.select('B11'), 'b8': image.select('B8')
    }).rename('AlOH_Index')

    combined = ee.Image.cat([ioi, cmi, fe_ratio, aloh_index]).updateMask(ndvi_mask)

    return image.addBands(combined)
