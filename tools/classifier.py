import ee
import os
import pandas as pd
import geemap


folder_path = 'outputs'
filename = 'tulu.csv'
file_path = os.path.join(folder_path, filename)
# Load training points from CSV
pits_df = pd.read_csv(file_path)

# Create FeatureCollection from CSV
features = []
for _, row in pits_df.iterrows():
    geom = ee.Geometry.Point([row['lon'], row['lat']])
    feat = ee.Feature(geom, {'class': 1})  # class 1 for alteration zone
    features.append(feat)

training_fc = ee.FeatureCollection(features)

# Function to sample image and classify
def classify_zones(image: ee.Image) -> ee.Image:
    bands = ['IOI', 'CMI', 'Fe_Ratio', 'AlOH_Index', 'PC1', 'PC2', 'PC3', 'SWIR_PC1', 'SWIR_PC2']
    image = image.select(bands)

    training = image.sampleRegions(
        collection=training_fc,
        properties=['class'],
        scale=10
    )

    classifier = ee.Classifier.smileRandomForest(10).train(
        features=training,
        classProperty='class',
        inputProperties=bands
    )

    classified = image.classify(classifier)

    # Store globally for app.py to retrieve (since agent expects only str)
    import streamlit as st
    st.session_state["classified_image"] = classified

    return classified
