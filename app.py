# app.py
import ee
import streamlit as st
from agent import agent
import geemap.foliumap as geemap

# Earth Engine init
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# Streamlit setup
st.set_page_config(page_title="GeoMinerAI Assistant", layout="wide")
st.title("🛰️ GeoMinerAI: Mineral Exploration Assistant")

st.markdown("""
Welcome to GeoMinerAI — your AI-powered assistant for extracting spectral indices,
applying PCA, and classifying alteration zones on the Jos Plateau using Sentinel-2 imagery.
""")

# Input prompt
query = st.text_area("🗣️ Ask a question or give a command", height=200,
                     placeholder="e.g. Classify alteration zones in the AOI...")

# Run button
if st.button("Run Agent"):
    if query.strip() == "":
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Running the assistant..."):
            try:
                response = agent.run(query)

                # Check if classified image was stored
                if "classified_image" in st.session_state:
                    classified = st.session_state["classified_image"]
                    st.success("✅ Alteration zones classified.")
                    Map = geemap.Map()
                    Map.centerObject(classified.geometry(), 13)
                    Map.addLayer(classified, {
                        'min': 0, 'max': 1,
                        'palette': ['white', 'red']
                    }, 'Classified Zones')
                    Map.to_streamlit(height=600)
                else:
                    st.success("✅ Done.")
                    st.write(response)

            except Exception as e:
                st.error(f"❌ Error: {e}")
