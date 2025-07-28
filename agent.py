# agent.py
import os
from dotenv import load_dotenv
from typing import Optional, List

import ee
import streamlit as st
from huggingface_hub import InferenceClient
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_core.language_models.llms import LLM

from tools.index_tools import extract_indices
from tools.pca import apply_pca
from tools.classifier import classify_zones as classify_func
from tools.indices import add_spectral_indices

# Earth Engine init
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# Load .env vars
load_dotenv()


# Custom DeepSeek LLM wrapper
class DeepSeekLLM(LLM):
    repo_id: str
    token: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = InferenceClient(model=self.repo_id, token=self.token)
        response = client.text_generation(prompt, max_new_tokens=256, stop_sequences=stop)
        return response

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    class Config:
        arbitrary_types_allowed = True


deepseek = DeepSeekLLM(
    repo_id="deepseek-ai/deepseek-llm-7b-chat",
    token=os.getenv("HUGGINGFACE_API_KEY")
)


# Wrapper for classification
def classify_zones_wrapper(_: str) -> str:
    try:
        aoi = ee.Geometry.Polygon([
            [8.982, 10.645],
            [9.024, 10.645],
            [9.024, 10.562],
            [8.982, 10.562],
            [8.982, 10.645]
        ])

        image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(aoi) \
            .filterDate('2024-01-01', '2024-05-01') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .median().clip(aoi)

        with_indices = add_spectral_indices(image)
        classified = classify_func(with_indices)

        st.session_state["classified_image"] = classified
        return "✅ Alteration zones classified. Map is available."
    except Exception as e:
        return f"❌ Classification failed: {str(e)}"


# IOI calculation tool
def run_ioi_tool(_: str) -> str:
    try:
        aoi = ee.Geometry.Polygon([
            [8.982, 10.645],
            [9.024, 10.645],
            [9.024, 10.562],
            [8.982, 10.562],
            [8.982, 10.645]
        ])

        image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(aoi) \
            .filterDate('2024-01-01', '2024-05-01') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .median().clip(aoi)

        result = add_spectral_indices(image)
        stats = result.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=10).getInfo()

        return f"Mean IOI: {stats.get('IOI')}, Mean CMI: {stats.get('CMI')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Register tools
tools = [
    Tool(name="ExtractIndices", func=extract_indices, description="Extract spectral indices for given points"),
    Tool(name="ApplyPCA", func=apply_pca, description="Apply PCA to indices image"),
    Tool(name="ClassifyZones", func=classify_zones_wrapper, description="Use Random Forest to classify alteration zones"),
    Tool(name="Calculate_Spectral_Indices", func=run_ioi_tool, description="Calculate IOI and CMI on Jos Plateau")
]

# Initialize LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=deepseek,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
