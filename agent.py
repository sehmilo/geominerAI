"""
GeoMinerAI Agent — LangChain agent with expanded geological toolset.
Handles NLP routing to geoprocessing, classification, cross-section,
and general Q&A tools.
"""

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
    try:
        ee.Authenticate()
        ee.Initialize()
    except Exception:
        pass

# Load .env vars
load_dotenv()


# ──────────────────────────────────────────────────────────────────────
# Custom DeepSeek LLM wrapper
# ──────────────────────────────────────────────────────────────────────
class DeepSeekLLM(LLM):
    repo_id: str
    token: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = InferenceClient(model=self.repo_id, token=self.token)
        response = client.text_generation(
            prompt, max_new_tokens=512, stop_sequences=stop
        )
        return response

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    class Config:
        arbitrary_types_allowed = True


deepseek = DeepSeekLLM(
    repo_id="deepseek-ai/deepseek-llm-7b-chat",
    token=os.getenv("HUGGINGFACE_API_KEY"),
)


# ──────────────────────────────────────────────────────────────────────
# Tool wrappers
# ──────────────────────────────────────────────────────────────────────
def classify_zones_wrapper(_: str) -> str:
    """Run the full alteration zone classification pipeline."""
    try:
        aoi = ee.Geometry.Polygon(
            [
                [8.982, 10.645],
                [9.024, 10.645],
                [9.024, 10.562],
                [8.982, 10.562],
                [8.982, 10.645],
            ]
        )

        image = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate("2024-01-01", "2024-05-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .median()
            .clip(aoi)
        )

        with_indices = add_spectral_indices(image)
        classified = classify_func(with_indices)

        st.session_state["classified_image"] = classified
        return "Alteration zones classified successfully. Map layer available."
    except Exception as e:
        return f"Classification failed: {str(e)}"


def run_ioi_tool(_: str) -> str:
    """Calculate Iron Oxide and Clay Mineral indices on the Jos Plateau AOI."""
    try:
        aoi = ee.Geometry.Polygon(
            [
                [8.982, 10.645],
                [9.024, 10.645],
                [9.024, 10.562],
                [8.982, 10.562],
                [8.982, 10.645],
            ]
        )

        image = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate("2024-01-01", "2024-05-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .median()
            .clip(aoi)
        )

        result = add_spectral_indices(image)
        stats = result.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=10
        ).getInfo()

        return (
            f"Mean IOI: {stats.get('IOI')}, "
            f"Mean CMI: {stats.get('CMI')}, "
            f"Mean Fe_Ratio: {stats.get('Fe_Ratio')}, "
            f"Mean AlOH: {stats.get('AlOH_Index')}"
        )
    except Exception as e:
        return f"Error: {str(e)}"


def run_cross_section_tool(input_str: str) -> str:
    """Generate a geological cross-section along a line defined by coordinates."""
    try:
        from tools.cross_section import build_cross_section, cross_section_to_ascii
        import json

        # Try to parse coordinates from input
        # Expected: "lon1,lat1 to lon2,lat2" or JSON array
        coords = None
        try:
            coords = json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            pass

        if coords is None:
            # Default demo section across Jos Plateau
            coords = [[8.99, 10.60], [9.02, 10.57]]

        section = build_cross_section(coords)
        ascii_art = cross_section_to_ascii(section)

        st.session_state["cross_section"] = {
            "data": section,
            "ascii": ascii_art,
        }

        summary = (
            f"Cross section generated.\n"
            f"Azimuth: {section['section_azimuth']:.1f} deg\n"
            f"Length: {section['metadata']['total_length_m']:.0f} m\n"
            f"Assumptions: {'; '.join(section['assumptions'])}\n"
            f"Uncertainties: {'; '.join(section['uncertainties'])}"
        )
        return summary
    except Exception as e:
        return f"Cross section failed: {str(e)}"


def list_geoprocessing_tools(_: str) -> str:
    """List all available geoprocessing tools."""
    from tools.geoprocessing import list_tools

    tools_list = list_tools()
    lines = ["Available geoprocessing tools:"]
    for t in tools_list:
        lines.append(f"  - {t['name']}: {t['description']} (params: {', '.join(t['params'])})")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Register tools
# ──────────────────────────────────────────────────────────────────────
tools = [
    Tool(
        name="ClassifyAlterationZones",
        func=classify_zones_wrapper,
        description=(
            "Classify alteration zones using Random Forest on Sentinel-2 "
            "imagery with spectral indices and PCA. Use for mineral "
            "exploration on the Jos Plateau."
        ),
    ),
    Tool(
        name="CalculateSpectralIndices",
        func=run_ioi_tool,
        description=(
            "Calculate spectral indices (IOI, CMI, Fe_Ratio, AlOH) on "
            "the Jos Plateau AOI from Sentinel-2 imagery."
        ),
    ),
    Tool(
        name="ExtractIndices",
        func=extract_indices,
        description="Extract spectral indices for given CSV points.",
    ),
    Tool(
        name="ApplyPCA",
        func=apply_pca,
        description="Apply PCA to an Earth Engine image.",
    ),
    Tool(
        name="GenerateCrossSection",
        func=run_cross_section_tool,
        description=(
            "Generate a geological cross-section along a line. Input can be "
            "JSON coordinates [[lon1,lat1],[lon2,lat2]] or text description. "
            "Extracts topographic profile, projects units to depth, interprets "
            "structures, and assesses mineralization potential."
        ),
    ),
    Tool(
        name="ListGeoprocessingTools",
        func=list_geoprocessing_tools,
        description=(
            "List all available geoprocessing tools including buffer, clip, "
            "dissolve, hotspot analysis, convex hull, spatial join, etc."
        ),
    ),
]

# ──────────────────────────────────────────────────────────────────────
# Initialize LangChain Agent
# ──────────────────────────────────────────────────────────────────────
agent = initialize_agent(
    tools=tools,
    llm=deepseek,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)
