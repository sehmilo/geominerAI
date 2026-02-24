"""
GeoMinerAI - Unified Mineral Exploration Interface
Single-tab layout with 4 panels:
  1. Live Map (upper area, interactive with drawing tools)
  2. Layer List (left sidebar, shows all uploaded/generated layers)
  3. Prompt Box (lower right, file upload + NLP chat)
  4. Map Layout (export & data extraction controls in sidebar)
"""

import json
import re
import ee
import streamlit as st
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from tools.layer_manager import LayerManager
from tools.file_loader import load_file, SUPPORTED_EXTENSIONS
from tools.map_layout import (
    get_download_data,
    extract_data_at_point,
    extract_data_in_polygon,
    generate_layout_metadata,
)

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoMinerAI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Earth Engine init
# ──────────────────────────────────────────────────────────────────────
if "ee_initialized" not in st.session_state:
    try:
        ee.Initialize()
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
        except Exception:
            pass  # EE may not be available in all environments
    st.session_state["ee_initialized"] = True

# ──────────────────────────────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────────────────────────────
if "layer_manager" not in st.session_state:
    st.session_state["layer_manager"] = LayerManager()
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "drawn_features" not in st.session_state:
    st.session_state["drawn_features"] = []
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [9.7, 8.5]  # Jos Plateau default
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 8

lm: LayerManager = st.session_state["layer_manager"]


# ──────────────────────────────────────────────────────────────────────
# Helper: build folium map with all visible layers
# ──────────────────────────────────────────────────────────────────────
def build_map() -> folium.Map:
    m = folium.Map(
        location=st.session_state["map_center"],
        zoom_start=st.session_state["map_zoom"],
        control_scale=True,
        tiles="OpenStreetMap",
    )

    # Satellite basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    # Terrain basemap
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Terrain",
    ).add_to(m)

    # Render visible vector layers
    for layer in lm.get_visible_layers():
        if layer["type"] in ("vector", "drawn") and isinstance(
            layer.get("data"), gpd.GeoDataFrame
        ):
            gdf = layer["data"]
            if len(gdf) == 0:
                continue
            if gdf.crs and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            geojson_data = json.loads(gdf.to_json())
            style = layer.get("style", {})
            tooltip_fields = []
            if (
                geojson_data.get("features")
                and geojson_data["features"][0].get("properties")
            ):
                tooltip_fields = list(
                    geojson_data["features"][0]["properties"].keys()
                )[:5]

            folium.GeoJson(
                geojson_data,
                name=layer.get("name", "Layer"),
                style_function=lambda x, s=style: {
                    "fillColor": s.get("fill_color", "#3388ff"),
                    "color": s.get("color", "#3388ff"),
                    "weight": s.get("weight", 2),
                    "fillOpacity": s.get("fill_opacity", 0.3),
                },
                tooltip=folium.GeoJsonTooltip(fields=tooltip_fields)
                if tooltip_fields
                else None,
            ).add_to(m)

    # Drawing controls
    Draw(
        draw_options={
            "polyline": True,
            "polygon": True,
            "rectangle": True,
            "circle": False,
            "marker": True,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ──────────────────────────────────────────────────────────────────────
# Helper: process NLP query through the agent
# ──────────────────────────────────────────────────────────────────────
def process_query(query: str) -> str:
    """Route user query through the LangChain agent."""
    try:
        from agent import agent

        response = agent.run(query)

        # Check if a classified image was generated
        if "classified_image" in st.session_state:
            classified = st.session_state.pop("classified_image")
            lm.add_layer(
                {
                    "type": "ee_image",
                    "name": "Alteration Classification",
                    "data": classified,
                    "vis_params": {
                        "min": 0,
                        "max": 1,
                        "palette": ["white", "red"],
                    },
                    "source": "generated",
                }
            )
            return response + "\n\n*Classification layer added to the map.*"

        return response
    except Exception as e:
        return f"Error: {str(e)}"


# ──────────────────────────────────────────────────────────────────────
# Helper: process drawn features from map
# ──────────────────────────────────────────────────────────────────────
def process_drawn_features(map_data: dict):
    """Save drawn geometries from the map as layers."""
    if not map_data:
        return
    all_drawings = map_data.get("all_drawings") or []
    if not all_drawings:
        return

    existing_count = len(st.session_state["drawn_features"])
    new_drawings = all_drawings[existing_count:]

    for i, drawing in enumerate(new_drawings):
        geom_type = drawing.get("geometry", {}).get("type", "Unknown")
        gdf = gpd.GeoDataFrame.from_features([drawing], crs="EPSG:4326")
        layer_name = f"Drawing {existing_count + i + 1} ({geom_type})"
        lm.add_layer(
            {
                "type": "drawn",
                "name": layer_name,
                "data": gdf,
                "source": "drawn",
                "style": {
                    "color": "#ff7800",
                    "fill_color": "#ff7800",
                    "weight": 3,
                    "fill_opacity": 0.2,
                },
            }
        )
        st.session_state["drawn_features"].append(drawing)


# ══════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════

# ── SIDEBAR: Layer List + Map Layout / Export ─────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ Layers")

    layers = lm.list_layers()
    if not layers:
        st.caption(
            "No layers loaded. Upload files or draw on the map to get started."
        )
    else:
        for layer_info in reversed(layers):  # top layers first
            lid = layer_info["id"]
            col1, col2, col3 = st.columns([0.1, 0.7, 0.2])

            with col1:
                visible = st.checkbox(
                    "vis",
                    value=layer_info["visible"],
                    key=f"vis_{lid}",
                    label_visibility="collapsed",
                )
                if visible != layer_info["visible"]:
                    lm.toggle_visibility(lid)
                    st.rerun()

            with col2:
                icon = {
                    "vector": "📍",
                    "raster": "🗺️",
                    "ee_image": "🛰️",
                    "drawn": "✏️",
                    "tabular": "📊",
                    "document": "📄",
                    "image": "🖼️",
                }.get(layer_info["type"], "📎")

                label = f"{icon} {layer_info['name']}"
                if layer_info["type"] == "vector":
                    label += f" ({layer_info.get('feature_count', '?')} feat)"
                elif layer_info["type"] == "tabular":
                    label += f" ({layer_info.get('row_count', '?')} rows)"

                st.markdown(
                    f"<small>{label}</small>", unsafe_allow_html=True
                )

            with col3:
                if st.button("🗑️", key=f"del_{lid}", help="Remove layer"):
                    lm.remove_layer(lid)
                    st.rerun()

        st.divider()

    if lm.count() > 0:
        if st.button("🧹 Clear All Layers"):
            lm.clear_all()
            st.session_state["drawn_features"] = []
            st.rerun()

    st.divider()

    # Map Layout & Export
    st.markdown("## 📐 Map Layout & Export")

    vector_layers = [l for l in layers if l["type"] in ("vector", "drawn")]

    if vector_layers:
        export_layer_name = st.selectbox(
            "Select layer to export",
            options=[l["name"] for l in vector_layers],
            key="export_select",
        )
        export_fmt = st.selectbox(
            "Format",
            options=["geojson", "csv", "shapefile", "kml"],
            key="export_fmt",
        )

        if st.button("⬇️ Prepare Download"):
            layer = lm.get_layer_by_name(export_layer_name)
            if layer:
                try:
                    data_bytes, filename, mime = get_download_data(
                        layer, export_fmt
                    )
                    st.download_button(
                        label=f"Save {filename}",
                        data=data_bytes,
                        file_name=filename,
                        mime=mime,
                        key="download_btn",
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")
    else:
        st.caption("Upload or generate vector layers to enable export.")

    with st.expander("Layout Settings"):
        layout_title = st.text_input(
            "Map title", value="GeoMinerAI Map", key="layout_title"
        )
        layout_author = st.text_input(
            "Author", value="GeoMinerAI", key="layout_author"
        )
        layout_notes = st.text_area("Notes", key="layout_notes", height=80)

        if st.button("Generate Layout Info"):
            meta = generate_layout_metadata(
                [lm.get_layer(l["id"]) for l in layers],
                title=layout_title,
                author=layout_author,
                notes=layout_notes,
            )
            st.json(meta)


# ── MAIN AREA ─────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='margin-bottom:0'>🛰️ GeoMinerAI</h2>"
    "<p style='color:gray;margin-top:0'>"
    "AI-Powered Mineral Exploration Platform</p>",
    unsafe_allow_html=True,
)

# ── Upper section: Live Map ───────────────────────────────────────────
map_col, info_col = st.columns([3, 1])

with map_col:
    folium_map = build_map()
    map_data = st_folium(
        folium_map,
        width=None,
        height=500,
        returned_objects=["all_drawings", "last_object_clicked"],
        key="main_map",
    )

    if map_data:
        process_drawn_features(map_data)
        clicked = map_data.get("last_object_clicked")
        if clicked:
            st.session_state["last_click"] = clicked

with info_col:
    st.markdown("#### 📌 Map Info")

    if "last_click" in st.session_state and st.session_state["last_click"]:
        click = st.session_state["last_click"]
        lat = click.get("lat", 0)
        lng = click.get("lng", 0)
        st.markdown(f"**Clicked:** {lat:.5f}, {lng:.5f}")

        visible_layers = lm.get_visible_layers()
        if visible_layers:
            results = extract_data_at_point(visible_layers, lng, lat)
            if results:
                for r in results:
                    with st.expander(f"📍 {r['layer']}"):
                        st.json(r.get("attributes", r))

    st.divider()
    st.markdown(f"**Layers:** {lm.count()}")
    bounds = lm.get_all_bounds()
    if bounds:
        st.markdown(
            f"**Extent:** {bounds[0]:.3f}, {bounds[1]:.3f} "
            f"to {bounds[2]:.3f}, {bounds[3]:.3f}"
        )

    # Cross-section quick launcher
    st.divider()
    st.markdown("#### ✂️ Cross Section")
    st.caption("Draw a line on the map (A to A'), then click below.")
    if st.button("🔬 Generate Cross Section", key="xsec_btn"):
        drawn = st.session_state.get("drawn_features", [])
        lines = [
            d
            for d in drawn
            if d.get("geometry", {}).get("type") in ("LineString", "Polyline")
        ]
        if lines:
            from tools.cross_section import (
                build_cross_section,
                cross_section_to_ascii,
            )

            line_geom = lines[-1]["geometry"]
            coords = line_geom["coordinates"]
            if len(coords) >= 2:
                section_data = build_cross_section([coords[0], coords[-1]])
                ascii_section = cross_section_to_ascii(section_data)
                st.session_state["cross_section"] = {
                    "data": section_data,
                    "ascii": ascii_section,
                }
                st.success("Cross section generated!")
        else:
            st.warning("Draw a line on the map first (A to A').")

st.divider()

# ── Lower section: Prompt Box + Output ────────────────────────────────
prompt_col, output_col = st.columns([1, 1])

with prompt_col:
    st.markdown("#### 💬 Prompt & Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload files (geodata, PDF, images, CSV, raster, Word...)",
        accept_multiple_files=True,
        type=list(SUPPORTED_EXTENSIONS.keys()),
        key="file_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if lm.get_layer_by_name(uf.name):
                continue
            try:
                file_bytes = uf.read()
                layer = load_file(file_bytes, uf.name)
                lm.add_layer(layer)
                st.success(f"Loaded: {uf.name} ({layer['type']})")
            except Exception as e:
                st.error(f"Failed to load {uf.name}: {e}")

    st.divider()

    user_query = st.text_area(
        "Ask GeoMinerAI anything...",
        height=120,
        placeholder=(
            "Examples:\n"
            "  Classify alteration zones in the AOI\n"
            "  Buffer the uploaded points by 500m\n"
            "  Run hotspot analysis on the borehole data\n"
            "  Generate a cross section along the drawn line\n"
            "  What are the spectral indices for this region?"
        ),
        key="user_prompt",
    )

    col_run, col_clear = st.columns(2)
    with col_run:
        run_clicked = st.button(
            "▶️ Run", type="primary", use_container_width=True
        )
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    if run_clicked and user_query.strip():
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_query}
        )

        with st.spinner("GeoMinerAI is thinking..."):
            query_lower = user_query.lower()

            if "cross section" in query_lower or "cross-section" in query_lower:
                from tools.cross_section import (
                    build_cross_section,
                    cross_section_to_ascii,
                )

                drawn = st.session_state.get("drawn_features", [])
                lines = [
                    d
                    for d in drawn
                    if d.get("geometry", {}).get("type")
                    in ("LineString", "Polyline")
                ]
                if lines:
                    line_geom = lines[-1]["geometry"]
                    coords = line_geom["coordinates"]
                    section_data = build_cross_section(
                        [coords[0], coords[-1]]
                    )
                    ascii_section = cross_section_to_ascii(section_data)
                    st.session_state["cross_section"] = {
                        "data": section_data,
                        "ascii": ascii_section,
                    }
                    response = (
                        "Cross section generated along the drawn line. "
                        "See the output panel for the full structural "
                        "interpretation."
                    )
                else:
                    response = (
                        "Please draw a line on the map first to define "
                        "the section line (A to A')."
                    )

            elif "buffer" in query_lower:
                from tools.geoprocessing import buffer_layer

                dist_match = re.search(r"(\d+)\s*m", query_lower)
                dist = float(dist_match.group(1)) if dist_match else 500.0
                vectors = lm.get_vector_layers()
                if vectors:
                    target = vectors[-1]
                    buffered = buffer_layer(target["data"], dist)
                    lm.add_layer(
                        {
                            "type": "vector",
                            "name": f"{target['name']} (buffer {dist}m)",
                            "data": buffered,
                            "source": "generated",
                            "style": {
                                "color": "#ff0000",
                                "fill_color": "#ff000044",
                                "weight": 2,
                                "fill_opacity": 0.15,
                            },
                        }
                    )
                    response = (
                        f"Buffer of {dist}m applied to "
                        f"'{target['name']}'. New layer added."
                    )
                else:
                    response = (
                        "No vector layers available to buffer. "
                        "Upload data first."
                    )

            elif "clip" in query_lower:
                from tools.geoprocessing import clip_layer

                vectors = lm.get_vector_layers()
                if len(vectors) >= 2:
                    clipped = clip_layer(
                        vectors[-2]["data"], vectors[-1]["data"]
                    )
                    lm.add_layer(
                        {
                            "type": "vector",
                            "name": f"{vectors[-2]['name']} (clipped)",
                            "data": clipped,
                            "source": "generated",
                        }
                    )
                    response = (
                        f"Clipped '{vectors[-2]['name']}' by "
                        f"'{vectors[-1]['name']}'."
                    )
                else:
                    response = "Need at least 2 vector layers for clipping."

            elif "hotspot" in query_lower or "density" in query_lower:
                from tools.geoprocessing import kernel_density_hotspot

                vectors = lm.get_vector_layers()
                point_layers = [
                    v
                    for v in vectors
                    if isinstance(v.get("data"), gpd.GeoDataFrame)
                    and len(v["data"]) > 0
                    and all(v["data"].geometry.geom_type == "Point")
                ]
                if point_layers:
                    target = point_layers[-1]
                    hotspot = kernel_density_hotspot(target["data"])
                    lm.add_layer(hotspot)
                    response = (
                        f"Hotspot density analysis completed on "
                        f"'{target['name']}'. Layer added."
                    )
                else:
                    response = (
                        "No point layers available for hotspot analysis. "
                        "Upload point data first."
                    )

            elif any(
                kw in query_lower
                for kw in ["classify", "alteration", "mineral"]
            ):
                response = process_query(user_query)

            elif "dissolve" in query_lower:
                from tools.geoprocessing import dissolve_layer

                vectors = lm.get_vector_layers()
                if vectors:
                    target = vectors[-1]
                    dissolved = dissolve_layer(target["data"])
                    lm.add_layer(
                        {
                            "type": "vector",
                            "name": f"{target['name']} (dissolved)",
                            "data": dissolved,
                            "source": "generated",
                        }
                    )
                    response = (
                        f"Dissolved '{target['name']}'. New layer added."
                    )
                else:
                    response = "No vector layers available to dissolve."

            elif "convex" in query_lower or "hull" in query_lower:
                from tools.geoprocessing import convex_hull

                vectors = lm.get_vector_layers()
                if vectors:
                    target = vectors[-1]
                    hull_result = convex_hull(target["data"])
                    lm.add_layer(
                        {
                            "type": "vector",
                            "name": f"{target['name']} (convex hull)",
                            "data": hull_result,
                            "source": "generated",
                        }
                    )
                    response = (
                        f"Convex hull generated for '{target['name']}'."
                    )
                else:
                    response = "No vector layers available."

            else:
                response = process_query(user_query)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response}
        )
        st.rerun()

with output_col:
    st.markdown("#### 📊 Output")

    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**GeoMinerAI:** {msg['content']}")

    if "cross_section" in st.session_state:
        xsec = st.session_state["cross_section"]
        st.divider()
        st.markdown("#### ✂️ Geological Cross Section")
        st.code(xsec["ascii"], language=None)

        with st.expander("Section Details"):
            data = xsec["data"]
            st.markdown(f"**Azimuth:** {data['section_azimuth']:.1f}°")
            st.markdown(
                f"**Length:** {data['metadata']['total_length_m']:.0f} m"
            )
            st.markdown(
                f"**VE:** {data['metadata']['vertical_exaggeration']}x"
            )

            if data.get("assumptions"):
                st.markdown("**Assumptions:**")
                for a in data["assumptions"]:
                    st.markdown(f"- {a}")

            if data.get("uncertainties"):
                st.markdown("**Uncertainties:**")
                for u in data["uncertainties"]:
                    st.markdown(f"- {u}")

            if data.get("mineralization_prospects"):
                st.markdown("**Mineralization Prospects:**")
                for p in data["mineralization_prospects"]:
                    st.markdown(
                        f"- **[{p['priority'].upper()}]** "
                        f"{p['description']}"
                    )
                    st.markdown(f"  → {p['recommended_action']}")
