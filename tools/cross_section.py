"""
Geological Cross-Section Generator for GeoMinerAI.

Constructs geological cross-sections following strict structural geology principles.
Designed for mineral exploration workflows on the Jos Plateau and beyond.

INPUT:
- Section line (A to A') as coordinate pairs
- DEM / elevation data (Earth Engine SRTM or uploaded raster)
- Geological map data (lithological units, contacts, dip/strike)
- Optional: drillhole data, geophysical data

OUTPUT:
- Topographic profile along section line
- Projected geological units
- Structural interpretations
- Mineralization potential zones
- Coordinate-based cross-section representation
"""

import math
import numpy as np
import ee


# ---------------------------------------------------------------------------
# Topographic profile extraction
# ---------------------------------------------------------------------------

def extract_elevation_profile(
    line_coords: list,
    num_points: int = 200,
    dem_source: str = "USGS/SRTMGL1_003",
) -> dict:
    """
    Extract elevation profile along a section line from Earth Engine DEM.

    Args:
        line_coords: [[lon1, lat1], [lon2, lat2]] endpoints of section line
        num_points: Number of sample points along the line
        dem_source: Earth Engine DEM asset ID

    Returns:
        dict with distances (m), elevations (m), coordinates, and metadata
    """
    start = line_coords[0]
    end = line_coords[1]

    line = ee.Geometry.LineString(line_coords)
    dem = ee.Image(dem_source).select("elevation")

    # Generate sample points along the line
    lons = np.linspace(start[0], end[0], num_points)
    lats = np.linspace(start[1], end[1], num_points)

    points = []
    for lon, lat in zip(lons, lats):
        points.append(ee.Feature(ee.Geometry.Point([lon, lat])))

    fc = ee.FeatureCollection(points)
    sampled = dem.sampleRegions(collection=fc, scale=30, geometries=True)
    info = sampled.getInfo()

    elevations = []
    coords = []
    for feat in info["features"]:
        elev = feat["properties"].get("elevation", 0)
        c = feat["geometry"]["coordinates"]
        elevations.append(elev)
        coords.append(c)

    # Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(coords)):
        d = _haversine(coords[i - 1], coords[i])
        distances.append(distances[-1] + d)

    total_length = distances[-1] if distances else 0

    return {
        "distances": distances,
        "elevations": elevations,
        "coordinates": coords,
        "total_length_m": total_length,
        "num_points": len(elevations),
        "start": start,
        "end": end,
        "vertical_exaggeration": 1.0,
    }


def _haversine(coord1, coord2):
    """Calculate distance in meters between two [lon, lat] points."""
    R = 6371000
    lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
    lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Apparent dip calculation
# ---------------------------------------------------------------------------

def calculate_apparent_dip(true_dip: float, dip_direction: float, section_azimuth: float) -> float:
    """
    Calculate apparent dip as seen on a cross-section.

    Args:
        true_dip: True dip angle in degrees
        dip_direction: Dip direction azimuth in degrees
        section_azimuth: Azimuth of the section line in degrees

    Returns:
        Apparent dip in degrees (positive = dipping right, negative = dipping left)
    """
    angle_diff = math.radians(dip_direction - section_azimuth)
    apparent = math.degrees(
        math.atan(math.tan(math.radians(true_dip)) * math.cos(angle_diff))
    )
    return apparent


def section_azimuth(start: list, end: list) -> float:
    """Calculate azimuth of section line from start to end [lon, lat]."""
    dlon = math.radians(end[0] - start[0])
    lat1 = math.radians(start[1])
    lat2 = math.radians(end[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


# ---------------------------------------------------------------------------
# Geological unit projection
# ---------------------------------------------------------------------------

def project_unit_to_depth(
    surface_elevation: float,
    surface_distance: float,
    dip_angle: float,
    unit_thickness: float,
    max_depth: float = 500,
) -> dict:
    """
    Project a geological unit from surface to depth along the cross-section.

    Args:
        surface_elevation: Elevation at unit contact (m)
        surface_distance: Horizontal distance along section (m)
        dip_angle: Apparent dip angle in degrees (+ = right, - = left)
        unit_thickness: Stratigraphic thickness of unit (m)
        max_depth: Maximum projection depth below surface (m)

    Returns:
        dict with top and bottom contact coordinates for the cross-section
    """
    dip_rad = math.radians(abs(dip_angle))
    direction = 1 if dip_angle >= 0 else -1

    # Project top contact
    top_points = []
    for depth_step in np.linspace(0, max_depth, 50):
        horiz_offset = depth_step / math.tan(dip_rad) if dip_rad > 0.01 else 0
        x = surface_distance + direction * horiz_offset
        y = surface_elevation - depth_step
        top_points.append([x, y])

    # Project bottom contact (offset by thickness)
    thickness_offset = unit_thickness / math.sin(dip_rad) if dip_rad > 0.01 else unit_thickness
    bottom_points = []
    for depth_step in np.linspace(0, max_depth, 50):
        horiz_offset = depth_step / math.tan(dip_rad) if dip_rad > 0.01 else 0
        x = surface_distance + direction * (horiz_offset + thickness_offset * math.cos(dip_rad))
        y = surface_elevation - depth_step - unit_thickness * math.cos(dip_rad)
        bottom_points.append([x, y])

    return {
        "top_contact": top_points,
        "bottom_contact": bottom_points,
        "dip_angle": dip_angle,
        "thickness": unit_thickness,
        "data_constrained": True,
    }


# ---------------------------------------------------------------------------
# Structural interpretation
# ---------------------------------------------------------------------------

def interpret_fold(dip_measurements: list) -> list:
    """
    Identify folds from a series of dip measurements along the section.

    Args:
        dip_measurements: list of dicts with 'distance', 'dip', 'dip_direction'

    Returns:
        list of interpreted fold structures
    """
    if len(dip_measurements) < 3:
        return []

    folds = []
    sorted_dips = sorted(dip_measurements, key=lambda x: x["distance"])

    for i in range(1, len(sorted_dips) - 1):
        prev_dip = sorted_dips[i - 1]["dip"]
        curr_dip = sorted_dips[i]["dip"]
        next_dip = sorted_dips[i + 1]["dip"]

        prev_dir = sorted_dips[i - 1].get("dip_direction", 0)
        next_dir = sorted_dips[i + 1].get("dip_direction", 0)

        # Check for dip reversal (fold hinge)
        dir_diff = abs(prev_dir - next_dir)
        if dir_diff > 90 and dir_diff < 270:
            if curr_dip < prev_dip and curr_dip < next_dip:
                fold_type = "anticline"
            elif curr_dip > prev_dip and curr_dip > next_dip:
                fold_type = "syncline"
            else:
                fold_type = "monocline"

            folds.append({
                "type": fold_type,
                "hinge_distance": sorted_dips[i]["distance"],
                "hinge_dip": curr_dip,
                "confidence": "inferred",
            })

    return folds


def interpret_faults(
    unit_offsets: list,
) -> list:
    """
    Identify faults from stratigraphic offset observations.

    Args:
        unit_offsets: list of dicts with 'distance', 'throw', 'heave', 'sense'

    Returns:
        list of interpreted fault structures
    """
    faults = []
    for offset in unit_offsets:
        throw = offset.get("throw", 0)
        sense = offset.get("sense", "unknown")

        if sense == "normal" or (throw > 0 and sense == "unknown"):
            fault_type = "normal"
        elif sense == "reverse" or (throw < 0 and sense == "unknown"):
            fault_type = "reverse"
        elif sense == "thrust":
            fault_type = "thrust"
        else:
            fault_type = "unknown"

        faults.append({
            "type": fault_type,
            "distance": offset["distance"],
            "throw": throw,
            "heave": offset.get("heave", 0),
            "dip": offset.get("fault_dip", 60),
            "confidence": "data_constrained" if throw != 0 else "inferred",
        })

    return faults


# ---------------------------------------------------------------------------
# Mineralization assessment
# ---------------------------------------------------------------------------

def assess_mineralization_potential(
    folds: list,
    faults: list,
    units: list,
) -> list:
    """
    Identify zones of mineralization potential based on structural controls.

    Args:
        folds: List of interpreted folds
        faults: List of interpreted faults
        units: List of geological units with properties

    Returns:
        List of prospective zones with justification
    """
    prospects = []

    # Fault-fold intersections
    for fold in folds:
        for fault in faults:
            distance_between = abs(fold["hinge_distance"] - fault["distance"])
            if distance_between < 500:  # within 500m
                prospects.append({
                    "type": "structural_intersection",
                    "distance": (fold["hinge_distance"] + fault["distance"]) / 2,
                    "description": (
                        f"{fold['type'].title()} hinge near {fault['type']} fault - "
                        f"potential structural trap for mineralization"
                    ),
                    "priority": "high",
                    "recommended_action": "Drill test at intersection zone",
                })

    # Fold hinges (pressure shadows, saddle reefs)
    for fold in folds:
        if fold["type"] == "anticline":
            prospects.append({
                "type": "fold_hinge",
                "distance": fold["hinge_distance"],
                "description": "Anticlinal hinge - potential saddle reef or vein concentration",
                "priority": "medium",
                "recommended_action": "Sample for vein density and assay",
            })

    # Fault damage zones
    for fault in faults:
        prospects.append({
            "type": "fault_zone",
            "distance": fault["distance"],
            "description": (
                f"{fault['type'].title()} fault damage zone - "
                f"enhanced permeability for fluid flow"
            ),
            "priority": "medium",
            "recommended_action": "Map alteration intensity across fault zone",
        })

    return sorted(prospects, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["priority"], 3))


# ---------------------------------------------------------------------------
# Full cross-section builder
# ---------------------------------------------------------------------------

def build_cross_section(
    line_coords: list,
    geological_units: list = None,
    dip_measurements: list = None,
    fault_data: list = None,
    drillhole_data: list = None,
    vertical_exaggeration: float = 1.0,
) -> dict:
    """
    Build a complete geological cross-section.

    Args:
        line_coords: [[lon1, lat1], [lon2, lat2]]
        geological_units: list of dicts with unit info (name, thickness, contacts)
        dip_measurements: list of dicts with dip/strike data
        fault_data: list of dicts with fault observations
        drillhole_data: list of dicts with drillhole intercepts
        vertical_exaggeration: VE factor (1.0 = no exaggeration)

    Returns:
        Complete cross-section data structure
    """
    # Step 1: Section orientation
    azimuth = section_azimuth(line_coords[0], line_coords[1])

    # Step 2: Extract topographic profile
    profile = extract_elevation_profile(line_coords)

    # Apply vertical exaggeration
    if vertical_exaggeration != 1.0:
        mean_elev = np.mean(profile["elevations"])
        profile["elevations"] = [
            mean_elev + (e - mean_elev) * vertical_exaggeration
            for e in profile["elevations"]
        ]
        profile["vertical_exaggeration"] = vertical_exaggeration

    # Step 3: Process dip measurements
    apparent_dips = []
    if dip_measurements:
        for m in dip_measurements:
            app_dip = calculate_apparent_dip(
                m["dip"], m["dip_direction"], azimuth
            )
            apparent_dips.append({
                **m,
                "apparent_dip": app_dip,
                "section_azimuth": azimuth,
            })

    # Step 4: Interpret structures
    folds = interpret_fold(dip_measurements or [])
    faults = interpret_faults(fault_data or [])

    # Step 5: Project units
    projected_units = []
    if geological_units:
        for unit in geological_units:
            dip = unit.get("dip", 30)
            dip_dir = unit.get("dip_direction", 0)
            app_dip = calculate_apparent_dip(dip, dip_dir, azimuth)

            projection = project_unit_to_depth(
                surface_elevation=unit.get("surface_elevation", profile["elevations"][0]),
                surface_distance=unit.get("surface_distance", 0),
                dip_angle=app_dip,
                unit_thickness=unit.get("thickness", 50),
            )
            projection["name"] = unit.get("name", "Unknown Unit")
            projection["lithology"] = unit.get("lithology", "")
            projection["color"] = unit.get("color", "#888888")
            projected_units.append(projection)

    # Step 6: Mineralization assessment
    prospects = assess_mineralization_potential(
        folds, faults, geological_units or []
    )

    # Step 7: Drillhole projection
    projected_drillholes = []
    if drillhole_data:
        for dh in drillhole_data:
            # Find closest point on section
            dh_lon, dh_lat = dh["lon"], dh["lat"]
            min_dist = float("inf")
            proj_distance = 0
            for i, coord in enumerate(profile["coordinates"]):
                d = _haversine([dh_lon, dh_lat], coord)
                if d < min_dist:
                    min_dist = d
                    proj_distance = profile["distances"][i]

            if min_dist < 1000:  # Only project if within 1km of section
                projected_drillholes.append({
                    "name": dh.get("name", "DH"),
                    "distance_along_section": proj_distance,
                    "collar_elevation": dh.get("elevation", 0),
                    "depth": dh.get("depth", 0),
                    "intercepts": dh.get("intercepts", []),
                    "offset_from_section": min_dist,
                    "projected": min_dist > 50,
                })

    # Build assumptions and uncertainties
    assumptions = [
        f"Section azimuth: {azimuth:.1f}° (A to A')",
        f"Vertical exaggeration: {vertical_exaggeration}x",
        "DEM source: SRTM 30m resolution",
    ]
    if not dip_measurements:
        assumptions.append("No dip data provided — units projected at assumed 30° dip")
    if not fault_data:
        assumptions.append("No fault data provided — section assumes unfaulted stratigraphy")

    uncertainties = []
    if not dip_measurements:
        uncertainties.append("Subsurface geometry entirely inferred without dip measurements")
    if not drillhole_data:
        uncertainties.append("No drillhole control — depth projections are unconstrained")
    if vertical_exaggeration != 1.0:
        uncertainties.append(
            f"Vertical exaggeration ({vertical_exaggeration}x) distorts true dip angles on section"
        )

    return {
        "section_azimuth": azimuth,
        "profile": profile,
        "apparent_dips": apparent_dips,
        "projected_units": projected_units,
        "folds": folds,
        "faults": faults,
        "drillholes": projected_drillholes,
        "mineralization_prospects": prospects,
        "assumptions": assumptions,
        "uncertainties": uncertainties,
        "metadata": {
            "start": line_coords[0],
            "end": line_coords[1],
            "total_length_m": profile["total_length_m"],
            "vertical_exaggeration": vertical_exaggeration,
        },
    }


def cross_section_to_ascii(section_data: dict, width: int = 80, height: int = 25) -> str:
    """
    Generate an ASCII representation of the cross-section.
    """
    profile = section_data["profile"]
    distances = profile["distances"]
    elevations = profile["elevations"]

    if not distances or not elevations:
        return "No profile data available."

    max_dist = max(distances) if distances else 1
    min_elev = min(elevations)
    max_elev = max(elevations)
    elev_range = max_elev - min_elev if max_elev > min_elev else 1

    # Create canvas
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    # Draw topographic profile
    for i, (d, e) in enumerate(zip(distances, elevations)):
        x = int((d / max_dist) * (width - 3)) + 1
        y = height - 2 - int(((e - min_elev) / elev_range) * (height - 4))
        y = max(1, min(height - 2, y))
        x = max(1, min(width - 2, x))
        canvas[y][x] = "▓"

        # Fill below surface
        for yy in range(y + 1, height - 1):
            if canvas[yy][x] == " ":
                canvas[yy][x] = "░"

    # Mark faults
    for fault in section_data.get("faults", []):
        fd = fault["distance"]
        x = int((fd / max_dist) * (width - 3)) + 1
        x = max(1, min(width - 2, x))
        for y in range(1, height - 1):
            if canvas[y][x] in ("░", " "):
                canvas[y][x] = "│"

    # Mark fold hinges
    for fold in section_data.get("folds", []):
        fd = fold["hinge_distance"]
        x = int((fd / max_dist) * (width - 3)) + 1
        x = max(1, min(width - 2, x))
        surface_y = None
        for y in range(1, height - 1):
            if canvas[y][x] == "▓":
                surface_y = y
                break
        if surface_y:
            symbol = "∧" if fold["type"] == "anticline" else "∨"
            canvas[max(0, surface_y - 1)][x] = symbol

    # Mark mineralization prospects
    for p in section_data.get("mineralization_prospects", []):
        pd_val = p["distance"]
        x = int((pd_val / max_dist) * (width - 3)) + 1
        x = max(1, min(width - 2, x))
        canvas[0][x] = "★"

    # Draw border
    for x in range(width):
        canvas[0][x] = canvas[0][x] if canvas[0][x] == "★" else "─"
        canvas[height - 1][x] = "─"
    for y in range(height):
        canvas[y][0] = "│"
        canvas[y][width - 1] = "│"
    canvas[0][0] = "┌"
    canvas[0][width - 1] = "┐"
    canvas[height - 1][0] = "└"
    canvas[height - 1][width - 1] = "┘"

    # Add labels
    lines = ["".join(row) for row in canvas]

    header = f"  A {'─' * (width - 8)} A'"
    elev_label = f"  Elev: {min_elev:.0f}m — {max_elev:.0f}m  |  Length: {max_dist:.0f}m  |  VE: {profile.get('vertical_exaggeration', 1.0)}x"

    result = [header] + lines + [elev_label]

    # Legend
    result.append("")
    result.append("  Legend: ▓ = Surface  ░ = Subsurface  │ = Fault  ∧ = Anticline  ∨ = Syncline  ★ = Prospect")

    # Structural summary
    if section_data.get("folds"):
        result.append("")
        result.append("  Folds:")
        for f in section_data["folds"]:
            result.append(f"    {f['type'].title()} at {f['hinge_distance']:.0f}m ({f['confidence']})")

    if section_data.get("faults"):
        result.append("")
        result.append("  Faults:")
        for f in section_data["faults"]:
            result.append(f"    {f['type'].title()} at {f['distance']:.0f}m, throw={f['throw']:.0f}m ({f['confidence']})")

    if section_data.get("mineralization_prospects"):
        result.append("")
        result.append("  Mineralization Prospects:")
        for p in section_data["mineralization_prospects"]:
            result.append(f"    [{p['priority'].upper()}] {p['description']}")
            result.append(f"           → {p['recommended_action']}")

    if section_data.get("assumptions"):
        result.append("")
        result.append("  Assumptions:")
        for a in section_data["assumptions"]:
            result.append(f"    - {a}")

    if section_data.get("uncertainties"):
        result.append("")
        result.append("  Uncertainties:")
        for u in section_data["uncertainties"]:
            result.append(f"    - {u}")

    return "\n".join(result)
