import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import time
import os
import json
from skimage.measure import find_contours

from SWIVVEL import run_analysis_on_folder


# SIDEBAR - CONFIGURATION

st.set_page_config(layout="wide", page_title="Vortex Detector")

st.sidebar.title("Detection Control")

# Data Loading
st.sidebar.header("1. Data Source")
data_folder = st.sidebar.text_input("Dataset Folder Path", value="./datasets/test_data")
target_file = st.sidebar.file_uploader("Targets JSON (Optional)", type=["json"])

# Load Targets if uploaded
targets_data = {}
if target_file is not None:
    try:
        targets_data = json.load(target_file)
        st.sidebar.success(f"Loaded targets for {len(targets_data)} frames.")
    except Exception as e:
        st.sidebar.error(f"Error loading JSON: {e}")

# Parameters
st.sidebar.header("2. Detection Parameters")

with st.sidebar.expander("Segmentation (Masking)", expanded=False):
    ow_thresh_str = st.text_input("OW Threshold", value="-9.71e-13")
    ow_threshold = float(ow_thresh_str)
    min_area = st.number_input("Min Area (px)", value=17)
    closing_size = st.number_input("Closing Size", value=64)
    erosion_size = st.number_input("Erosion Size", value=4)
    scale_factor = st.number_input("Scale Factor", value=2)

with st.sidebar.expander("Analysis & Radii", expanded=False):
    analysis_radius = st.number_input("Analysis Radius (km)", value=909)
    radius_max = st.number_input("Max Radius (km)", value=378)
    radius_step = st.number_input("Radius Step (km)", value=49)
    radius_cutoff = st.slider("Radius Cutoff", 0.0, 1.0, 0.145)

with st.sidebar.expander("Scoring Weights", expanded=True):
    w_vort = st.number_input("Weight: Vorticity", value=94.38)
    w_aspect = st.number_input("Weight: Aspect Ratio", value=181.19)
    w_consist = st.number_input("Weight: Consistency", value=125.31)
    w_persist = st.number_input("Weight: Persistence", value=46.35)
    score_thresh = st.number_input("Total Score Threshold", value=112.18)

with st.sidebar.expander("Tracking", expanded=False):
    max_track_dist = st.number_input("Max Tracking Dist (km)", value=1000)
    persist_thresh = st.slider("Persistent Consistency Thresh", 0.0, 1.0, 0.5)

# Construct Params Dictionary
detection_params = {
    "ow_threshold": ow_threshold,
    "min_area": int(min_area),
    "closing_size": int(closing_size),
    "erosion_size": int(erosion_size),
    "scale_factor": int(scale_factor),
    "analysis_radius_km": int(analysis_radius),
    "radius_max_km": int(radius_max),
    "radius_step_km": int(radius_step),
    "radius_cutoff_threshold": radius_cutoff,
    "max_tracking_distance_km": int(max_track_dist),
    "weight_vorticity": w_vort,
    "weight_aspect_ratio": w_aspect,
    "weight_consistency": w_consist,
    "weight_persistence": w_persist,
    "total_score_threshold": score_thresh
}

run_btn = st.sidebar.button("RUN ANALYSIS", type="primary")

# ANALYSIS EXECUTION

if 'results' not in st.session_state:
    st.session_state.results = {}
if 'sorted_keys' not in st.session_state:
    st.session_state.sorted_keys = []

if "map_center" not in st.session_state:
    st.session_state.map_center = [20, -60]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 3

if run_btn:
    if not os.path.exists(data_folder):
        st.error(f"Folder not found: {data_folder}")
    else:
        with st.spinner("Running detection algorithm... This may take a moment."):
            st.session_state.results = {}
            results = run_analysis_on_folder(data_folder, detection_params)

            if results:
                st.session_state.results = results
                st.session_state.sorted_keys = sorted(results.keys())
                st.success(f"Processed {len(results)} frames successfully!")
            else:
                st.warning("No results found. Check your parameters or data folder.")

# VISUALIZATION DASHBOARD

st.title("Vortex Tracking Dashboard")

if not st.session_state.results:
    st.info("Please verify parameters and click 'RUN ANALYSIS' in the sidebar.")
    st.stop()

# Controls Row
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])

if 'play_idx' not in st.session_state:
    st.session_state.play_idx = 0

with col_ctrl1:
    play = st.checkbox("Auto-Play Animation")
    speed = st.slider("Playback Speed (sec/frame)", 0.5, 3.0, 1.0, help="Wait time between frames.")

with col_ctrl2:
    show_contours = st.checkbox("Show Contours", value=True)
    show_centers = st.checkbox("Show Centers (Red)", value=True)
    show_vectors = st.checkbox("Show Wind Vectors", value=False)

with col_ctrl3:
    show_radii = st.checkbox("Show Radii", value=True)
    show_trails = st.checkbox("Show Trails (Cyan)", value=True)
    show_targets = st.checkbox("Show Targets (Orange)", value=True)
    vector_stride = st.slider("Vector Stride", 10, 100, 40)

# Frame Selection Logic
if play:
    selected_idx = st.session_state.play_idx
    st.info(f"Playing Frame: {selected_idx + 1}/{len(st.session_state.sorted_keys)}")
else:
    selected_idx = st.slider("Frame Timeline", 0, len(st.session_state.sorted_keys) - 1, st.session_state.play_idx)
    st.session_state.play_idx = selected_idx

# Get data for current frame
current_key = st.session_state.sorted_keys[selected_idx]
frame_data = st.session_state.results[current_key]
centers = frame_data['centers']
mask = frame_data['region']['mask']
data_obj = frame_data['data']
Lon = data_obj['Lon']
Lat = data_obj['Lat']
u_grid = data_obj['u']
v_grid = data_obj['v']

st.subheader(f"Frame: {current_key} | Detections: {len(centers)}")

# MAP GENERATION

def get_track_history(current_idx, v_id, max_history=10):
    path = []
    for i in range(max(0, current_idx - max_history), current_idx + 1):
        k = st.session_state.sorted_keys[i]
        day_centers = st.session_state.results[k]['centers']
        found = next((c for c in day_centers if c.get('id') == v_id), None)
        if found:
            path.append((found['lat'], found['lon']))
    return path

def get_target_history(targets_dict, current_idx, sorted_keys, target_name, max_history=10):
    path = []
    start_idx = max(0, current_idx - max_history)
    for i in range(start_idx, current_idx + 1):
        k = sorted_keys[i]
        if k in targets_dict:
            day_targets = targets_dict[k]
            found = next((t for t in day_targets if t.get('name') == target_name), None)
            if found:
                path.append((found['lat'], found['lon']))
    return path

# Initialize Map using Session State for Persistence
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom,
    tiles="CartoDB dark_matter"
)

# Draw Wind Vectors
if show_vectors:
    ys, xs = np.arange(0, Lat.shape[0], vector_stride), np.arange(0, Lat.shape[1], vector_stride)
    mesh_ys, mesh_xs = np.meshgrid(ys, xs, indexing='ij')
    flat_ys = mesh_ys.flatten()
    flat_xs = mesh_xs.flatten()

    try:
        lats_sub = Lat[flat_ys, flat_xs]
        lons_sub = Lon[flat_ys, flat_xs]
        u_sub = u_grid.values[flat_ys, flat_xs] if hasattr(u_grid, 'values') else u_grid[flat_ys, flat_xs]
        v_sub = v_grid.values[flat_ys, flat_xs] if hasattr(v_grid, 'values') else v_grid[flat_ys, flat_xs]
    except Exception as e:
        lats_sub = []

    vec_scale = 0.2
    for i in range(len(lats_sub)):
        lat_s, lon_s = float(lats_sub[i]), float(lons_sub[i])
        u_v, v_v = float(u_sub[i]), float(v_sub[i])
        if np.isnan(u_v) or np.isnan(v_v) or (u_v == 0 and v_v == 0): continue
        lat_e = lat_s + (v_v * vec_scale)
        lon_e = lon_s + (u_v * vec_scale)
        folium.PolyLine([(lat_s, lon_s), (lat_e, lon_e)], color="gray", weight=1, opacity=0.6).add_to(m)

# Draw Contours
if show_contours:
    contours = find_contours(mask, 0.5)
    for contour in contours:
        mapped_contour = []
        for r, c in contour:
            r_int, c_int = int(r), int(c)
            if 0 <= r_int < Lat.shape[0] and 0 <= c_int < Lat.shape[1]:
                mapped_contour.append((Lat[r_int, c_int], Lon[r_int, c_int]))
        if len(mapped_contour) > 2:
            folium.Polygon(mapped_contour, color="lime", weight=1, fill=True, fill_color="lime", fill_opacity=0.3).add_to(m)

# Loop through Detected Centers
for vortex in centers:
    lat, lon = vortex['lat'], vortex['lon']
    v_id = vortex.get('id')
    v_id_str = str(v_id) if v_id is not None else "New"
    radius_km = vortex.get('radius_km', 50)

    if show_trails and v_id is not None:
        history = get_track_history(selected_idx, v_id)
        if len(history) > 1:
            folium.PolyLine(history, color="cyan", weight=2.5, opacity=0.7, dash_array='5, 5').add_to(m)

    if show_radii:
        folium.Circle([lat, lon], radius=radius_km * 1000, color="red", weight=1, fill=False, opacity=0.5).add_to(m)

    if show_centers:
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 220px;">
            <h5 style="margin:0;">Vortex {v_id_str}</h5>
            <hr style="margin: 5px 0;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>Total Score:</b></td><td style="text-align:right;">{vortex.get('total_score', 0):.2f}</td></tr>
                <tr><td><b>Radius:</b></td><td style="text-align:right;">{radius_km:.1f} km</td></tr>
                <tr><td><b>Area:</b></td><td style="text-align:right;">{vortex.get('area', 0)} px</td></tr>
                <tr><td><b>Consistency:</b></td><td style="text-align:right;">{vortex.get('weighted_consistency_score', 0):.3f}</td></tr>
                <tr><td><b>Aspect Ratio:</b></td><td style="text-align:right;">{vortex.get('aspect_ratio', 0):.2f}</td></tr>
                <tr><td><b>Peak Vort.:</b></td><td style="text-align:right;">{vortex.get('peak_vorticity', 0):.2e}</td></tr>
                <tr><td><b>Mean Vort.:</b></td><td style="text-align:right;">{vortex.get('mean_vorticity', 0):.2e}</td></tr>
            </table>
        </div>
        """
        folium.CircleMarker([lat, lon], radius=4, color="red", fill=True, fill_color="white", fill_opacity=1.0, popup=folium.Popup(popup_html, max_width=200)).add_to(m)

# Draw Targets
if show_targets and targets_data:
    if current_key in targets_data:
        frame_targets = targets_data[current_key]
        for t in frame_targets:
            t_lat, t_lon, t_name = t['lat'], t['lon'], t['name']

            folium.Marker(
                location=[t_lat, t_lon],
                tooltip=f"Target: {t_name}",
                icon=folium.Icon(color="orange", icon="info-sign")
            ).add_to(m)

            target_path = get_target_history(
                targets_data,
                selected_idx,
                st.session_state.sorted_keys,
                t_name
            )
            if len(target_path) > 1:
                folium.PolyLine(
                    target_path,
                    color="orange",
                    weight=3,
                    opacity=0.8,
                    dash_array='2, 5',
                    tooltip=f"Path: {t_name}"
                ).add_to(m)

# Capture Map State (Zoom/Center)
# Only return zoom/center if playing (to persist view across auto-reloads).
# If static (not playing), return nothing to allow free panning without reloads.
ret_objects = ["zoom", "center"] if play else []

st_map_data = st_folium(m, width="100%", height=500, returned_objects=ret_objects)

# Update session state safely (Fix for KeyError and Persistence)
if st_map_data:
    if st_map_data.get("zoom") is not None:
        st.session_state.map_zoom = st_map_data["zoom"]

    # Safe dictionary access
    center_data = st_map_data.get("center")
    if center_data is not None:
        st.session_state.map_center = [center_data["lat"], center_data["lng"]]

# DATA TABLE & PLAYBACK UPDATE
if centers:
    with st.expander(f"Detailed Data for {current_key}", expanded=False):
        df = pd.DataFrame(centers)
        display_cols = ['id', 'lat', 'lon', 'total_score', 'radius_km', 'weighted_consistency_score']
        final_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[final_cols].style.highlight_max(axis=0))

if play:
    time.sleep(speed)
    st.session_state.play_idx = (st.session_state.play_idx + 1) % len(st.session_state.sorted_keys)
    st.rerun()

