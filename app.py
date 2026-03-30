import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import os
import json
import requests
import ollama as ollama_client
from skimage.measure import find_contours

from SWIVVEL import run_analysis_on_folder


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Vortex Detector")


# SIDEBAR
st.sidebar.title("Detection Control")

# ── 1. Data Source ─────────────────────────────────────────────────────────────
st.sidebar.header("1. Data Source")
data_folder = st.sidebar.text_input("Dataset Folder Path", value="./datasets/test_data")
target_file = st.sidebar.file_uploader("Targets JSON (Optional)", type=["json"])

targets_data = {}
if target_file is not None:
    try:
        targets_data = json.load(target_file)
        st.sidebar.success(f"Loaded targets for {len(targets_data)} frames.")
    except Exception as e:
        st.sidebar.error(f"Error loading JSON: {e}")

# ── 2. Detection Parameters ────────────────────────────────────────────────────
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
    w_vort    = st.number_input("Weight: Vorticity",    value=94.38)
    w_aspect  = st.number_input("Weight: Aspect Ratio", value=181.19)
    w_consist = st.number_input("Weight: Consistency",  value=125.31)
    w_persist = st.number_input("Weight: Persistence",  value=46.35)
    score_thresh = st.number_input("Total Score Threshold", value=112.18)

with st.sidebar.expander("Tracking", expanded=False):
    max_track_dist  = st.number_input("Max Tracking Dist (km)", value=1000)
    persist_thresh  = st.slider("Persistent Consistency Thresh", 0.0, 1.0, 0.5)

detection_params = {
    "ow_threshold":             ow_threshold,
    "min_area":                 int(min_area),
    "closing_size":             int(closing_size),
    "erosion_size":             int(erosion_size),
    "scale_factor":             int(scale_factor),
    "analysis_radius_km":       int(analysis_radius),
    "radius_max_km":            int(radius_max),
    "radius_step_km":           int(radius_step),
    "radius_cutoff_threshold":  radius_cutoff,
    "max_tracking_distance_km": int(max_track_dist),
    "weight_vorticity":         w_vort,
    "weight_aspect_ratio":      w_aspect,
    "weight_consistency":       w_consist,
    "weight_persistence":       w_persist,
    "total_score_threshold":    score_thresh,
}

# ── 3. LLM Insights (Ollama) ───────────────────────────────────────────────────
st.sidebar.header("3. LLM Insights (Ollama)")
ollama_host  = st.sidebar.text_input("Ollama Host",  value="http://localhost:11434")
ollama_model = st.sidebar.text_input("Ollama Model", value="llama3.1:8b")
ranges_file  = st.sidebar.file_uploader(
    "Value Ranges JSON (Optional)",
    type=["json"],
    help="Upload the value_ranges.json produced by get_value_ranges.py to give the LLM better context on typical metric ranges."
)

value_ranges = {}
if ranges_file is not None:
    try:
        value_ranges = json.load(ranges_file).get("metrics", {})
        st.sidebar.success("Value ranges loaded.")
    except Exception as e:
        st.sidebar.warning(f"Could not parse ranges file: {e}")

run_btn = st.sidebar.button("RUN ANALYSIS", type="primary")


# SESSION STATE INITIALISATION
for key, default in [
    ("results", {}),
    ("sorted_keys", []),
    ("map_center", [20, -60]),
    ("map_zoom", 3),
    ("play_idx", 0),
    ("selected_vortex", None),
    ("llm_response", None),
    ("last_click_key", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ANALYSIS EXECUTION
if run_btn:
    if not os.path.exists(data_folder):
        st.error(f"Folder not found: {data_folder}")
    else:
        with st.spinner("Running detection algorithm… This may take a moment."):
            st.session_state.results = {}
            results = run_analysis_on_folder(data_folder, detection_params)

            if results:
                st.session_state.results      = results
                st.session_state.sorted_keys  = sorted(results.keys())
                st.session_state.selected_vortex = None
                st.session_state.llm_response    = None
                st.success(f"Processed {len(results)} frames successfully!")
            else:
                st.warning("No results found. Check your parameters or data folder.")


# LLM HELPER
def _range_context(key: str, label: str) -> str:
    """Return a compact range string if we have stats for this metric."""
    s = value_ranges.get(key)
    if not s or s.get("n", 0) == 0:
        return ""
    return (
        f"  [dataset range — median: {s['median']:.3g}, "
        f"p25–p75: {s['p25']:.3g}–{s['p75']:.3g}, "
        f"max: {s['max']:.3g}]"
    )


def build_llm_prompt(v: dict, params: dict) -> str:
    sb  = v.get("score_breakdown", {})
    rm  = v.get("raw_metrics", {})
    total = v.get("total_score", 0.0)
    def pct(x):
        if not np.isfinite(total) or total < 1e-9 or not np.isfinite(x):
            return "n/a"
        return f"{100 * x / total:.0f}%"

    tracked = sb.get("persistence", 0) > 0

    range_lines = ""
    if value_ranges:
        range_lines = (
            "\nDataset percentile context (for reference):\n"
            + f"  • Vorticity/Coriolis ratio{_range_context('metric_vorticity', '')}\n"
            + f"  • Flow consistency       {_range_context('metric_consistency', '')}\n"
            + f"  • Total score            {_range_context('total_score', '')}\n"
            + f"  • Radius (km)            {_range_context('radius_km', '')}\n"
        )

    prompt = f"""You are an expert oceanographer / meteorologist interpreting an automatically detected atmospheric or oceanic vortex from satellite-derived wind stress data.

Below are the quantitative characteristics of the detection, each with a brief explanation of what the statistic represents.

--- LOCATION & SCALE ---
  Location    : {v['lat']:.2f}°N, {v['lon']:.2f}°E
    (Geographic centre of the detected vortex in decimal degrees.)

  Radius      : {v['radius_km']:.0f} km
    (Estimated outer radius of organised circular flow, found by stepping outward from
    the centre until the flow consistency drops below the cutoff threshold.
    Mesoscale ocean eddies are typically 50–300 km; tropical disturbances 300–800 km.)

--- VORTICITY ---
  Peak vorticity  : {v['peak_vorticity']:.3e} s⁻¹
    (Maximum curl of the wind/current field inside the detected region.
    Positive = cyclonic in the Northern Hemisphere. Typical mesoscale values are
    1e-5 to 1e-4 s⁻¹; values near 0 suggest weak rotation.)

  Mean vorticity  : {v['mean_vorticity']:.3e} s⁻¹
    (Average vorticity across all pixels in the detected region.
    A large gap between peak and mean suggests rotation concentrated in a small core.)

  Vorticity / Coriolis ratio : {rm.get('metric_vorticity', 0):.2f}  (Rossby number proxy)
    (Peak vorticity divided by the local Coriolis parameter f = 2Ω sin(lat).
    Values >> 1 indicate strongly ageostrophic, intense rotation;
    values near 1 indicate geostrophically balanced, large-scale flow.)

--- SHAPE ---
  Aspect ratio    : {v['aspect_ratio']:.2f}
    (Ratio of the longest to shortest axis of the detected region (eigenvalue method).
    1.0 = perfect circle; higher values indicate an elongated or sheared feature.
    Circular vortices typically score below 2.0; streaks or fronts score much higher.)

  Shape score metric : {rm.get('metric_aspect', 0):.3f}  (= 1 / aspect ratio, 0–1 scale)
    (Inverse aspect ratio used internally. 1.0 = perfectly circular, 0 = infinitely elongated.)

--- FLOW ORGANISATION ---
  Flow consistency : {v['weighted_consistency_score']:.3f}  (0–1 scale)
    (Magnitude-weighted fraction of the tangential velocity around the vortex centre
    that is coherently rotating in one direction. 1.0 means all flow is perfectly
    circular; 0.5 means half the flow is counter-rotating or radial noise.
    Values above 0.6 are considered well-organised.)

--- DETECTION SCORING ---
  Total score     : {total:.2f}  (threshold to pass detection: {params['total_score_threshold']:.2f})
    (Weighted sum of the four test scores below. A vortex must exceed the threshold
    to be reported. Higher scores indicate stronger, more confident detections.)

  Score breakdown (weighted contribution to total — must sum to > {params['total_score_threshold']:.2f}):
    Vorticity    : {sb.get('vorticity', 0):.2f}  ({pct(sb.get('vorticity', 0))} of total)
      (= Rossby proxy {rm.get('metric_vorticity', 0):.3f} × weight {params['weight_vorticity']:.2f}.
      Rewards intense rotation relative to local Coriolis. No individual cap — unbounded.)
    Shape        : {sb.get('shape', 0):.2f}  ({pct(sb.get('shape', 0))} of total)
      (= circularity {rm.get('metric_aspect', 0):.3f} × weight {params['weight_aspect_ratio']:.2f}.
      Raw metric is 0–1, so max possible contribution is {params['weight_aspect_ratio']:.2f}.)
    Consistency  : {sb.get('consistency', 0):.2f}  ({pct(sb.get('consistency', 0))} of total)
      (= flow score {rm.get('metric_consistency', 0):.3f} × weight {params['weight_consistency']:.2f}.
      Raw metric is 0–1, so max possible contribution is {params['weight_consistency']:.2f}.)
    Persistence  : {sb.get('persistence', 0):.2f}  ({pct(sb.get('persistence', 0))} of total)
      (= 1 if tracked from previous frame, 0 if new × weight {params['weight_persistence']:.2f}.
      Binary — either 0 or {params['weight_persistence']:.2f}.)

  Tracked from previous frame : {'Yes — persistent feature' if tracked else 'No — new detection this frame'}
    (Whether this vortex was matched to a detection within {params['max_tracking_distance_km']:.0f} km in the prior frame.)
{range_lines}
--- YOUR TASK ---
In 4–6 sentences provide:
1. A likely geophysical interpretation (e.g. mesoscale eddy, tropical disturbance, wind curl feature) given the location and scale.
2. What the dominant score contributor(s) reveal about the physical character of this vortex.
3. A confidence assessment — flag any metrics that could indicate a false positive.
4. Any actionable recommendation (e.g. worth monitoring, likely transient noise, consistent with a known feature type).

Be concise and technical. Do not repeat the raw numbers verbatim."""

    return prompt


def call_ollama(prompt: str, host: str, model: str) -> str:
    """Generate a response via the ollama Python package."""
    try:
        client = ollama_client.Client(host=host)
        response = client.generate(model=model, prompt=prompt)
        # Response is a GenerateResponse object in newer versions
        if hasattr(response, "response"):
            return response.response.strip()
        elif isinstance(response, dict):
            return response.get("response", str(response)).strip()
        return str(response).strip()
    except ollama_client.ResponseError as e:
        return f"Ollama error: {e.error}"
    except Exception as exc:
        return f"Could not reach Ollama ({exc}). Check the host setting in the sidebar."


# VISUALIZATION DASHBOARD
st.title("Vortex Tracking Dashboard")

if not st.session_state.results:
    st.info("Please verify parameters and click **RUN ANALYSIS** in the sidebar.")
    st.stop()

# ── Playback controls ──────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])

with col_ctrl1:
    play  = st.checkbox("Auto-Play Animation")
    speed = st.slider("Playback Speed (sec/frame)", 0.5, 3.0, 1.0)

with col_ctrl2:
    show_contours = st.checkbox("Show Contours",      value=True)
    show_centers  = st.checkbox("Show Centers (Red)", value=True)
    show_vectors  = st.checkbox("Show Wind Vectors",  value=False)

with col_ctrl3:
    show_radii   = st.checkbox("Show Radii",           value=True)
    show_trails  = st.checkbox("Show Trails (Cyan)",   value=True)
    show_targets = st.checkbox("Show Targets (Orange)", value=True)
    vector_stride = st.slider("Vector Stride", 10, 100, 40)

# ── Frame selection ────────────────────────────────────────────────────────────
if play:
    selected_idx = st.session_state.play_idx
    st.info(f"Playing Frame: {selected_idx + 1}/{len(st.session_state.sorted_keys)}")
else:
    selected_idx = st.slider(
        "Frame Timeline", 0, len(st.session_state.sorted_keys) - 1,
        st.session_state.play_idx
    )
    st.session_state.play_idx = selected_idx

current_key = st.session_state.sorted_keys[selected_idx]
frame_data  = st.session_state.results[current_key]
centers     = frame_data["centers"]
mask        = frame_data["region"]["mask"]
data_obj    = frame_data["data"]
Lon         = data_obj["Lon"]
Lat         = data_obj["Lat"]
u_grid      = data_obj["u"]
v_grid      = data_obj["v"]

st.subheader(f"Frame: {current_key}  |  Detections: {len(centers)}")


# ── Helper functions ───────────────────────────────────────────────────────────
def get_track_history(current_idx, v_id, max_history=10):
    path = []
    for i in range(max(0, current_idx - max_history), current_idx + 1):
        k = st.session_state.sorted_keys[i]
        found = next((c for c in st.session_state.results[k]["centers"] if c.get("id") == v_id), None)
        if found:
            path.append((found["lat"], found["lon"]))
    return path


def get_target_history(targets_dict, current_idx, sorted_keys, target_name, max_history=10):
    path = []
    for i in range(max(0, current_idx - max_history), current_idx + 1):
        k = sorted_keys[i]
        if k in targets_dict:
            found = next((t for t in targets_dict[k] if t.get("name") == target_name), None)
            if found:
                path.append((found["lat"], found["lon"]))
    return path


# MAP
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom,
    tiles="CartoDB dark_matter",
)

# Wind vectors
if show_vectors:
    ys = np.arange(0, Lat.shape[0], vector_stride)
    xs = np.arange(0, Lat.shape[1], vector_stride)
    mesh_ys, mesh_xs = np.meshgrid(ys, xs, indexing="ij")
    flat_ys, flat_xs = mesh_ys.flatten(), mesh_xs.flatten()
    try:
        lats_sub = Lat[flat_ys, flat_xs]
        lons_sub = Lon[flat_ys, flat_xs]
        u_sub = u_grid.values[flat_ys, flat_xs] if hasattr(u_grid, "values") else u_grid[flat_ys, flat_xs]
        v_sub = v_grid.values[flat_ys, flat_xs] if hasattr(v_grid, "values") else v_grid[flat_ys, flat_xs]
        vec_scale = 0.2
        for i in range(len(lats_sub)):
            lat_s, lon_s = float(lats_sub[i]), float(lons_sub[i])
            u_v, v_v = float(u_sub[i]), float(v_sub[i])
            if np.isnan(u_v) or np.isnan(v_v) or (u_v == 0 and v_v == 0):
                continue
            folium.PolyLine(
                [(lat_s, lon_s), (lat_s + v_v * vec_scale, lon_s + u_v * vec_scale)],
                color="gray", weight=1, opacity=0.6,
            ).add_to(m)
    except Exception:
        pass

# Contours
if show_contours:
    for contour in find_contours(mask, 0.5):
        mc = []
        for r, c in contour:
            r_i, c_i = int(r), int(c)
            if 0 <= r_i < Lat.shape[0] and 0 <= c_i < Lat.shape[1]:
                mc.append((Lat[r_i, c_i], Lon[r_i, c_i]))
        if len(mc) > 2:
            folium.Polygon(mc, color="lime", weight=1, fill=True,
                           fill_color="lime", fill_opacity=0.3).add_to(m)

# Vortex centres
for vortex in centers:
    lat, lon  = vortex["lat"], vortex["lon"]
    v_id      = vortex.get("id")
    v_id_str  = str(v_id) if v_id is not None else "New"
    radius_km = vortex.get("radius_km", 50)
    sb        = vortex.get("score_breakdown", {})

    if show_trails and v_id is not None:
        history = get_track_history(selected_idx, v_id)
        if len(history) > 1:
            folium.PolyLine(history, color="cyan", weight=2.5, opacity=0.7, dash_array="5,5").add_to(m)

    if show_radii:
        folium.Circle([lat, lon], radius=radius_km * 1000,
                      color="red", weight=1, fill=False, opacity=0.5).add_to(m)

    if show_centers:
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 230px;">
            <h5 style="margin:0; color:#333;">Vortex {v_id_str}</h5>
            <hr style="margin: 5px 0;">
            <table style="width:100%; border-collapse:collapse; font-size:0.85em;">
                <tr><td><b>Total Score:</b></td><td style="text-align:right;">{vortex.get('total_score', 0):.2f}</td></tr>
                <tr style="background:#f7f7f7"><td><b>Radius:</b></td><td style="text-align:right;">{radius_km:.1f} km</td></tr>
                <tr><td><b>Area:</b></td><td style="text-align:right;">{vortex.get('area', 0)} px</td></tr>
                <tr style="background:#f7f7f7"><td><b>Consistency:</b></td><td style="text-align:right;">{vortex.get('weighted_consistency_score', 0):.3f}</td></tr>
                <tr><td><b>Aspect Ratio:</b></td><td style="text-align:right;">{vortex.get('aspect_ratio', 0):.2f}</td></tr>
                <tr style="background:#f7f7f7"><td><b>Peak Vort.:</b></td><td style="text-align:right;">{vortex.get('peak_vorticity', 0):.2e}</td></tr>
                <tr><td colspan="2" style="padding-top:6px; color:#555;"><i>Click marker, then press<br><b>"Get LLM Insights"</b> below the map.</i></td></tr>
            </table>
        </div>
        """
        folium.CircleMarker(
            [lat, lon], radius=6,
            color="red", fill=True, fill_color="white", fill_opacity=1.0,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"Vortex {v_id_str}  |  score {vortex.get('total_score', 0):.1f}",
        ).add_to(m)

# Targets
if show_targets and targets_data and current_key in targets_data:
    for t in targets_data[current_key]:
        t_lat, t_lon, t_name = t["lat"], t["lon"], t["name"]
        folium.Marker(
            [t_lat, t_lon], tooltip=f"Target: {t_name}",
            icon=folium.Icon(color="orange", icon="info-sign"),
        ).add_to(m)
        path = get_target_history(targets_data, selected_idx, st.session_state.sorted_keys, t_name)
        if len(path) > 1:
            folium.PolyLine(path, color="orange", weight=3, opacity=0.8,
                            dash_array="2,5", tooltip=f"Path: {t_name}").add_to(m)

# ── Render map ─────────────────────────────────────────────────────────────────
ret_objects = ["last_object_clicked"]
if play:
    ret_objects += ["zoom", "center"]

st_map_data = st_folium(m, width="100%", height=500, returned_objects=ret_objects)

# Persist zoom/pan state
if st_map_data:
    if st_map_data.get("zoom") is not None:
        st.session_state.map_zoom = st_map_data["zoom"]
    center_data = st_map_data.get("center")
    if center_data:
        st.session_state.map_center = [center_data["lat"], center_data["lng"]]

# ── Detect map click → select vortex ──────────────────────────────────────────
if st_map_data:
    clicked = st_map_data.get("last_object_clicked")
    if clicked:
        click_key = f"{clicked.get('lat', '')},{clicked.get('lng', '')}"
        if click_key != st.session_state.last_click_key and centers:
            st.session_state.last_click_key = click_key
            clat, clng = clicked["lat"], clicked["lng"]
            # Find nearest vortex centre to the click
            nearest = min(
                centers,
                key=lambda c: (c["lat"] - clat) ** 2 + (c["lon"] - clng) ** 2,
            )
            dist_sq = (nearest["lat"] - clat) ** 2 + (nearest["lon"] - clng) ** 2
            if dist_sq < 4.0:   # within ~2° ≈ reasonable click tolerance
                if st.session_state.selected_vortex != nearest:
                    st.session_state.selected_vortex = nearest
                    st.session_state.llm_response    = None  # force fresh insight


# BELOW-MAP PANELS  —  score chart  |  LLM insights
col_chart, col_llm = st.columns([3, 2], gap="large")

sel = st.session_state.selected_vortex

# ── Score breakdown chart (selected vortex only) ───────────────────────────────
with col_chart:
    st.markdown("### Score Breakdown")

    if sel is None:
        st.info("Click a vortex marker on the map to see its score breakdown here.")
    else:
        def _sb_val(v, key):
            val = v.get("score_breakdown", {}).get(key, 0)
            return val if (val is not None and np.isfinite(val)) else 0.0

        v_id_display = str(sel.get("id")) if sel.get("id") is not None else "New"

        component_names  = ["Vorticity", "Shape", "Consistency", "Persistence"]
        component_keys   = ["vorticity", "shape", "consistency", "persistence"]
        component_colors = ["#EF553B",   "#636EFA", "#00CC96",   "#FFA15A"]
        values = [_sb_val(sel, k) for k in component_keys]
        total  = sum(values)

        fig = go.Figure()

        for name, val, color in zip(component_names, values, component_colors):
            pct_label = f"{100 * val / total:.0f}%" if total > 1e-9 else "0%"
            fig.add_trace(go.Bar(
                name=name,
                x=[val],
                y=[f"Vortex {v_id_display}"],
                orientation="h",
                marker_color=color,
                text=f"{name}<br>{val:.1f} ({pct_label})",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"<b>{name}</b><br>Score: {val:.2f}<br>Share: {pct_label}<extra></extra>",
            ))

        # Threshold marker line (as a shape on a horizontal bar is awkward —
        # show it as an annotation on the x-axis instead)
        fig.add_vline(
            x=score_thresh,
            line_dash="dash",
            line_color="white",
            line_width=1.5,
            annotation_text=f"Threshold ({score_thresh:.1f})",
            annotation_position="top right",
            annotation_font_color="white",
            annotation_font_size=11,
        )

        # Total score label to the right of the bar
        fig.add_annotation(
            x=total, y=0,
            text=f"  Total: {total:.1f}",
            showarrow=False,
            xanchor="left",
            font=dict(color="white", size=13),
        )

        fig.update_layout(
            barmode="stack",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
            margin=dict(l=10, r=120, t=50, b=10),
            xaxis_title="Score Contribution",
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333", showticklabels=False),
            height=180,
        )

        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Coloured segments show each test's weighted contribution to the total score. "
            "The dashed line marks the detection threshold."
        )

# ── LLM Insights panel ────────────────────────────────────────────────────────
with col_llm:
    st.markdown("### LLM Insights")

    if sel is None:
        st.info(
            "Click any vortex marker on the map to select it, "
            "then press **Get LLM Insights** to have the model interpret the detection."
        )
    else:
        v_id_str = str(sel.get("id")) if sel.get("id") is not None else "New"
        sb = sel.get("score_breakdown", {})

        # Mini summary card
        st.markdown(
            f"""
            <div style="background:#1a1a2e; border-left:3px solid #EF553B;
                        padding:10px 14px; border-radius:6px; margin-bottom:10px;">
                <span style="color:#EF553B; font-weight:700;">Vortex {v_id_str}</span>
                &nbsp;|&nbsp;
                <span style="color:#aaa;">{sel['lat']:.2f}°N, {sel['lon']:.2f}°E</span><br>
                <small style="color:#ccc;">
                    Score <b style="color:white">{sel['total_score']:.1f}</b>
                    &nbsp;·&nbsp; Radius <b style="color:white">{sel['radius_km']:.0f} km</b>
                    &nbsp;·&nbsp; Consistency <b style="color:white">{sel['weighted_consistency_score']:.3f}</b>
                    &nbsp;·&nbsp; {'Tracked' if sb.get('persistence', 0) > 0 else 'New'}
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Ollama model helper — list available models so user can pick the exact name
        list_col, _ = st.columns([1, 2])
        with list_col:
            if st.button("List available models"):
                try:
                    client = ollama_client.Client(host=ollama_host)
                    response = client.list()
                    # Newer versions return a ListResponse object; older return a dict
                    if hasattr(response, "models"):
                        raw_models = response.models
                        model_names = [
                            m.model if hasattr(m, "model") else str(m)
                            for m in raw_models
                        ]
                    else:
                        raw_models = response.get("models", [])
                        model_names = [
                            m.get("model", m.get("name", str(m)))
                            for m in raw_models
                        ]
                    if model_names:
                        st.info("Available models:\n" + "\n".join(f"• `{n}`" for n in model_names))
                    else:
                        st.warning(
                            "Ollama is reachable but returned no models — "
                            f"raw response: `{response}`"
                        )
                except Exception as exc:
                    st.error(f"Could not reach Ollama: {exc}")

        if st.button("Get LLM Insights", type="primary"):
            with st.spinner(f"Asking {ollama_model}…"):
                prompt = build_llm_prompt(sel, detection_params)
                st.session_state.llm_response = call_ollama(prompt, ollama_host, ollama_model)

        if st.session_state.llm_response:
            resp = st.session_state.llm_response
            is_error = resp.startswith("⚠️")
            bg  = "#2a1a1a" if is_error else "#0d1f17"
            bdr = "#EF553B" if is_error else "#00CC96"
            st.markdown(
                f"""
                <div style="background:{bg}; border-left:3px solid {bdr};
                            padding:12px 14px; border-radius:6px;
                            font-size:0.9em; line-height:1.55; color:#ddd;">
                    {resp.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Expandable prompt preview (for debugging / transparency)
        with st.expander("View prompt sent to model"):
            if sel:
                st.code(build_llm_prompt(sel, detection_params), language="text")


# DATA TABLE
if centers:
    with st.expander(f"Detailed Data for {current_key}", expanded=False):
        df = pd.DataFrame(centers)
        display_cols = [
            "id", "lat", "lon", "total_score",
            "radius_km", "weighted_consistency_score",
            "aspect_ratio", "peak_vorticity",
        ]
        final_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[final_cols].style.highlight_max(axis=0), width='stretch')


# AUTO-PLAY LOOP
if play:
    time.sleep(speed)
    st.session_state.play_idx = (st.session_state.play_idx + 1) % len(st.session_state.sorted_keys)
    st.rerun()