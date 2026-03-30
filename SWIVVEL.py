import os
import glob
import time
import numpy as np
import xarray as xr
from typing import Tuple
from scipy.ndimage import binary_closing
from skimage.morphology import remove_small_objects, remove_small_holes, binary_erosion
from haversine import haversine
from contextlib import redirect_stdout


def load_and_prepare_data(file_path):
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
        return None

    u = ds['x_tau'].isel(time=0, zlev=0)
    v = ds['y_tau'].isel(time=0, zlev=0)
    lat = ds['lat']
    lon = ds['lon']

    # Standardize longitude
    if np.any(lon > 180):
        lon_vals = ((lon.values + 180) % 360) - 180
        sort_idx = np.argsort(lon_vals)
        lon_sorted = lon_vals[sort_idx]
        lon = xr.DataArray(lon_sorted, dims=lon.dims, name='lon')
        u = u.isel(lon=sort_idx)
        v = v.isel(lon=sort_idx)
        u = u.assign_coords(lon=lon)
        v = v.assign_coords(lon=lon)

    # Handle missing data
    np.random.seed(42)
    drop_mask = np.random.rand(*u.shape) < 0.0
    u = u.where(~drop_mask)
    v = v.where(~drop_mask)

    # Create 2D Lon/Lat grids
    Lon, Lat = np.meshgrid(lon, lat)

    return {'u': u, 'v': v, 'Lon': Lon, 'Lat': Lat}


def compute_gradients(u, v, lon2d, lat2d):
    R = 6371000.0
    lon_r = np.deg2rad(lon2d)
    lat_r = np.deg2rad(lat2d)
    duf_dlon, duf_dlat = np.gradient(u, lon_r[0, :], lat_r[:, 0], axis=(1, 0))
    dvf_dlon, dvf_dlat = np.gradient(v, lon_r[0, :], lat_r[:, 0], axis=(1, 0))
    cos_lat = np.cos(lat_r)
    du_dx = duf_dlon / (R * cos_lat)
    du_dy = duf_dlat / R
    dv_dx = dvf_dlon / (R * cos_lat)
    dv_dy = dvf_dlat / R
    return du_dx, du_dy, dv_dx, dv_dy


def okubo_weiss(u_region, v_region, lon2d_region, lat2d_region):
    du_dx, du_dy, dv_dx, dv_dy = compute_gradients(u_region, v_region, lon2d_region, lat2d_region)
    s_n = du_dx - dv_dy
    s_s = du_dy + dv_dx
    omega = dv_dx - du_dy
    OW = s_n**2 + s_s**2 - omega**2
    return OW, omega


def connected_components_mask(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                stack = [(i, j)]
                comp = []
                seen[i, j] = True
                while stack:
                    ci, cj = stack.pop()
                    comp.append((ci, cj))
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < H and 0 <= nj < W and mask[ni, nj] and not seen[ni, nj]:
                            seen[ni, nj] = True
                            stack.append((ni, nj))
                components.append(comp)
    return components


def process_and_compute_ow_mask(
    u: xr.DataArray,
    v: xr.DataArray,
    scale_factor: int,
    ow_threshold: float,
) -> Tuple[xr.DataArray, xr.DataArray]:
    # Scale Down Data
    u_coarse = u.coarsen(lon=scale_factor, lat=scale_factor, boundary="pad").mean()
    v_coarse = v.coarsen(lon=scale_factor, lat=scale_factor, boundary="pad").mean()

    lon_coarse = u_coarse.lon
    lat_coarse = u_coarse.lat

    Lon_coarse, Lat_coarse = np.meshgrid(lon_coarse, lat_coarse)

    u_coarse_np = u_coarse.values
    v_coarse_np = v_coarse.values

    OW_coarse_np, omega_coarse_np = okubo_weiss(u_coarse_np, v_coarse_np, Lon_coarse, Lat_coarse)

    # Mask
    initial_mask = OW_coarse_np < ow_threshold

    land_mask = np.isnan(u_coarse_np)
    initial_mask[land_mask] = False

    final_coarse_mask_da = xr.DataArray(
        initial_mask,
        coords={"lat": lat_coarse, "lon": lon_coarse},
        dims=["lat", "lon"]
    )

    omega_coarse_da = xr.DataArray(
        omega_coarse_np,
        coords={"lat": lat_coarse, "lon": lon_coarse},
        dims=["lat", "lon"]
    )

    # Scale Up
    ow_mask_upscaled = final_coarse_mask_da.reindex_like(u, method="nearest")
    omega_upscaled = omega_coarse_da.reindex_like(u, method="nearest")

    return ow_mask_upscaled, omega_upscaled


def analyze_weighted_consistency(center_lon, center_lat, u_fine, v_fine, Lon, Lat, radius_km=150):
    R_EARTH = 6371.0
    radius_deg = radius_km / R_EARTH * (180 / np.pi)

    dist_sq = (Lon - center_lon)**2 + (Lat - center_lat)**2
    window_mask = dist_sq < radius_deg**2

    if np.sum(window_mask) < 10:
        return 0.0

    local_lons = Lon[window_mask]
    local_lats = Lat[window_mask]
    local_u = u_fine[window_mask]
    local_v = v_fine[window_mask]

    dlon = np.deg2rad(local_lons - center_lon)
    lat_rad = np.deg2rad(local_lats)
    center_lat_rad = np.deg2rad(center_lat)

    y = np.sin(dlon) * np.cos(lat_rad)
    x = np.cos(center_lat_rad) * np.sin(lat_rad) - \
        np.sin(center_lat_rad) * np.cos(lat_rad) * np.cos(dlon)
    theta = np.arctan2(y, x)

    v_tangential = local_u * np.cos(theta) - local_v * np.sin(theta)

    if v_tangential.size == 0:
        return 0.0

    magnitudes = np.abs(v_tangential)
    total_magnitude = np.nansum(magnitudes)
    net_momentum = np.nansum(v_tangential)

    if total_magnitude < 1e-6:
        return 0.0

    weighted_score = np.abs(net_momentum / total_magnitude)

    if np.isnan(weighted_score):
        return 0.0

    return weighted_score


def estimate_vortex_radius_by_consistency(center_lon, center_lat, u_fine, v_fine, Lon, Lat,
                                          max_radius_km=1000, step_km=100, cutoff_threshold=0.5):
    best_radius = step_km * 2

    for r in range(step_km * 2, max_radius_km + step_km, step_km):
        score = analyze_weighted_consistency(
            center_lon, center_lat, u_fine, v_fine, Lon, Lat, radius_km=r
        )

        if score < cutoff_threshold:
            return max(step_km, r - step_km)

        best_radius = r

    return best_radius


def detect_vortices_multi(u_region, v_region,
                          # --- Segmentation Parameters ---
                          ow_threshold=-2e-12,
                          min_area=50,
                          closing_size=3,
                          erosion_size=3,
                          scale_factor=1,

                          # --- Analysis Parameters ---
                          analysis_radius_km=150,
                          radius_max_km=800,
                          radius_step_km=25,
                          radius_cutoff_threshold=0.5,
                          previous_centers=None,
                          max_tracking_distance_km=500,

                          # --- SCORING WEIGHTS ---
                          weight_vorticity=1.0,
                          weight_aspect_ratio=1.0,
                          weight_consistency=4.0,
                          weight_persistence=2.0,

                          # --- TOTAL THRESHOLD ---
                          total_score_threshold=3.5):

    mask, omega = process_and_compute_ow_mask(
        u_region, v_region,
        scale_factor=scale_factor,
        ow_threshold=ow_threshold
    )

    try:
        mask_np = np.asarray(mask.values, dtype=bool)
    except Exception:
        mask_np = np.asarray(mask, dtype=bool)

    try:
        omega_np = np.asarray(omega.values)
    except Exception:
        omega_np = np.asarray(omega)

    try:
        u_np = np.asarray(u_region.values)
    except Exception:
        u_np = np.asarray(u_region)

    try:
        v_np = np.asarray(v_region.values)
    except Exception:
        v_np = np.asarray(v_region)

    try:
        lon_vals = u_region.lon.values
        lat_vals = u_region.lat.values
        Lon2d, Lat2d = np.meshgrid(lon_vals, lat_vals)
        lon2d_region = np.asarray(Lon2d)
        lat2d_region = np.asarray(Lat2d)
    except Exception:
        raise RuntimeError("Unable to construct lon/lat grids from inputs.")

    # Morphological operations
    mask_np = remove_small_objects(mask_np, min_size=int(min_area))

    if closing_size > 1:
        structure = np.ones((int(closing_size), int(closing_size)))
        if mask_np.shape[0] >= structure.shape[0] and mask_np.shape[1] >= structure.shape[1]:
            mask_np = binary_closing(mask_np, structure=structure)

    mask_np = remove_small_holes(mask_np, area_threshold=1000)
    if erosion_size > 0:
        mask_np = binary_erosion(mask_np, footprint=np.ones((int(erosion_size), int(erosion_size))))

    comps = connected_components_mask(mask_np)

    centers = []
    OMEGA = 7.2921e-5

    for comp in comps:
        ys = np.array([p[0] for p in comp])
        xs = np.array([p[1] for p in comp])
        omegas = omega_np[ys, xs]
        abs_omegas = np.abs(omegas)
        lats = lat2d_region[ys, xs]

        # Centroid Calculation
        mean_y = np.mean(ys)
        mean_x = np.mean(xs)
        distances_sq = (ys - mean_y)**2 + (xs - mean_x)**2
        center_idx_in_comp = np.argmin(distances_sq)
        cy, cx = int(ys[center_idx_in_comp]), int(xs[center_idx_in_comp])

        candidate_lon = float(lon2d_region[cy, cx])
        candidate_lat = float(lat2d_region[cy, cx])

        comp_scores = {}
        total_score = 0.0

        # --- Vorticity Score ---
        mean_lat_rad = np.deg2rad(np.nanmean(lats))
        f = 2 * OMEGA * np.sin(mean_lat_rad)
        f_mag = max(np.abs(f), 1e-6)

        peak_abs_omega = np.nanmax(abs_omegas) if np.any(np.isfinite(abs_omegas)) else 0.0
        metric_vorticity = float(peak_abs_omega) / f_mag if np.isfinite(peak_abs_omega) else 0.0

        comp_scores['vorticity'] = metric_vorticity * weight_vorticity
        total_score += comp_scores['vorticity']

        # --- Shape Score (Aspect Ratio) ---
        aspect_ratio = 1.0
        if len(comp) > 2:
            points = np.column_stack((xs, ys))
            cov = np.cov(points, rowvar=False)
            det = np.linalg.det(cov)
            if np.isfinite(det) and det != 0:
                eigenvalues, _ = np.linalg.eig(cov)
                eigenvalues = np.real(eigenvalues)
                ev_min = np.min(eigenvalues)
                ev_max = np.max(eigenvalues)
                if ev_min > 0 and np.isfinite(ev_max / ev_min):
                    aspect_ratio = np.sqrt(ev_max / ev_min)

        metric_ar = 1.0 / aspect_ratio
        comp_scores['shape'] = metric_ar * weight_aspect_ratio
        total_score += comp_scores['shape']

        # --- Consistency Score ---
        flow_score = analyze_weighted_consistency(
            candidate_lon, candidate_lat, u_np, v_np, lon2d_region, lat2d_region,
            radius_km=analysis_radius_km
        )
        comp_scores['consistency'] = flow_score * weight_consistency
        total_score += comp_scores['consistency']

        # --- Persistence Score ---
        is_persistent = False
        matched_id = None

        if previous_centers:
            for prev_vortex in previous_centers:
                dist = haversine(
                    (candidate_lat, candidate_lon),
                    (prev_vortex['lat'], prev_vortex['lon'])
                )
                if dist < max_tracking_distance_km:
                    is_persistent = True
                    matched_id = prev_vortex.get('id')
                    break

        metric_persistence = 1.0 if is_persistent else 0.0
        comp_scores['persistence'] = metric_persistence * weight_persistence
        total_score += comp_scores['persistence']

        # Sanitise — replace any lingering NaN/inf with 0 so they never
        # propagate into results or the threshold comparison.
        def _safe(x) -> float:
            v = float(x)
            return v if np.isfinite(v) else 0.0

        for k in list(comp_scores):
            comp_scores[k] = _safe(comp_scores[k])
        total_score = sum(comp_scores.values())

        # --- TOTAL THRESHOLD ---
        if total_score < total_score_threshold:
            continue

        # Radius Estimation
        radius_km = estimate_vortex_radius_by_consistency(
            candidate_lon, candidate_lat, u_np, v_np, lon2d_region, lat2d_region,
            max_radius_km=radius_max_km,
            step_km=radius_step_km,
            cutoff_threshold=radius_cutoff_threshold
        )

        centers.append({
            'lat':                        candidate_lat,
            'lon':                        candidate_lon,
            'area':                       len(comp),
            'radius_km':                  _safe(radius_km),
            'peak_vorticity':             _safe(omega_np[cy, cx]),
            'mean_vorticity':             _safe(np.nanmean(omegas)),
            'aspect_ratio':               _safe(aspect_ratio),
            'weighted_consistency_score': _safe(flow_score),
            'total_score':                total_score,
            'id':                         matched_id,
            'score_breakdown': {
                'vorticity':   comp_scores['vorticity'],
                'shape':       comp_scores['shape'],
                'consistency': comp_scores['consistency'],
                'persistence': comp_scores['persistence'],
            },
            'raw_metrics': {
                'metric_vorticity':   _safe(metric_vorticity),
                'metric_aspect':      _safe(metric_ar),
                'metric_consistency': _safe(flow_score),
                'metric_persistence': _safe(metric_persistence),
                'coriolis_param':     _safe(f),
            },
        })

    region_coords = {
        'mask': mask_np,
        'lat': lat2d_region,
        'lon': lon2d_region
    }

    return centers, region_coords


def run_analysis_on_folder(folder_path, detection_params):
    search_path = os.path.join(folder_path, '*.nc')
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"Warning: No .nc files found in {folder_path}")
        return {}

    print(f"Found {len(file_list)} files to process...")
    results_dict = {}

    previous_centers = []
    next_unique_id = 0

    for file_path in file_list:
        start_time = time.time()
        print(f"Processing {os.path.basename(file_path)}...")

        data = load_and_prepare_data(file_path)
        if data is None:
            continue

        run_params = detection_params.copy()
        run_params['previous_centers'] = previous_centers

        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull):
                centers, region = detect_vortices_multi(
                    data['u'], data['v'],
                    **run_params
                )

        for center in centers:
            if center.get('id') is None:
                center['id'] = next_unique_id
                next_unique_id += 1

        file_key = os.path.basename(file_path)
        results_dict[file_key] = {
            'centers': centers,
            'region': region,
            'data': data
        }

        previous_centers = centers
        print(f"...done in {time.time() - start_time:.2f}s. Found {len(centers)} centers.")

    return results_dict