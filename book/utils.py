"""
Utility functions for corridor network risk analysis.

This module consolidates helper functions used across the corridor_analysis notebook,
including infrastructure ID normalization, CRS repair, geometry lookups, and cost calculations.
"""

import re
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from time import perf_counter
from shapely import wkb, wkt
from shapely.geometry import box
from scipy.interpolate import interp1d


# ── Infrastructure ID Normalization ───────────────────────────────────────────

def _norm_key(value):
    """Normalise an infrastructure ID to a clean lowercase string for joining."""
    if pd.isna(value):
        return ''
    text = str(value).strip().lower()
    if not text:
        return ''
    if text.endswith('.0') and text[:-2].replace('-', '', 1).isdigit():
        return text[:-2]
    if text.replace('-', '', 1).isdigit():
        return text
    try:
        f = float(text)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return text


def _canonical_transport_id(value):
    """Strip OSM/routing prefixes and separators from an infrastructure ID."""
    text = _norm_key(value)
    if not text:
        return ''
    text = text.replace('europe-latest', '')
    text = text.replace('europe_latest', '')
    text = text.replace('europe/latest', '')
    text = text.strip('_-/ ')
    text = re.sub(r'[_\-/\s]+', '', text)
    return text


def _map_unique(series, func):
    """Apply a scalar function to a Series, computing it once per unique
    value instead of once per row.

    Equivalent to ``series.map(func)``, but for columns that reference a
    small set of distinct IDs from many repeated rows (e.g. an infra edge/
    port ID referenced by millions of disruption-scenario rows), this avoids
    recomputing ``func`` for duplicates. Measured ~30x faster on a column
    with ~1,150x duplication; no benefit (but no regression either) when
    values are mostly unique.
    """
    uniques = series.unique()
    mapping = {v: func(v) for v in uniques}
    return series.map(mapping)


# ── Column and File Utilities ─────────────────────────────────────────────────

def _pick_first_existing(df, candidates):
    """Return the first column in *candidates* that exists in *df*, or None."""
    if df is None or len(df) == 0:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _pick_existing(candidates):
    """Return the first candidate Path that exists, or the first candidate."""
    return next((p for p in candidates if p.exists()), candidates[0])


def _timed_load(path: Path, label: str):
    """Read a parquet-like object and print elapsed time with a label."""
    t0 = perf_counter()
    obj = _safe_read_parquet_any(path)
    obj = _as_geodataframe(obj)
    print(f'[{label}] loaded in {perf_counter()-t0:.1f}s: {path}')
    return obj


def _coerce_loaded_geo(obj, label: str):
    if isinstance(obj, gpd.GeoDataFrame):
        return obj
    if not isinstance(obj, pd.DataFrame) or obj.empty:
        return obj

    preferred = [
        'geometry', 'geom', 'the_geom', 'wkb_geometry', 'wkt_geometry',
        'shape', 'geometrie', 'GEOMETRY', 'GEOM',
    ]
    dynamic = [c for c in obj.columns if ('geom' in str(c).lower() or str(c).lower().endswith('wkt') or str(c).lower().endswith('wkb'))]
    candidates = [c for c in preferred if c in obj.columns] + [c for c in dynamic if c not in preferred]

    def _parse_one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if hasattr(v, 'geom_type'):
            return v
        if isinstance(v, memoryview):
            v = v.tobytes()
        if isinstance(v, (bytes, bytearray)):
            try:
                return wkb.loads(bytes(v))
            except Exception:
                return None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return wkt.loads(s)
            except Exception:
                try:
                    return wkb.loads(bytes.fromhex(s))
                except Exception:
                    return None
        return None

    for col in candidates:
        parsed = obj[col].map(_parse_one)
        if parsed.notna().any():
            gdf = gpd.GeoDataFrame(obj.copy(), geometry=parsed, crs='EPSG:4326')
            print(f'[fix] coerced {label} -> GeoDataFrame using column `{col}`')
            return gdf

    return obj

def assign_primary_corridor(frame: gpd.GeoDataFrame, corridor_label='Rhine-Alpine') -> gpd.GeoDataFrame:
    """Ensure a GeoDataFrame has a `corridor` column populated with a primary label."""
    if frame is None or len(frame) == 0:
        return frame
    if 'corridor' not in frame.columns:
        frame = frame.copy()
        frame['corridor'] = corridor_label
    return frame


def _bounds_with_padding(area_gdf: gpd.GeoDataFrame, pad_ratio: float = 0.04):
    """Return x/y bounds with fractional padding for plotting."""
    minx, miny, maxx, maxy = area_gdf.total_bounds
    dx = max((maxx - minx) * pad_ratio, 1e-6)
    dy = max((maxy - miny) * pad_ratio, 1e-6)
    return (minx - dx, maxx + dx), (miny - dy, maxy + dy)


# Aliases for backward compatibility
_first_existing = _pick_first_existing
_pick_col = _pick_first_existing


# ── Mode and Type Mapping ─────────────────────────────────────────────────────

def _mode_canon(mode_text):
    """Map a raw mode string to its canonical infrastructure-type label."""
    m = str(mode_text).strip().lower()
    mapper = {
        'road': 'road edge', 'roads': 'road edge', 'road edge': 'road edge',
        'rail': 'rail edge', 'rails': 'rail edge', 'rail edge': 'rail edge',
        'iww': 'iww port', 'iww_port': 'iww port', 'iww_ports': 'iww port',
        'port': 'port', 'ports': 'port',
        'airport': 'airport', 'airports': 'airport',
    }
    return mapper.get(m, None)


# ── Geometry Lookups ──────────────────────────────────────────────────────────

def _build_geo_keys(gdf, infra_type_label, candidate_cols, include_keys=None, extra_filter=None):
    """Build a normalised (infra_type, match_key, geometry) lookup from a GeoDataFrame."""
    empty = gpd.GeoDataFrame(columns=['infra_type', 'match_key', 'geometry'], geometry='geometry', crs=None)
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return empty

    # Backward compatibility: older calls passed `extra_filter` as the 4th positional argument.
    if callable(include_keys) and extra_filter is None:
        extra_filter = include_keys
        include_keys = None

    work = gdf.copy()
    if extra_filter is not None:
        mask = extra_filter(work)
        if mask is not None:
            work = work.loc[mask].copy()
    if work.empty:
        return empty.copy().__finalize__(gdf)
    frames = []
    for col in candidate_cols:
        if col in work.columns:
            tmp = work[[col, 'geometry']].copy()
            # Use type-aware normalization for road/rail to handle canonical IDs
            tmp['match_key'] = tmp[col].map(lambda v, _t=infra_type_label: _norm_infra_key_by_type(v, _t))
            tmp['infra_type'] = infra_type_label
            tmp = tmp[tmp['match_key'].ne('') & tmp.geometry.notna()].copy()
            if not tmp.empty:
                frames.append(tmp[['infra_type', 'match_key', 'geometry']])
    if not frames:
        return gpd.GeoDataFrame(columns=['infra_type', 'match_key', 'geometry'], geometry='geometry', crs=work.crs)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['infra_type', 'match_key'])

    if include_keys is not None:
        # Also use type-aware normalization for include_keys
        keep = {_norm_infra_key_by_type(x, infra_type_label) for x in include_keys}
        keep = {k for k in keep if k}
        if keep:
            out = out[out['match_key'].astype(str).isin(keep)].copy()

    return gpd.GeoDataFrame(out, geometry='geometry', crs=work.crs)


def _build_geo_lookup(gdf, cols):
    """Build a simple (_match_key, geometry) lookup from a GeoDataFrame."""
    empty = gpd.GeoDataFrame(columns=['_match_key', 'geometry'], geometry='geometry', crs=None)
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return empty
    frames = []
    for c in cols:
        if c in gdf.columns:
            t = gdf[[c, 'geometry']].copy()
            t['_match_key'] = t[c].map(_norm_key)
            t = t[t['_match_key'].ne('') & t.geometry.notna()].copy()
            if not t.empty:
                frames.append(t[['_match_key', 'geometry']])
    if not frames:
        return gpd.GeoDataFrame(columns=['_match_key', 'geometry'], geometry='geometry', crs=gdf.crs)
    out = pd.concat(frames, ignore_index=True).drop_duplicates('_match_key')
    return gpd.GeoDataFrame(out, geometry='geometry', crs=gdf.crs)


def _hybas_id_col(gdf, label):
    """Return the basin-ID column name for a HydroBASINS GeoDataFrame."""
    for cand in ['HYBAS_ID', 'hybas_id', 'basin_id', 'BASIN_ID', 'id']:
        if cand in gdf.columns:
            return cand
    raise ValueError(f'No HYBAS_ID column in {label}. Cols: {gdf.columns.tolist()}')


# ── CRS Repair ────────────────────────────────────────────────────────────────

def _infer_and_fix_crs(gdf):
    """Correct CRS if coordinates are metric but GDF is mislabeled as EPSG:4326.
    
    Uses a pixel-sampling approach: detects if most representative points
    fall within geographic bounds (-180/180 longitude, -90/90 latitude).
    For projected-like coordinates with missing/wrong CRS, assigns EPSG:3035
    by default, but accepts EPSG:3857 when values look Web-Mercator-like.
    """
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return gdf
    out = gdf.copy()
    valid = out[out.geometry.notna() & ~out.geometry.is_empty]
    if valid.empty:
        return out
    rp = valid.geometry.representative_point()
    x = pd.Series(rp.x).replace([np.inf, -np.inf], np.nan).dropna()
    y = pd.Series(rp.y).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty or y.empty:
        return out
    frac_deg = ((x.abs() <= 180) & (y.abs() <= 90)).mean()
    current = str(out.crs) if out.crs is not None else ''
    if frac_deg >= 0.95:
        if out.crs is None:
            out = out.set_crs('EPSG:4326', allow_override=True)
        return out
    if (not current) or ('4326' in current):
        p95_abs_x = float(np.nanpercentile(x.abs(), 95)) if len(x) else 0.0
        p95_abs_y = float(np.nanpercentile(y.abs(), 95)) if len(y) else 0.0
        fallback_crs = 'EPSG:3857' if max(p95_abs_x, p95_abs_y) > 8_000_000 else 'EPSG:3035'
        out = out.set_crs(fallback_crs, allow_override=True)
    return out


def _as_geodataframe(frame, default_crs=None):
    """Coerce a DataFrame/GeoDataFrame with geometry values into a GeoDataFrame."""
    if frame is None or getattr(frame, 'empty', True):
        return frame
    if isinstance(frame, gpd.GeoDataFrame):
        return frame
    if not hasattr(frame, 'columns') or 'geometry' not in frame.columns:
        return frame

    frame_copy = frame.copy()
    frame_copy['geometry'] = frame_copy['geometry'].map(_parse_geom)
    return gpd.GeoDataFrame(frame_copy, geometry='geometry', crs=default_crs)


def fix_mislabeled_crs(gdf, name, source_crs="EPSG:3035", target_crs="EPSG:4326"):
    """Fix mislabeled CRS using bounds detection.
    
    Checks if bounding box looks projected (coordinates beyond ±180/±90).
    If so, applies source_crs and reprojects to target_crs.
    """
    gdf = _as_geodataframe(gdf, default_crs=target_crs)
    if gdf is None or getattr(gdf, 'empty', True):
        print(f"{name}: empty/missing, skipped")
        return gdf

    if not isinstance(gdf, gpd.GeoDataFrame):
        print(f"{name}: not a GeoDataFrame, skipped")
        return gdf

    b = gdf.total_bounds
    looks_projected = abs(b[0]) > 180 or abs(b[2]) > 180 or abs(b[1]) > 90 or abs(b[3]) > 90

    if looks_projected:
        # Coordinates are projected; force correct source CRS first
        gdf_fixed = gdf.copy()
        gdf_fixed = gdf_fixed.set_crs(source_crs, allow_override=True).to_crs(target_crs)
        b2 = gdf_fixed.total_bounds
        print(f"{name}: FIXED ({source_crs} -> {target_crs})")
        print(f"  before: [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}]")
        print(f"  after : [{b2[0]:.2f}, {b2[1]:.2f}, {b2[2]:.2f}, {b2[3]:.2f}]")
        return gdf_fixed

    # Already geographic-like bounds
    if gdf.crs != target_crs:
        gdf_fixed = gdf.to_crs(target_crs)
        print(f"{name}: reprojected {gdf.crs} -> {target_crs}")
        return gdf_fixed

    print(f"{name}: no fix needed")
    return gdf


def _country_join(asset_gdf, nuts2_gdf):
    """Spatial-join assets to NUTS2 and return country code (first 2 chars of NUTS ID)."""
    target_crs = nuts2_gdf.crs
    if asset_gdf.crs != target_crs:
        asset_gdf = asset_gdf.to_crs(target_crs)
    nuts2_code_col = _pick_first_existing(
        nuts2_gdf,
        ['NUTS_ID', 'NUTS_ID_2021', 'NUTS_ID_2016', 'nuts_id', 'id', 'NUTS2', 'NUTS_CODE'],
    )
    if nuts2_code_col is None:
        raise ValueError('Could not find NUTS2 ID/code column in `nuts2`.')
    join_right = nuts2_gdf[[nuts2_code_col, 'geometry']].rename(columns={nuts2_code_col: '_nuts_code'}).copy()
    joined = gpd.sjoin(asset_gdf, join_right, how='left', predicate='intersects')
    joined['country'] = joined['_nuts_code'].astype(str).str[:2].str.upper()
    joined.loc[joined['_nuts_code'].isna(), 'country'] = 'UNK'
    return joined


# ── Study Area Building ───────────────────────────────────────────────────────

def build_study_area(*gdfs, buffer_m=0.0, corridor_label=None):
    """Create a bounding-box study area from one or more GeoDataFrames.
    
    Parameters
    ----------
    *gdfs : GeoDataFrame
        Non-empty GeoDataFrames to include in the bounding box.
    buffer_m : float, optional
        Buffer distance in meters (default 0.0).
    corridor_label : str, optional
        Label for the analysis_scope column (default None).
    
    Returns
    -------
    GeoDataFrame
        Single-row GeoDataFrame with bounding-box geometry.
    """
    valid = []
    for frame in gdfs:
        gdf = _as_geodataframe(frame)
        if gdf is None or getattr(gdf, 'empty', True) or not isinstance(gdf, gpd.GeoDataFrame):
            continue
        gdf = gdf[gdf.geometry.notna()].copy()
        if not gdf.empty:
            valid.append(gdf)

    if not valid:
        raise ValueError('No non-empty GeoDataFrames available to build the study area.')

    base_crs = next((gdf.crs for gdf in valid if getattr(gdf, 'crs', None) is not None), 'EPSG:4326')
    minx = miny = float('inf')
    maxx = maxy = float('-inf')

    for gdf in valid:
        gdf_crs = getattr(gdf, 'crs', None)
        if gdf_crs is None:
            gdf = gdf.set_crs(base_crs, allow_override=True)
        elif gdf_crs != base_crs:
            gdf = gdf.to_crs(base_crs)
        bx0, by0, bx1, by1 = gdf.total_bounds
        if not np.isfinite([bx0, by0, bx1, by1]).all():
            continue
        minx, miny = min(minx, bx0), min(miny, by0)
        maxx, maxy = max(maxx, bx1), max(maxy, by1)

    if not np.isfinite([minx, miny, maxx, maxy]).all() or maxx <= minx or maxy <= miny:
        raise ValueError('Could not derive valid study-area bounds from provided geometries.')

    study_geom = box(minx, miny, maxx, maxy)
    if buffer_m > 0:
        study_geom = study_geom.buffer(buffer_m)

    label = corridor_label or 'corridor'
    return gpd.GeoDataFrame(
        {'analysis_scope': [label]},
        geometry=[study_geom],
        crs=base_crs
    )


def _select_infra_source(key, corridor_infrastructure_files, infrastructure_files):
    return _pick_existing([corridor_infrastructure_files[key], infrastructure_files[key]])


def _load_corridor_layers(corridor_infrastructure_files, infrastructure_files):
    layer_specs = [
        ('rail_edges', 'railway_edges', 'rail_edges'),
        ('road_edges', 'road_edges', 'road_edges '),
        ('iww_edges', 'iww_edges', 'iww_edges'),
        ('rail_stations', 'railway_stations', 'rail_stations'),
        ('iww_nodes', 'iww_nodes', 'iww_nodes'),
        ('ports', 'ports', 'ports'),
        ('airports', 'airports', 'airports'),
    ]
    layers = {}
    for layer_name, file_key, label in layer_specs:
        src = _select_infra_source(file_key, corridor_infrastructure_files, infrastructure_files)
        loaded = _timed_load(src, label)
        layers[layer_name] = _coerce_loaded_geo(loaded, layer_name)
    return layers


def _extract_iww_ports(iww_nodes):
    if not isinstance(iww_nodes, gpd.GeoDataFrame):
        return gpd.GeoDataFrame()
    if 'feature' in iww_nodes.columns:
        return iww_nodes[iww_nodes['feature'].eq('port')].copy()
    return gpd.GeoDataFrame(columns=iww_nodes.columns, crs=iww_nodes.crs)


def load_nuts2_background(nuts2_candidates, study_area=None):
    nuts2_file = _pick_existing(nuts2_candidates)
    if str(nuts2_file).endswith('.parquet'):
        nuts2 = _read_parquet_as_gdf(nuts2_file) if nuts2_file.exists() else gpd.GeoDataFrame()
    else:
        nuts2 = gpd.read_file(nuts2_file) if nuts2_file.exists() else gpd.GeoDataFrame()

    if nuts2.empty:
        print(f'[WARN] NUTS2 regions not loaded (checked: {nuts2_candidates})')
        return nuts2, nuts2_file

    print(f'NUTS2: {len(nuts2):,} regions from {nuts2_file}')
    if (
        isinstance(study_area, gpd.GeoDataFrame)
        and not study_area.empty
        and study_area.crs is not None
        and nuts2.crs != study_area.crs
    ):
        nuts2 = nuts2.to_crs(study_area.crs)

    if 'LEVL_CODE' in nuts2.columns:
        nuts2 = nuts2[nuts2['LEVL_CODE'].astype(str).eq('2') | nuts2['LEVL_CODE'].eq(2)].copy()

    return nuts2, nuts2_file


def _layer_score_for_target_crs(layer_target, ref_bg):
    if not isinstance(layer_target, gpd.GeoDataFrame) or layer_target.empty:
        return -1e12
    valid = layer_target[layer_target.geometry.notna() & ~layer_target.geometry.is_empty]
    if valid.empty:
        return -1e12

    rp = valid.geometry.representative_point()
    x = pd.Series(rp.x).replace([np.inf, -np.inf], np.nan).dropna()
    y = pd.Series(rp.y).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty or y.empty:
        return -1e12

    frac_deg = float(((x.abs() <= 180) & (y.abs() <= 90)).mean())
    if frac_deg <= 0:
        return -1e12

    score = frac_deg * 100.0

    if isinstance(ref_bg, gpd.GeoDataFrame) and not ref_bg.empty:
        lb = valid.total_bounds
        rb = ref_bg.total_bounds
        if np.isfinite(lb).all() and np.isfinite(rb).all() and lb[2] > lb[0] and lb[3] > lb[1] and rb[2] > rb[0] and rb[3] > rb[1]:
            ix0 = max(lb[0], rb[0])
            iy0 = max(lb[1], rb[1])
            ix1 = min(lb[2], rb[2])
            iy1 = min(lb[3], rb[3])
            inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
            area_l = (lb[2] - lb[0]) * (lb[3] - lb[1])
            area_r = (rb[2] - rb[0]) * (rb[3] - rb[1])
            union = area_l + area_r - inter
            iou = (inter / union) if union > 0 else 0.0
            score += iou * 1000.0

    return score


def fix_layer_to_target_crs(layer_name, layer_obj, target_crs='EPSG:4326', ref_bg=None):
    if layer_obj is None:
        return layer_obj

    layer = _coerce_loaded_geo(layer_obj, layer_name)
    if not isinstance(layer, gpd.GeoDataFrame):
        print(f'[WARN] {layer_name}: not a GeoDataFrame after coercion; skipping CRS projection')
        return layer_obj
    if layer.empty:
        return layer

    attempts = []
    if layer.crs is not None:
        try:
            base = layer.to_crs(target_crs) if str(layer.crs) != str(target_crs) else layer.copy()
            attempts.append((f'declared:{layer.crs}', base))
        except Exception:
            pass

    try:
        inferred = _infer_and_fix_crs(layer.copy())
        if isinstance(inferred, gpd.GeoDataFrame) and inferred.crs is not None:
            base = inferred.to_crs(target_crs) if str(inferred.crs) != str(target_crs) else inferred.copy()
            attempts.append((f'inferred:{inferred.crs}', base))
    except Exception:
        pass

    candidate_source_crs = ['EPSG:3035', 'EPSG:3857', 'EPSG:4326']
    if layer.crs is not None:
        declared = str(layer.crs)
        if declared not in candidate_source_crs:
            candidate_source_crs = [declared] + candidate_source_crs

    for source_crs in candidate_source_crs:
        try:
            cand = layer.copy().set_crs(source_crs, allow_override=True)
            cand = cand.to_crs(target_crs) if str(cand.crs) != str(target_crs) else cand
            attempts.append((f'override:{source_crs}', cand))
        except Exception:
            continue

    if not attempts:
        print(f'[WARN] {layer_name}: no valid CRS conversion attempt')
        return layer

    best_label, best_layer = max(attempts, key=lambda item: _layer_score_for_target_crs(item[1], ref_bg))
    print(f'[fix] {layer_name}: selected CRS path `{best_label}` -> {target_crs}')
    return best_layer


def prepare_corridor_data_context(
    corridor_infrastructure_files,
    infrastructure_files,
    nuts2_candidates,
    include_iww=True,
    target_corridor_label='corridor',
    target_corridor_code='N/A',
    target_corridor_data_label='N/A',
    target_corridor_buffer_m=50_000,
    target_crs='EPSG:4326',
):
    t_cell = perf_counter()

    layers = _load_corridor_layers(corridor_infrastructure_files, infrastructure_files)
    layers['ports'] = assign_primary_corridor(layers['ports'])
    layers['airports'] = assign_primary_corridor(layers['airports'])
    iww_ports = _extract_iww_ports(layers['iww_nodes'])

    print(f'Study corridor: {target_corridor_label} (code={target_corridor_code}, data label={target_corridor_data_label})')
    print(f"  Rail edges   : {len(layers['rail_edges']):,}")
    print(f"  Road edges   : {len(layers['road_edges']):,}")
    print(f"  Rail stations: {len(layers['rail_stations']):,}")
    print(f"  Ports        : {len(layers['ports']):,}")
    print(f"  Airports     : {len(layers['airports']):,}")
    print(f"  IWW edges    : {len(layers['iww_edges']):,}")
    print(f"  IWW ports    : {len(iww_ports):,}")
    print(f'[CELL TOTAL] setup elapsed: {perf_counter()-t_cell:.1f}s')
    print('Data loaded ✓')

    study_area = build_study_area(
        layers['rail_edges'],
        layers['road_edges'],
        layers['rail_stations'],
        layers['ports'],
        layers['airports'],
        layers['iww_edges'] if include_iww else None,
        iww_ports if include_iww else None,
        buffer_m=target_corridor_buffer_m,
        corridor_label=target_corridor_label,
    ).to_crs(target_crs)

    country_row = study_area.copy()
    xlim, ylim = _bounds_with_padding(study_area)

    nuts2, nuts2_file = load_nuts2_background(nuts2_candidates, study_area=study_area)

    ref_bg = None
    if isinstance(nuts2, gpd.GeoDataFrame) and not nuts2.empty:
        ref_bg = nuts2.to_crs(target_crs) if nuts2.crs != target_crs else nuts2
    elif isinstance(study_area, gpd.GeoDataFrame) and not study_area.empty:
        ref_bg = study_area

    for layer_name in ['road_edges', 'rail_edges', 'iww_edges', 'rail_stations', 'iww_nodes', 'ports', 'airports']:
        layers[layer_name] = fix_layer_to_target_crs(layer_name, layers.get(layer_name), target_crs=target_crs, ref_bg=ref_bg)

    iww_ports = _extract_iww_ports(layers['iww_nodes'])

    infra_vars = {
        'road_edges': layers['road_edges'],
        'rail_edges': layers['rail_edges'],
        'iww_nodes': layers['iww_nodes'],
        'ports': layers['ports'],
        'airports': layers['airports'],
    }

    print(f'[fix] enforced target CRS for infrastructure layers: {target_crs}')

    out = {
        'study_area': study_area,
        'country_row': country_row,
        'xlim': xlim,
        'ylim': ylim,
        'nuts2': nuts2,
        'nuts2_file': nuts2_file,
        'target_crs': target_crs,
        'iww_ports': iww_ports,
        'infra_vars': infra_vars,
    }
    out.update(layers)
    return out


# ── Parquet Reading ───────────────────────────────────────────────────────────

def _safe_read_parquet_any(path):
    """Read parquet file with fallback: tries file handle first, then direct path."""
    try:
        with open(path, 'rb') as fh:
            return pd.read_parquet(fh)
    except Exception:
        return pd.read_parquet(path)


def _parse_geom(value):
    """Parse WKB/WKT geometry from various formats (bytes, hex string, WKT)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if hasattr(value, 'geom_type'):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return wkb.loads(bytes(value))
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.startswith(('POINT', 'LINESTRING', 'POLYGON', 'MULTI', 'GEOMETRYCOLLECTION')):
                return wkt.loads(text)
            return wkb.loads(bytes.fromhex(text))
        except Exception:
            return None
    return None


def _read_parquet_as_gdf(path, default_crs='EPSG:4326'):
    """Read parquet file as GeoDataFrame, auto-detecting geometry column.
    
    Attempts standard geopandas.read_parquet first, then falls back to
    manual geometry parsing (WKB/WKT detection).
    """
    try:
        return gpd.read_parquet(path)
    except Exception:
        pass

    df = _safe_read_parquet_any(path)
    geom_col = next((c for c in df.columns if c.lower() in ('geometry', 'geom', 'the_geom', 'wkb_geometry')), None)
    if geom_col is None:
        geom_col = next((c for c in df.columns if 'geom' in c.lower()), None)

    if geom_col is None:
        return gpd.GeoDataFrame(df, geometry=None, crs=default_crs)

    geom = df[geom_col].apply(_parse_geom)
    out = df.drop(columns=[geom_col], errors='ignore').copy()
    return gpd.GeoDataFrame(out, geometry=geom, crs=default_crs)


def _read_parquet_as_gdf_fallback(path, default_crs='EPSG:4326'):
    """Read parquet as GeoDataFrame with timing and progress output."""
    t0 = perf_counter()
    df = pd.read_parquet(path)
    print(f"    [fallback] pd.read_parquet {path.name}: {perf_counter()-t0:.1f}s | rows={len(df):,}")

    geom_col = next((c for c in df.columns if c.lower() in ('geometry', 'geom', 'the_geom', 'wkb_geometry')), None)
    if geom_col is None:
        geom_col = next((c for c in df.columns if 'geom' in c.lower()), None)
    if geom_col is None:
        raise ValueError(f'No geometry column found in {path.name}')

    t1 = perf_counter()
    geom = df[geom_col].apply(_parse_geom)
    out = gpd.GeoDataFrame(df.drop(columns=[geom_col], errors='ignore'), geometry=geom, crs=default_crs)
    out = out[out.geometry.notna()].copy()
    print(f"    [fallback] geometry parse {path.name}: {perf_counter()-t1:.1f}s | kept={len(out):,}")
    return out


# ── Disruption Calculations ───────────────────────────────────────────────────

def _compute_disruption_cost_eur(df):
    """Calculate disruption cost in euros: rerouting vs isolation mode.
    
    Formula:
    - Rerouting: 1000 * 0.25 * travel_time_h * value_ths_tons * days_disrupted / 360
    - Isolated: 1000 * value_ths_tons * days_disrupted / 360 * kfactor
    """
    def _col(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors='coerce').fillna(default)
        return pd.Series(default, index=df.index, dtype='float64')

    original_tt = _col('original_travel_time_h', 0.0)
    value = _col('value', 0.0)
    days = _col('days_disrupted_computed', 0.0)
    kfactor = _col('kfactor', 0.0)

    rerouted_cost = 1000.0 * 0.25 * original_tt * value * days / 360.0
    isolated_cost = 1000.0 * value * days / 360.0 * kfactor

    status = df.get('status', pd.Series('', index=df.index)).astype(str).str.lower()
    out = np.where(status.eq('rerouted'), rerouted_cost, isolated_cost)
    return pd.Series(out, index=df.index, dtype='float64')


def _vuln_to_days(vuln_ratio, mode_col, recovery_table):
    """Interpolate disrupted days from recovery table for a given vulnerability ratio.
    
    Parameters
    ----------
    vuln_ratio : float
        Vulnerability ratio (0 to 1).
    mode_col : str
        Mode column name in recovery_table (e.g., 'road', 'rail', 'port').
    recovery_table : DataFrame
        Recovery table with 'frac_damage' column and mode columns.
    
    Returns
    -------
    float
        Interpolated disrupted days.
    """
    frac = recovery_table['frac_damage'].values
    days = recovery_table[mode_col].values
    f = interp1d(frac, days, kind='linear', bounds_error=False, fill_value=(0.0, days[-1]))
    return float(f(vuln_ratio))


def concat_tagged_scenarios(file_list, status_label):
    """Load parquet scenario files and tag rows with a disruption status.

    Parameters
    ----------
    file_list : list[Path]
        Scenario parquet files.
    status_label : str
        Status to assign, e.g. ``'isolated'`` or ``'rerouted'``.

    Returns
    -------
    DataFrame
        Concatenated scenario rows (empty DataFrame if no valid pieces).
    """
    pieces = []
    for fpath in file_list:
        try:
            part = _safe_read_parquet_any(fpath)
        except Exception as exc:
            print(f'[WARN] Failed reading scenario file: {fpath} ({exc})')
            continue
        part['status'] = status_label
        pieces.append(part)
    valid = [df for df in pieces if isinstance(df, pd.DataFrame) and not df.empty]
    return pd.concat(valid, ignore_index=True, copy=False) if valid else pd.DataFrame()


def load_or_build_disruption(disruption_path, isolated_files, rerouted_files, use_cache=True, force_rebuild=False):
    """Load cached disruption table or rebuild from isolated/rerouted scenario files.

    Returns a tuple ``(disruption_combined, isolated_df, rerouted_df)``.
    """
    isolated_df = pd.DataFrame()
    rerouted_df = pd.DataFrame()

    if use_cache and disruption_path.exists() and not force_rebuild:
        try:
            out = _safe_read_parquet_any(disruption_path)
            return out, isolated_df, rerouted_df
        except Exception as exc:
            print(f'[WARN] Failed reading cached disruption file: {disruption_path} ({exc})')
            print('[WARN] Falling back to rebuild from isolated/rerouted scenario files.')

    isolated_df = concat_tagged_scenarios(isolated_files, 'isolated')
    rerouted_df = concat_tagged_scenarios(rerouted_files, 'rerouted')

    required_id_cols = ['origin_industry_id', 'dest_industry_id']
    if all(col in isolated_df.columns for col in required_id_cols):
        isolated_df = isolated_df.dropna(subset=required_id_cols)
    if all(col in rerouted_df.columns for col in required_id_cols):
        rerouted_df = rerouted_df.dropna(subset=required_id_cols)

    out = pd.concat([rerouted_df, isolated_df], ignore_index=True, copy=False)
    out['disruption_state'] = out['status']
    try:
        disruption_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(disruption_path, index=False)
    except Exception as exc:
        print(f'[WARN] Could not write cache parquet: {disruption_path} ({exc})')
    return out, isolated_df, rerouted_df


def prepare_disruption_sectors_from_od(disruption_df, od_df):
    """Attach OD sector/value attributes to disruption rows using vectorized lookup."""
    def _first_nonempty(series):
        vals = series.dropna().astype(str).str.strip()
        vals = vals[vals.ne('')]
        return vals.iloc[0] if not vals.empty else np.nan

    out = disruption_df.rename(columns={
        'origin_industry_id': 'origin_id',
        'dest_industry_id': 'destination_id',
    }, copy=False)

    for key in ['origin_id', 'destination_id']:
        out[key] = out[key].astype(str).str.strip()

    od_join_cols = ['origin_id', 'destination_id', 'origin_sector', 'value']
    if 'origin_NUTS2' in od_df.columns:
        od_join_cols.append('origin_NUTS2')

    od_join_df = od_df[od_join_cols].copy()
    for key in ['origin_id', 'destination_id']:
        od_join_df[key] = od_join_df[key].astype(str).str.strip()

    od_agg = {'origin_sector': _first_nonempty, 'value': 'sum'}
    if 'origin_NUTS2' in od_join_df.columns:
        od_agg['origin_NUTS2'] = _first_nonempty

    od_lookup = (
        od_join_df.groupby(['origin_id', 'destination_id'], sort=False, as_index=True)
        .agg(od_agg)
    )

    od_keys = pd.MultiIndex.from_frame(out[['origin_id', 'destination_id']])
    for col in ['origin_sector', 'value'] + (['origin_NUTS2'] if 'origin_NUTS2' in od_lookup.columns else []):
        out[col] = od_lookup[col].reindex(od_keys).to_numpy()

    return out[
        out['origin_sector'].notna() & (out['origin_sector'].astype(str).str.strip() != '')
    ].copy()


def ensure_original_travel_time(df, source_rerouted_df=None):
    """Ensure ``original_travel_time_h`` is populated, with optional fallback join from rerouted source rows."""
    out = df.copy()
    if 'original_travel_time_h' not in out.columns:
        out['original_travel_time_h'] = np.nan

    tt_candidates = [
        'original_travel_time_h',
        'original_travel_time',
        'travel_time_original_h',
        'old_travel_time_h',
        'original_tt_h',
    ]

    for tt_col in tt_candidates:
        if tt_col in out.columns and tt_col != 'original_travel_time_h':
            out['original_travel_time_h'] = out['original_travel_time_h'].fillna(
                pd.to_numeric(out[tt_col], errors='coerce')
            )

    need_tt = out['status'].astype(str).str.lower().eq('rerouted') & out['original_travel_time_h'].isna()
    if not need_tt.any() or source_rerouted_df is None or source_rerouted_df.empty:
        return out

    src_tt_col = next((c for c in tt_candidates if c in source_rerouted_df.columns), None)
    join_keys_pref = ['origin_id', 'destination_id', 'failed_infr_id', 'failed_infr_type', 'return_period']
    join_keys = [k for k in join_keys_pref if (k in out.columns and k in source_rerouted_df.columns)]

    if src_tt_col is None or len(join_keys) < 2:
        return out

    rhs = source_rerouted_df[join_keys + [src_tt_col]].rename(columns={src_tt_col: '_tt_rhs'})
    rhs = rhs.drop_duplicates(subset=join_keys)

    lhs = out.reset_index().rename(columns={'index': '_row_id'})
    lhs = lhs.merge(rhs, on=join_keys, how='left')
    lhs['_tt_rhs'] = pd.to_numeric(lhs['_tt_rhs'], errors='coerce')

    fill_mask = lhs['status'].astype(str).str.lower().eq('rerouted') & lhs['original_travel_time_h'].isna()
    lhs.loc[fill_mask, 'original_travel_time_h'] = lhs.loc[fill_mask, '_tt_rhs']
    return lhs.drop(columns=['_tt_rhs']).set_index('_row_id').sort_index()


def build_vulnerability_lookup_vectorized(haz_dir, tent_sources):
    """Build long vulnerability lookup ``(infra_type, infra_id_norm, rp_num, vuln_ratio_at_rp)`` from TENT hazards."""
    rp_pattern = re.compile(r'rp0*(\d+)', flags=re.I)
    all_parts = []

    for infra_type, cfg in tent_sources.items():
        haz_file = Path(haz_dir) / cfg['file']
        if not haz_file.exists():
            continue

        haz = _safe_read_parquet_any(haz_file)
        # enrich_key overrides key for enrichment lookups (e.g. port_code vs port_name)
        enrich_cfg_key = cfg.get('enrich_key', cfg['key'])
        key_col = next((c for c in haz.columns if str(c).lower() == str(enrich_cfg_key).lower()), None)
        if key_col is None:
            # fall back to the plotting key
            key_col = next((c for c in haz.columns if str(c).lower() == str(cfg['key']).lower()), None)
        if key_col is None:
            continue

        rp_cols = [c for c in haz.columns if rp_pattern.search(str(c))]
        if not rp_cols:
            continue

        tmp = haz[[key_col] + rp_cols].copy()
        # Road/rail IDs in hazard files have an 'europe_latest-XX_YYY' prefix;
        # _canonical_transport_id strips it so '27532' and 'europe_latest-2_7532' both map to '27532'.
        # Other types (ports, airports) use plain codes so _norm_key suffices.
        _id_norm_fn = _canonical_transport_id if infra_type in ('road edge', 'rail edge') else _norm_key
        tmp['infra_id_norm'] = tmp[key_col].map(_id_norm_fn)

        long = tmp.melt(
            id_vars=['infra_id_norm'],
            value_vars=rp_cols,
            var_name='rp_col',
            value_name='vuln_ratio_at_rp',
        )
        long['vuln_ratio_at_rp'] = pd.to_numeric(long['vuln_ratio_at_rp'], errors='coerce')
        long = long.dropna(subset=['infra_id_norm', 'vuln_ratio_at_rp'])

        long['rp_num'] = long['rp_col'].astype(str).str.extract(rp_pattern, expand=False)
        long['rp_num'] = pd.to_numeric(long['rp_num'], errors='coerce').round().astype('Int64')
        long = long.dropna(subset=['rp_num'])

        long['infra_type'] = infra_type
        all_parts.append(long[['infra_type', 'infra_id_norm', 'rp_num', 'vuln_ratio_at_rp']])

    if not all_parts:
        return pd.DataFrame(columns=['infra_type', 'infra_id_norm', 'rp_num', 'vuln_ratio_at_rp'])

    out = pd.concat(all_parts, ignore_index=True)
    out = out.sort_values(['infra_type', 'infra_id_norm', 'rp_num'])
    return out.drop_duplicates(subset=['infra_type', 'infra_id_norm', 'rp_num'], keep='first')


def attach_vulnerability_from_tent(df, haz_dir, tent_sources, geo_layers=None):
    """Attach ``vuln_ratio_at_rp`` to disruption rows using TENT hazard layers and RP matching.

    Parameters
    ----------
    geo_layers : dict, optional
        Mapping of layer name to GeoDataFrame/DataFrame.  Currently used to resolve the
        port ``port_code → port_name`` translation: the TENT hazard file keys on
        ``port_name`` but disruption rows carry ``port_code`` in ``failed_infr_id``.
        Pass ``{'ports': ports_gdf}`` to enable the bridge.
    """
    out = df.copy()

    mode_canon = {
        'rail': 'rail edge', 'rails': 'rail edge', 'rail edge': 'rail edge',
        'road': 'road edge', 'roads': 'road edge', 'road edge': 'road edge',
        'iww': 'iww port', 'iww port': 'iww port', 'iww ports': 'iww port', 'iww_port': 'iww port', 'iww_ports': 'iww port',
        'airport': 'airport', 'airports': 'airport',
        'port': 'port', 'ports': 'port',
    }

    id_col = 'failed_infr_id' if 'failed_infr_id' in out.columns else ('failed_infra_id' if 'failed_infra_id' in out.columns else None)
    rp_col = next((c for c in ['return_period', 'rp', 'RP', 'returnperiod'] if c in out.columns), None)
    if id_col is None or rp_col is None:
        out['vuln_ratio_at_rp'] = np.nan
        return out

    # Build port_code -> _norm_key(port_name) bridge from the ports geometry layer.
    # Disruption failed_infr_id for ports stores port_code (e.g. 'DEHAM'), but the
    # TENT hazard file keys on port_name (e.g. 'Port of Hamburg').
    _port_code_to_name_norm: dict = {}
    if geo_layers is not None:
        _ports_gdf = geo_layers.get('ports')
        if isinstance(_ports_gdf, pd.DataFrame) and 'port_code' in _ports_gdf.columns and 'port_name' in _ports_gdf.columns:
            _tmp = _ports_gdf[['port_code', 'port_name']].dropna().copy()
            _port_code_to_name_norm = {
                _norm_key(code): _norm_key(name)
                for code, name in zip(_tmp['port_code'], _tmp['port_name'])
                if _norm_key(code) and _norm_key(name)
            }

    out['_infra_type_raw'] = (
        out['failed_infr_type'].fillna('').astype(str)
        .str.split('|').str[0].str.lower().str.replace('_', ' ', regex=False).str.strip()
    )
    out['_infra_id_raw'] = (
        out[id_col].fillna('').astype(str)
        .str.split('|').str[0].str.strip()
    )
    out['infra_type'] = out['_infra_type_raw'].map(mode_canon)
    # Use _canonical_transport_id for edge types so that disruption IDs like '27532' and
    # hazard IDs like 'europe_latest-2_7532' normalise to the same token.
    # Mask-split rather than np.where(cond, map(a), map(b)): np.where evaluates
    # BOTH branches unconditionally, which on tens of millions of rows means
    # running the (Python-level, regex-using) .map() twice over the full
    # column instead of once each over half of it.
    _edge_types = {'road edge', 'rail edge'}
    _is_edge = out['infra_type'].isin(_edge_types)
    out['infra_id_norm'] = pd.Series(pd.NA, index=out.index, dtype=object)
    out.loc[_is_edge, 'infra_id_norm'] = _map_unique(out.loc[_is_edge, '_infra_id_raw'], _canonical_transport_id)
    out.loc[~_is_edge, 'infra_id_norm'] = _map_unique(out.loc[~_is_edge, '_infra_id_raw'], _norm_key)
    # Translate port_code → port_name (normalised) for port rows so they align with
    # the TENT hazard lookup which is keyed on port_name.
    if _port_code_to_name_norm:
        _port_mask = out['infra_type'].eq('port')
        out.loc[_port_mask, 'infra_id_norm'] = (
            out.loc[_port_mask, 'infra_id_norm']
            .map(lambda x: _port_code_to_name_norm.get(x, x))
        )
    out['rp_num'] = pd.to_numeric(out[rp_col], errors='coerce')
    if out['rp_num'].isna().all():
        out['rp_num'] = pd.to_numeric(
            out[rp_col].astype(str).str.extract(r'(\d+)', expand=False),
            errors='coerce',
        )
    out['rp_num'] = out['rp_num'].round().astype('Int64')

    vuln_lookup = build_vulnerability_lookup_vectorized(haz_dir, tent_sources)
    if vuln_lookup.empty:
        out['vuln_ratio_at_rp'] = np.nan
        return out

    out = out.merge(
        vuln_lookup,
        on=['infra_type', 'infra_id_norm', 'rp_num'],
        how='left',
    )
    return out


def compute_days_from_vulnerability(df, recovery_days_table, default_vuln_ratio=None):
    """Compute ``days_disrupted_computed`` from vulnerability and mode-specific recovery curves."""
    out = df.copy()

    mode_to_recovery = (
        out['failed_infr_type'].fillna('').astype(str)
        .str.split('|').str[0].str.lower().str.replace('_', ' ', regex=False).str.strip()
        .map({
            'road': 'road', 'roads': 'road', 'road edge': 'road',
            'rail': 'rail', 'rails': 'rail', 'rail edge': 'rail',
            'port': 'port', 'ports': 'port', 'iww': 'port', 'iww port': 'port', 'iww ports': 'port',
            'airport': 'airport', 'airports': 'airport',
        })
    )

    vuln_source = out.get('vuln_ratio_at_rp', default_vuln_ratio if default_vuln_ratio is not None else np.nan)
    if not isinstance(vuln_source, pd.Series):
        vuln_source = pd.Series(vuln_source, index=out.index)
    vuln = pd.to_numeric(vuln_source, errors='coerce').clip(lower=0.0, upper=1.0)

    frac = pd.to_numeric(recovery_days_table['frac_damage'], errors='coerce').to_numpy()
    vuln_np = vuln.to_numpy(dtype='float64', copy=False)
    vuln_valid = np.isfinite(vuln_np)
    mode_np = mode_to_recovery.to_numpy(dtype=object, copy=False)

    unique_modes = [m for m in pd.Series(mode_np).dropna().unique() if m in recovery_days_table.columns]
    if not unique_modes:
        out['vuln_ratio_at_rp'] = vuln
        out['days_disrupted_computed'] = 0.0
        return out

    interp_matrix = np.zeros((len(unique_modes), len(out)), dtype='float64')
    valid_mode_rows = np.zeros(len(unique_modes), dtype=bool)

    for i, rec_col in enumerate(unique_modes):
        days_curve = pd.to_numeric(recovery_days_table[rec_col], errors='coerce').to_numpy()
        valid = np.isfinite(frac) & np.isfinite(days_curve)
        if valid.sum() < 2:
            continue

        x = frac[valid]
        y = days_curve[valid]
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        interp_matrix[i, :] = np.interp(vuln_np, x, y, left=0.0, right=float(y[-1]))
        valid_mode_rows[i] = True

    mode_to_idx = {m: i for i, m in enumerate(unique_modes) if valid_mode_rows[i]}
    mode_idx = (
        pd.Series(mode_np)
        .map(mode_to_idx)
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )

    days_out_np = np.zeros(len(out), dtype='float64')
    pick_mask = (mode_idx >= 0) & vuln_valid
    if pick_mask.any():
        row_idx = mode_idx[pick_mask]
        col_idx = np.nonzero(pick_mask)[0]
        days_out_np[pick_mask] = interp_matrix[row_idx, col_idx]

    out['vuln_ratio_at_rp'] = vuln
    out['days_disrupted_computed'] = days_out_np
    return out


# ── Data Processing and Aggregation ───────────────────────────────────────────

def extract_cost_column(df, default_value=0.0):
    """Extract the appropriate cost column from a disruption dataframe.
    
    Tries multiple column names in order of preference. Returns normalized series.
    
    Parameters
    ----------
    df : DataFrame
        Disruption dataframe with cost columns.
    default_value : float
        Default fill value for missing data.
    
    Returns
    -------
    Series
        Numeric cost column (or None if no valid cost column found).
    """
    cost_candidates = [
        'disruption_cost_eur', 'total_cost_eur', 'rerouting_cost_eur_event',
        'isolation_cost_eur_event', 'rerouting_cost_eur', 'isolation_cost_eur'
    ]
    for col in cost_candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').fillna(default_value)

    if any(col in df.columns for col in ['days_disrupted_computed', 'kfactor', 'original_travel_time_h']):
        return _compute_disruption_cost_eur(df).fillna(default_value)

    if 'value' in df.columns:
        return pd.to_numeric(df['value'], errors='coerce').fillna(default_value)

    return None


def normalize_sector_names(sector_series, sector_code_map=None):
    """Normalize sector names: lowercase, strip whitespace, optionally map codes.
    
    Parameters
    ----------
    sector_series : Series
        Series containing sector names/codes.
    sector_code_map : dict, optional
        Mapping from codes to sector names (e.g., SECTOR_CODE_TO_NAME).
    
    Returns
    -------
    Series
        Normalized sector names.
    """
    normalized = (
        sector_series.fillna('unknown')
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if sector_code_map is not None:
        code_map_lower = {str(k).strip().lower(): str(v).strip().lower() for k, v in sector_code_map.items()}
        normalized = normalized.map(lambda v: code_map_lower.get(v, v))
    return normalized


def aggregate_disruption_by_origin(disruption_df, origin_id_col='origin_id'):
    """Aggregate disruption data by origin ID, summing costs and finding modal sector.
    
    Parameters
    ----------
    disruption_df : DataFrame
        Disruption data with origin ID and sector columns.
    origin_id_col : str
        Name of the origin ID column.
    
    Returns
    -------
    DataFrame
        Aggregated data with one row per origin ID.
    """
    agg_spec = {'total_cost_eur': ('total_cost_eur', 'sum')}
    
    if 'origin_sector' in disruption_df.columns:
        agg_spec['origin_sector'] = (
            'origin_sector',
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        )
    
    work = disruption_df.copy()
    work[origin_id_col] = work[origin_id_col].map(_norm_key)
    work = work[work[origin_id_col].ne('')].copy()

    aggregated = (
        work.groupby(origin_id_col, as_index=False)
        .agg(**agg_spec)
        .rename(columns={origin_id_col: 'industry_id'})
    )
    aggregated['industry_id'] = aggregated['industry_id'].map(_norm_key)
    aggregated = aggregated[aggregated['industry_id'].ne('')].copy()
    return aggregated


def merge_geometry_to_disruption(disruption_agg, location_gdf, location_id_col):
    """Merge geometry from location GeoDataFrame to aggregated disruption data.
    
    Parameters
    ----------
    disruption_agg : DataFrame
        Aggregated disruption data with 'industry_id' column.
    location_gdf : GeoDataFrame
        Location GeoDataFrame with ID and geometry.
    location_id_col : str
        Name of the ID column in location_gdf.
    
    Returns
    -------
    GeoDataFrame
        Merged data with geometry, filtered to non-null geometries.
    """
    left = disruption_agg.copy()
    left['industry_id'] = left['industry_id'].map(_norm_key)
    left = left[left['industry_id'].ne('')].copy()

    right = location_gdf[[location_id_col, 'geometry']].copy()
    right['_loc_norm_id'] = right[location_id_col].map(_norm_key)
    right = right[right['_loc_norm_id'].ne('')].drop_duplicates('_loc_norm_id')

    merged = left.merge(
        right[['_loc_norm_id', 'geometry']],
        left_on='industry_id',
        right_on='_loc_norm_id',
        how='left'
    )
    result = gpd.GeoDataFrame(
        merged.drop(columns=[location_id_col, '_loc_norm_id'], errors='ignore'),
        geometry='geometry',
        crs=location_gdf.crs
    )
    return result[result.geometry.notna()].copy()


def scale_by_log_value(values, min_size=40, max_size=260):
    """Scale sizes logarithmically based on values.
    
    Used for bubble map sizing where visualization range is (min_size, max_size)
    and values span logarithmically from min to max observed.
    
    Parameters
    ----------
    values : Series or array
        Values to scale (e.g., costs).
    min_size : float
        Minimum output size.
    max_size : float
        Maximum output size.
    
    Returns
    -------
    ndarray or Series
        Scaled sizes.
    """
    values = pd.Series(values) if not isinstance(values, pd.Series) else values.copy()
    values_safe = values.clip(lower=0.0)
    log_vals = np.log10(values_safe + 1.0)
    log_min = float(log_vals.min())
    log_max = float(log_vals.max())
    
    if log_max <= log_min:
        return pd.Series((min_size + max_size) / 2, index=values.index)
    
    return min_size + ((log_vals - log_min) / (log_max - log_min)) * (max_size - min_size)


def get_robust_bounds(gdf, q_low=5, q_high=95, pad_frac=0.15):
    """Compute robust spatial bounds using percentiles, with optional padding.
    
    Avoids outliers by using percentile-based bounds rather than total bounds.
    Falls back to total_bounds if insufficient data.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Spatial data to compute bounds for.
    q_low : float
        Lower percentile for bounds (default 5).
    q_high : float
        Upper percentile for bounds (default 95).
    pad_frac : float
        Fraction of range to pad on each side (default 0.15).
    
    Returns
    -------
    tuple or None
        Bounds as (minx, miny, maxx, maxy), or None if invalid.
    """
    if gdf.empty:
        return None
    
    rp = gdf.geometry.representative_point()
    x = pd.Series(rp.x).replace([np.inf, -np.inf], np.nan).dropna()
    y = pd.Series(rp.y).replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(x) >= 8 and len(y) >= 8:
        x1, x2 = np.nanpercentile(x, [q_low, q_high])
        y1, y2 = np.nanpercentile(y, [q_low, q_high])
        if np.isfinite([x1, x2, y1, y2]).all() and x2 > x1 and y2 > y1:
            px = max((x2 - x1) * pad_frac, 0.01)
            py = max((y2 - y1) * pad_frac, 0.01)
            return (x1 - px, y1 - py, x2 + px, y2 + py)
    
    bx = gdf.total_bounds
    if not np.isfinite(bx).all() or bx[2] <= bx[0] or bx[3] <= bx[1]:
        return None
    
    xspan = bx[2] - bx[0]
    yspan = bx[3] - bx[1]
    
    min_span = 1.5 if (gdf.crs and '4326' in str(gdf.crs)) else 150000.0
    
    if xspan < min_span:
        cx = 0.5 * (bx[0] + bx[2])
        bx[0], bx[2] = cx - 0.5 * min_span, cx + 0.5 * min_span
        xspan = min_span
    if yspan < min_span:
        cy = 0.5 * (bx[1] + bx[3])
        bx[1], bx[3] = cy - 0.5 * min_span, cy + 0.5 * min_span
        yspan = min_span
    
    px = xspan * pad_frac
    py = yspan * pad_frac
    return (bx[0] - px, bx[1] - py, bx[2] + px, bx[3] + py)


def apply_bounds_to_ax(ax, bounds):
    """Apply spatial bounds to a matplotlib axis.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Axes to set bounds on.
    bounds : tuple
        Bounds as (minx, miny, maxx, maxy).
    """
    if bounds is None:
        return
    minx, miny, maxx, maxy = bounds
    if np.isfinite([minx, miny, maxx, maxy]).all() and maxx > minx and maxy > miny:
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)


def create_industry_bubble_legend(bubble_labels=None, bubble_markers=None):
    """Create bubble size legend handles and labels for industry cost maps.
    
    Parameters
    ----------
    bubble_labels : list[str], optional
        Legend labels to show. Defaults to ['10', '100', '1K', '10K'].
    bubble_markers : list[float], optional
        Marker sizes for the legend bubbles. Defaults to [3.5, 6.0, 9.0, 12.0].
    
    Returns
    -------
    tuple
        (handles, labels) for use in fig.legend().
    """
    from matplotlib.lines import Line2D

    bubble_labels = ['10', '100', '1K', '10K'] if bubble_labels is None else list(bubble_labels)
    bubble_markers = [3.5, 6.0, 9.0, 12.0] if bubble_markers is None else list(bubble_markers)

    bubble_handles = [
        Line2D(
            [0], [0], marker='o', linestyle='None', markersize=size,
            markerfacecolor='none', markeredgecolor='0.35', markeredgewidth=1.0
        )
        for size in bubble_markers
    ]
    return bubble_handles, bubble_labels


def create_sector_legend(sector_cmap):
    """Create sector color legend handles.
    
    Parameters
    ----------
    sector_cmap : dict
        Mapping from sector name to color.
    
    Returns
    -------
    list
        Handles for use in fig.legend()
    """
    from matplotlib.lines import Line2D
    
    return [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
               markeredgecolor='white', markersize=8, label=sector)
        for sector, color in sorted(sector_cmap.items())
    ]


def prepare_industry_disruption_plot_data(disruption_df, industry_locs_gdf, location_id_col, plot_rps, sector_code_map=None):
    """Prepare aggregated disruption data by return period for plotting.
    
    Handles: cost extraction, aggregation, geometry merging, sector normalization.
    
    Parameters
    ----------
    disruption_df : DataFrame
        Disruption data with return_period/rp column and origin_sector.
    industry_locs_gdf : GeoDataFrame
        Industry locations with geometry.
    location_id_col : str
        ID column in industry_locs_gdf.
    plot_rps : list
        Return periods to plot (e.g., [10, 50, 100, 500]).
    sector_code_map : dict, optional
        Mapping from sector codes to names.
    
    Returns
    -------
    dict
        {rp: {'plot_gdf': GeoDataFrame, 'nuts2_plot': GeoDataFrame}, ...}
        or {rp: None} if no data for that RP.
    """
    rp_col = next((c for c in ['return_period', 'rp', 'RP', 'returnperiod'] 
                   if c in disruption_df.columns), None)
    if rp_col is None:
        raise ValueError('No return-period column found in disruption data')
    
    src_all = disruption_df.copy()
    src_all['_rp'] = pd.to_numeric(src_all[rp_col], errors='coerce')
    src_all['total_cost_eur'] = extract_cost_column(src_all)
    if pd.to_numeric(src_all['total_cost_eur'], errors='coerce').fillna(0.0).max() <= 0:
        if 'value' in src_all.columns:
            src_all['total_cost_eur'] = pd.to_numeric(src_all['value'], errors='coerce').fillna(0.0)
    
    locs = industry_locs_gdf.copy()
    locs[location_id_col] = locs[location_id_col].map(_norm_key)
    
    rp_plot_data = {}
    all_costs = []

    loc_keys = set(locs[location_id_col].dropna().astype(str))
    
    for target_rp in plot_rps:
        src = src_all[src_all['_rp'].eq(target_rp)].copy()
        src = src[src['total_cost_eur'] > 0].copy()

        
        if src.empty:
            rp_plot_data[target_rp] = None
            continue
        
        ind = aggregate_disruption_by_origin(src, origin_id_col='origin_id')
        matched_ids = set(ind['industry_id'].dropna().astype(str)) & loc_keys
        plot_gdf = merge_geometry_to_disruption(ind, locs, location_id_col)
        
        if plot_gdf.empty:
            rp_plot_data[target_rp] = None
            continue
        
        if 'origin_sector' in plot_gdf.columns:
            plot_gdf['origin_sector'] = normalize_sector_names(plot_gdf['origin_sector'], sector_code_map)
            plot_gdf = plot_gdf[~plot_gdf['origin_sector'].isin(['energy', 'unknown'])].copy()
        
        if plot_gdf.empty:
            rp_plot_data[target_rp] = None
            continue
        
        rp_plot_data[target_rp] = {'plot_gdf': plot_gdf, 'nuts2_plot': gpd.GeoDataFrame()}
        all_costs.append(plot_gdf['total_cost_eur'].values)
    
    return rp_plot_data, (np.concatenate(all_costs) if all_costs else np.array([1.0]))


def plot_industry_disruption_panels(
    disruption_df,
    industry_locs_gdf,
    location_id_col,
    plot_rps,
    nuts2=None,
    sector_code_map=None,
    corridor_label=None,
    figsize=(16, 14),
    bubble_labels=None,
    bubble_markers=None,
    show=True,
):
    """Plot RP-specific failing-industry bubble panels.
    
    Parameters
    ----------
    disruption_df : DataFrame
        Disruption data with return-period and cost fields.
    industry_locs_gdf : GeoDataFrame
        Industry locations with geometry.
    location_id_col : str
        Matching ID column in industry_locs_gdf.
    plot_rps : list
        Return periods to plot.
    nuts2 : GeoDataFrame, optional
        Background polygons.
    sector_code_map : dict, optional
        Mapping from sector codes to sector names.
    corridor_label : str, optional
        Label appended to subplot titles.
    figsize : tuple, optional
        Figure size.
    bubble_labels : list[str], optional
        Bubble legend labels.
    bubble_markers : list[float], optional
        Bubble legend marker sizes.
    show : bool, optional
        Whether to call plt.show().
    
    Returns
    -------
    dict
        Contains the matplotlib figure, axes, plot data, and cost array.
    """
    import matplotlib.pyplot as plt

    rp_plot_data, all_costs = prepare_industry_disruption_plot_data(
        disruption_df,
        industry_locs_gdf,
        location_id_col,
        plot_rps,
        sector_code_map=sector_code_map,
    )

    if not any(v is not None for v in rp_plot_data.values()):
        print('[WARN] No industries left after filtering.')
        print('[INFO] Possible causes: wrong location ID column, no geometry matches after merge, or all sectors normalized to energy/unknown.')
        return {'fig': None, 'axes': None, 'rp_plot_data': rp_plot_data, 'all_costs': all_costs}

    sector_cmap = {
        'chemicals': '#1f77b4',
        'commercial': '#ff7f0e',
        'food and beverage': '#d62728',
        'intensive livestock': '#9467bd',
        'metals': '#8c564b',
        'minerals': '#e377c2',
        'paper and wood': '#7f7f7f',
    }

    bubble_handles, bubble_labels = create_industry_bubble_legend(
        bubble_labels=bubble_labels,
        bubble_markers=bubble_markers,
    )
    sector_handles = create_sector_legend(sector_cmap)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=False)
    fig.set_facecolor('white')
    fig.subplots_adjust(left=0.035, right=0.99, top=0.83, bottom=0.16, wspace=0.06, hspace=0.12)
    axes = np.asarray(axes).flatten()

    log_vals = np.log10(np.asarray(all_costs, dtype=float) + 1.0)

    def _scale_bubble(value):
        scaled = np.log10(max(float(value), 0.0) + 1.0)
        log_min, log_max = float(np.nanmin(log_vals)), float(np.nanmax(log_vals))
        if not np.isfinite(log_min) or not np.isfinite(log_max) or log_max <= log_min:
            return 93.0
        return 6.0 + ((scaled - log_min) / (log_max - log_min)) * 174.0

    for ax, target_rp in zip(axes, plot_rps):
        pack = rp_plot_data.get(target_rp)

        if pack is None:
            ax.set_axis_off()
            ax.set_title(f'RP{target_rp}\n(no data)', fontsize=13, pad=8)
            continue

        plot_gdf = pack['plot_gdf']
        nuts2_rp = nuts2.copy() if isinstance(nuts2, gpd.GeoDataFrame) and not nuts2.empty else gpd.GeoDataFrame()
        if not nuts2_rp.empty and nuts2_rp.crs != plot_gdf.crs:
            try:
                nuts2_rp = nuts2_rp.to_crs(plot_gdf.crs)
            except Exception:
                pass

        if not nuts2_rp.empty:
            nuts2_rp.plot(ax=ax, color='#f7f7f7', edgecolor='black', linewidth=0.5, zorder=0)

        for sector, sector_df in plot_gdf.groupby('origin_sector', dropna=False):
            sector_df.plot(
                ax=ax,
                markersize=sector_df['total_cost_eur'].map(_scale_bubble),
                color=sector_cmap.get(str(sector).strip().lower(), 'tab:blue'),
                alpha=0.72,
                edgecolor='white',
                linewidth=0.5,
                zorder=3,
            )

        if not nuts2_rp.empty:
            nuts2_rp.boundary.plot(ax=ax, color='black', linewidth=0.5, zorder=1)

        ax.set_axis_off()
        title = f'RP{target_rp}'
        if corridor_label:
            title += f' ({corridor_label})'
        ax.set_title(title, fontsize=12, pad=7)

        try:
            minx, miny, maxx, maxy = plot_gdf.total_bounds
            if np.isfinite([minx, miny, maxx, maxy]).all() and maxx > minx and maxy > miny:
                padx = (maxx - minx) * 0.2
                pady = (maxy - miny) * 0.2
                ax.set_xlim(minx - padx, maxx + padx)
                ax.set_ylim(miny - pady, maxy + pady)
        except Exception:
            try:
                bounds = get_robust_bounds(plot_gdf)
                apply_bounds_to_ax(ax, bounds)
            except Exception:
                pass

        ax.set_aspect('equal', adjustable='box')
        ax.set_anchor('C')

    fig.legend(
        handles=sector_handles,
        title='Sector',
        loc='upper center',
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        borderpad=0.65,
        fontsize=11,
        title_fontsize=13,
    )
    fig.legend(
        handles=bubble_handles,
        labels=bubble_labels,
        title='Total cost (EUR)',
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=max(1, len(bubble_labels)),
        frameon=True,
        framealpha=0.95,
        borderpad=0.65,
        markerscale=1,
        handletextpad=0.8,
        labelspacing=0.7,
        fontsize=10,
        title_fontsize=12,
    )

    fig.suptitle('Failing Industries — Total Disruption Cost by Return Period', fontsize=20, y=0.998)
    if show:
        plt.show()

    for target_rp in plot_rps:
        pack = rp_plot_data.get(target_rp)
        n = 0 if pack is None else len(pack['plot_gdf'])
        print(f'RP{target_rp}: plotted {n:,} industries.')

    return {'fig': fig, 'axes': axes, 'rp_plot_data': rp_plot_data, 'all_costs': all_costs}


def prepare_mode_cost_data(disruption_df, industry_locs_gdf, location_id_col, target_rp=100, sector_code_map=None):
    """Prepare disruption data aggregated by mode, with exclusive priority assignment.
    
    Prevents double-counting using priority order: road > rail > iww > port > airport.
    
    Parameters
    ----------
    disruption_df : DataFrame
        Disruption data with failed_infr_type, origin_id, return_period columns.
    industry_locs_gdf : GeoDataFrame
        Industry locations with geometry.
    location_id_col : str
        ID column in industry_locs_gdf.
    target_rp : int
        Return period to filter on (default 100).
    sector_code_map : dict, optional
        Mapping from sector codes to names.
    
    Returns
    -------
    GeoDataFrame
        Industries with assigned mode, cost, and geometry. One row per industry.
    """
    src = disruption_df.copy()
    
    rp_col = next((c for c in ['return_period', 'rp', 'RP', 'returnperiod'] if c in src.columns), None)
    if rp_col is None:
        raise ValueError('No return-period column')
    
    src['_rp'] = pd.to_numeric(src[rp_col], errors='coerce')
    src = src[src['_rp'].eq(target_rp)].copy()
    
    if src.empty:
        raise ValueError(f'No rows for RP{target_rp}')
    
    src['_cost_eur'] = extract_cost_column(src)
    cost_numeric = pd.to_numeric(src['_cost_eur'], errors='coerce').fillna(0.0)
    if cost_numeric.max() <= 0 and 'value' in src.columns:
        fallback_value = pd.to_numeric(src['value'], errors='coerce').fillna(0.0)
        if fallback_value.max() > 0:
            src['_cost_eur'] = fallback_value
            cost_numeric = fallback_value
    src = src[src['_cost_eur'] > 0].copy()
    
    if src.empty:
        raise ValueError(
            f'No positive costs for RP{target_rp} '
            f"(cost columns present: {[c for c in ['disruption_cost_eur', 'total_cost_eur', 'rerouting_cost_eur_event', 'isolation_cost_eur_event', 'rerouting_cost_eur', 'isolation_cost_eur'] if c in disruption_df.columns]}, "
            f"value_present={'value' in disruption_df.columns}, max_cost={float(cost_numeric.max()):.3g})"
        )
    
    # Parse modes from failed_infr_type
    mode_canon = {
        'road': 'road', 'roads': 'road', 'road edge': 'road',
        'rail': 'rail', 'rails': 'rail', 'rail edge': 'rail',
        'iww': 'iww', 'iww port': 'iww', 'iww ports': 'iww', 'iww_port': 'iww', 'iww_ports': 'iww',
        'port': 'port', 'ports': 'port',
        'airport': 'airport', 'airports': 'airport',
    }
    
    src['_mode_token'] = (src['failed_infr_type'].fillna('').astype(str)
                          .str.lower().str.replace('_', ' ', regex=False).str.split('|'))
    src = src.explode('_mode_token')
    src['_mode_token'] = src['_mode_token'].astype(str).str.strip()
    src['transport_mode'] = src['_mode_token'].map(mode_canon)
    src = src[src['transport_mode'].fillna('').ne('')].copy()
    
    if src.empty:
        raise ValueError('No valid modes found')
    
    # Aggregate by (origin_id, mode)
    agg_spec = {'_cost_eur': ('_cost_eur', 'sum')}
    if 'origin_sector' in src.columns:
        agg_spec['origin_sector'] = (
            'origin_sector',
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'
        )
    else:
        src['origin_sector'] = 'unknown'
        agg_spec['origin_sector'] = ('origin_sector', 'first')
    
    mode_cost = (src.groupby(['origin_id', 'transport_mode'], as_index=False).agg(**agg_spec)
                 .rename(columns={'_cost_eur': 'total_cost_eur'}))
    
    if 'origin_sector' in mode_cost.columns:
        mode_cost['origin_sector'] = normalize_sector_names(mode_cost['origin_sector'], sector_code_map)
    
    # Exclusive assignment by priority
    mode_priority = ['road', 'rail', 'iww', 'port', 'airport']
    rank_map = {m: i for i, m in enumerate(mode_priority)}
    mode_cost['_rank'] = mode_cost['transport_mode'].map(rank_map).fillna(999).astype(int)
    mode_cost = mode_cost.sort_values(['origin_id', '_rank', 'total_cost_eur'], ascending=[True, True, False])
    mode_cost_exclusive = mode_cost.drop_duplicates(subset=['origin_id'], keep='first').copy()
    
    # Merge with geometry
    locs = industry_locs_gdf.copy()
    locs[location_id_col] = locs[location_id_col].astype(str).str.strip()
    
    result = mode_cost_exclusive.merge(
        locs[[location_id_col, 'geometry']],
        left_on='origin_id',
        right_on=location_id_col,
        how='left'
    )
    result = gpd.GeoDataFrame(
        result.drop(columns=[location_id_col, '_rank'], errors='ignore'),
        geometry='geometry',
        crs=locs.crs
    )
    result = result[result.geometry.notna()].copy()
    
    return result


def _pick_cost_series_with_name(df):
    """Return a numeric cost/loss series and the source column name if found."""
    candidate_cols = [
        'disruption_cost_eur',
        'total_cost_eur',
        'total_loss_eur',
        'loss_eur',
        'cost_eur',
        'cost',
    ]
    for col in candidate_cols:
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').fillna(0.0), col

    inferred_cols = [
        c for c in df.columns
        if ('cost' in str(c).lower() or 'loss' in str(c).lower())
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if inferred_cols:
        col = inferred_cols[0]
        return pd.to_numeric(df[col], errors='coerce').fillna(0.0), col

    return pd.Series(np.nan, index=df.index, dtype='float64'), None


def analyze_mode_overlap_and_cost_audit(
    disruption_df,
    mode_priority=None,
    mode_canon=None,
):
    """Compute exclusive mode assignment overlap outputs and per-mode cost audit.

    Parameters
    ----------
    disruption_df : DataFrame
        Input disruption table.
    mode_priority : list[str], optional
        Exclusive assignment priority from highest to lowest.
    mode_canon : dict, optional
        Mapping from raw failed infrastructure labels to canonical modes.

    Returns
    -------
    dict
        Dictionary containing overlap and audit outputs.
    """
    if disruption_df is None or not isinstance(disruption_df, pd.DataFrame) or disruption_df.empty:
        raise ValueError('`disruption_df` is missing/empty.')
    if 'origin_id' not in disruption_df.columns or 'failed_infr_type' not in disruption_df.columns:
        raise ValueError('Input table must contain `origin_id` and `failed_infr_type`.')

    if mode_priority is None:
        mode_priority = ['airport', 'port', 'iww', 'rail', 'road']

    if mode_canon is None:
        mode_canon = {
            'road': 'road', 'roads': 'road', 'road edge': 'road',
            'rail': 'rail', 'rails': 'rail', 'rail edge': 'rail',
            'iww': 'iww', 'iww port': 'iww', 'iww ports': 'iww', 'iww_port': 'iww', 'iww_ports': 'iww',
            'port': 'port', 'ports': 'port',
            'airport': 'airport', 'airports': 'airport',
        }

    work = disruption_df.copy()
    work['_cost'], cost_source_col = _pick_cost_series_with_name(work)

    if cost_source_col is not None:
        work = work[work['_cost'] > 0].copy()
    work['origin_id'] = work['origin_id'].astype(str).str.strip()

    work['_mode_tokens'] = (
        work['failed_infr_type'].fillna('').astype(str)
        .str.lower().str.replace('_', ' ', regex=False).str.split('|')
    )
    work = work.explode('_mode_tokens')
    work['_mode_tokens'] = work['_mode_tokens'].astype(str).str.strip()
    work['transport_mode'] = work['_mode_tokens'].map(mode_canon)
    work = work[work['transport_mode'].fillna('').ne('')].copy()

    if work.empty:
        assigned = pd.DataFrame(columns=['origin_id', 'transport_mode', 'total_cost_eur', '_rank'])
        mode_sets = {}
        overlap_df = pd.DataFrame(
            columns=['mode_a', 'mode_b', 'n_a', 'n_b', 'intersection', 'union', 'jaccard', 'overlap_vs_a_pct', 'overlap_vs_b_pct']
        )
        intersection_matrix = pd.DataFrame(dtype=int)
    else:
        by_mode = (
            work.groupby(['origin_id', 'transport_mode'], as_index=False)
            .agg(total_cost_eur=('_cost', 'sum'))
        )

        rank_map = {m: i for i, m in enumerate(mode_priority)}
        by_mode['_rank'] = by_mode['transport_mode'].map(rank_map).fillna(999).astype(int)

        assigned = by_mode.sort_values(['origin_id', '_rank', 'total_cost_eur'], ascending=[True, True, False])
        assigned = assigned.drop_duplicates(subset=['origin_id'], keep='first').copy()

        mode_sets = {
            mode: set(group['origin_id'].dropna().astype(str))
            for mode, group in assigned.groupby('transport_mode')
        }

        modes = sorted(mode_sets.keys())
        overlap_rows = []
        for i, mode_a in enumerate(modes):
            for mode_b in modes[i + 1:]:
                set_a, set_b = mode_sets[mode_a], mode_sets[mode_b]
                inter = set_a & set_b
                union = set_a | set_b
                overlap_rows.append({
                    'mode_a': mode_a,
                    'mode_b': mode_b,
                    'n_a': len(set_a),
                    'n_b': len(set_b),
                    'intersection': len(inter),
                    'union': len(union),
                    'jaccard': (len(inter) / len(union)) if union else 0.0,
                    'overlap_vs_a_pct': (100.0 * len(inter) / len(set_a)) if set_a else 0.0,
                    'overlap_vs_b_pct': (100.0 * len(inter) / len(set_b)) if set_b else 0.0,
                })

        overlap_df = pd.DataFrame(overlap_rows)
        if overlap_df.empty:
            overlap_df = pd.DataFrame(
                columns=['mode_a', 'mode_b', 'n_a', 'n_b', 'intersection', 'union', 'jaccard', 'overlap_vs_a_pct', 'overlap_vs_b_pct']
            )
        else:
            overlap_df = overlap_df.sort_values(['intersection', 'jaccard'], ascending=False)

        intersection_matrix = pd.DataFrame(0, index=modes, columns=modes, dtype=int)
        for mode_a in modes:
            for mode_b in modes:
                intersection_matrix.loc[mode_a, mode_b] = len(mode_sets[mode_a] & mode_sets[mode_b])

    audit_df = disruption_df.copy()
    audit_df['_mode'] = (
        audit_df['failed_infr_type'].fillna('').astype(str)
        .str.split('|').str[0].str.lower().str.replace('_', ' ', regex=False).str.strip()
        .map(mode_canon)
    )

    def _scol(df, name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors='coerce').fillna(default)
        return pd.Series(default, index=df.index, dtype='float64')

    audit_df['_value'] = _scol(audit_df, 'value', 0.0)
    audit_df['_days'] = _scol(audit_df, 'days_disrupted_computed', 0.0)
    audit_df['_kfactor'] = _scol(audit_df, 'kfactor', 0.0)
    audit_df['_ott'] = _scol(audit_df, 'original_travel_time_h', 0.0)
    audit_df['_status'] = audit_df.get('status', pd.Series('', index=audit_df.index)).astype(str).str.lower()
    audit_df['_cost'], audit_cost_source_col = _pick_cost_series_with_name(audit_df)
    audit_df['_vuln'] = audit_df['vuln_ratio_at_rp'] if 'vuln_ratio_at_rp' in audit_df.columns else pd.Series(np.nan, index=audit_df.index)
    audit_df['_vuln'] = pd.to_numeric(audit_df['_vuln'], errors='coerce')
    audit_df = audit_df[audit_df['_mode'].notna()].copy()

    audit_rows = []
    for mode, group in audit_df.groupby('_mode'):
        n = len(group)
        if n == 0:
            continue
        rerouted = group['_status'].eq('rerouted')
        isolated = ~rerouted

        audit_rows.append({
            'mode': mode,
            'rows': n,
            'cost_source_col': audit_cost_source_col if audit_cost_source_col is not None else '(none found)',
            'cost_pos_rows': int((group['_cost'] > 0).sum()) if audit_cost_source_col is not None else np.nan,
            'cost_pos_pct': 100.0 * (group['_cost'] > 0).mean() if audit_cost_source_col is not None else np.nan,
            'value_pos_pct': 100.0 * (group['_value'] > 0).mean(),
            'vuln_ratio_cov_pct': 100.0 * group['_vuln'].notna().mean(),
            'mean_vuln_ratio': float(group['_vuln'].mean()) if group['_vuln'].notna().any() else np.nan,
            'days_pos_pct': 100.0 * (group['_days'] > 0).mean(),
            'kfactor_pos_pct': 100.0 * (group['_kfactor'] > 0).mean(),
            'ott_pos_pct': 100.0 * (group['_ott'] > 0).mean(),
            'rerouted_pct': 100.0 * rerouted.mean(),
            'rerouted_with_ott_pos_pct': 100.0 * ((rerouted) & (group['_ott'] > 0)).mean(),
            'isolated_with_kfactor_pos_pct': 100.0 * ((isolated) & (group['_kfactor'] > 0)).mean(),
            'mean_value': float(group['_value'].mean()),
            'mean_days': float(group['_days'].mean()),
            'mean_kfactor': float(group['_kfactor'].mean()),
            'mean_ott': float(group['_ott'].mean()),
            'sum_cost': float(group['_cost'].sum()) if audit_cost_source_col is not None else np.nan,
        })

    audit_table = pd.DataFrame(audit_rows)
    if not audit_table.empty:
        audit_table = audit_table.sort_values('rows', ascending=False)

    rerouted_missing_ott_warning = False
    if 'original_travel_time_h' in disruption_df.columns and not audit_df.empty:
        rerouted_missing_ott_warning = (
            audit_df['_status'].eq('rerouted').any()
            and not (audit_df.loc[audit_df['_status'].eq('rerouted'), '_ott'] > 0).any()
        )

    return {
        'priority': list(mode_priority),
        'cost_source_col_overlap': cost_source_col,
        'assigned': assigned,
        'mode_sets': mode_sets,
        'overlap_df': overlap_df,
        'intersection_matrix': intersection_matrix,
        'audit_df': audit_table,
        'audit_cost_source_col': audit_cost_source_col,
        'has_original_travel_time_h': 'original_travel_time_h' in disruption_df.columns,
        'rerouted_missing_ott_warning': rerouted_missing_ott_warning,
        'audit_work_df': audit_df,
    }


def export_disrupted_industry_costs(
    disruption_df,
    industry_locs_gdf,
    export_path,
    days_col='days_disrupted_computed',
):
    """Export disrupted origin industries with cost fields as a shapefile.

    Returns
    -------
    dict
        Contains keys: ``ok``, ``gdf``, ``cost_source_col``, ``export_path``, ``warning``.
    """
    out = {
        'ok': False,
        'gdf': gpd.GeoDataFrame(),
        'cost_source_col': None,
        'export_path': Path(export_path),
        'warning': None,
    }

    if disruption_df is None or not isinstance(disruption_df, pd.DataFrame) or disruption_df.empty:
        out['warning'] = '`disruption_df` is missing or empty; export skipped.'
        return out

    if not isinstance(industry_locs_gdf, gpd.GeoDataFrame) or industry_locs_gdf.empty:
        out['warning'] = '`industry_locs` is missing or empty; export skipped.'
        return out

    disruption_export = disruption_df.copy()
    disruption_export['_cost_export'], cost_col = _pick_cost_series_with_name(disruption_export)
    out['cost_source_col'] = cost_col
    disruption_export['origin_id'] = disruption_export.get('origin_id', pd.Series('', index=disruption_export.index)).astype(str).str.strip()

    industry_id_col = next((
        col for col in ['id', 'industry_id', 'origin_id', 'ind_id', 'osmid']
        if col in industry_locs_gdf.columns
    ), None)

    if industry_id_col is None:
        out['warning'] = 'Could not find an industry ID column in `industry_locs`; export skipped.'
        return out

    industry_geom = industry_locs_gdf.copy()
    industry_geom[industry_id_col] = industry_geom[industry_id_col].astype(str).str.strip()
    if industry_geom.crs is None:
        industry_geom = industry_geom.set_crs('EPSG:4326', allow_override=True)

    merged = disruption_export.merge(
        industry_geom[[industry_id_col, 'geometry']].rename(columns={industry_id_col: 'origin_id'}),
        on='origin_id',
        how='left',
    )
    merged = merged[merged['geometry'].notna()].copy()
    if merged.empty:
        out['warning'] = 'No disruption rows matched origin industry geometries; export skipped.'
        return out

    sector_col = next((
        col for col in ['origin_sector', 'sector', 'sector_name']
        if col in merged.columns
    ), None)

    export_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs=industry_geom.crs)
    export_gdf['sector'] = export_gdf[sector_col].astype(str).str.strip() if sector_col else ''
    export_gdf['infr_type'] = export_gdf.get('failed_infr_type', pd.Series('', index=export_gdf.index)).astype(str).str.strip()
    export_gdf['infr_id'] = export_gdf.get('failed_infr_id', pd.Series('', index=export_gdf.index)).astype(str).str.strip()
    export_gdf['cost_eur'] = pd.to_numeric(export_gdf['_cost_export'], errors='coerce').fillna(0.0)
    export_gdf['status'] = export_gdf.get('status', pd.Series('', index=export_gdf.index)).astype(str).str.strip()
    export_gdf['days_disr'] = pd.to_numeric(export_gdf.get(days_col, pd.Series(0.0, index=export_gdf.index)), errors='coerce').fillna(0.0)

    disrupted_industry_costs_gdf = export_gdf[[
        'origin_id',
        'sector',
        'infr_type',
        'infr_id',
        'cost_eur',
        'status',
        'days_disr',
        'geometry',
    ]].copy()

    out_path = Path(export_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    disrupted_industry_costs_gdf.to_file(out_path)

    out['ok'] = True
    out['gdf'] = disrupted_industry_costs_gdf
    out['export_path'] = out_path
    return out


def _ensure_gdf_crs(gdf, target_crs="EPSG:4326"):
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs(target_crs, allow_override=True)
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf


def _to_4326(gdf):
    if gdf is None or (isinstance(gdf, gpd.GeoDataFrame) and gdf.empty):
        return gdf
    if gdf.crs is None or str(gdf.crs) == "EPSG:4326":
        return gdf
    try:
        return gdf.to_crs("EPSG:4326")
    except Exception:
        return gdf


def _filter_has_L_in_corridor(gdf):
    """Keep rows where CORRIDOR(S) includes token L (alone or with other letters)."""
    if gdf is None or (isinstance(gdf, gpd.GeoDataFrame) and gdf.empty):
        return gdf

    corridor_cols = [c for c in ("CORRIDOR", "CORRIDORS", "corridor", "corridors") if c in gdf.columns]
    if not corridor_cols:
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)

    # Accept forms like: L, A|L, L|B, A;L, A,L
    token_pat = r"(^|[|,;\s])L($|[|,;\s])"

    mask = pd.Series(False, index=gdf.index)
    for col in corridor_cols:
        ser = (
            gdf[col]
            .astype(str)
            .str.upper()
            .str.replace("/", "|", regex=False)
            .str.replace("-", "|", regex=False)
        )
        mask = mask | ser.str.contains(token_pat, regex=True, na=False)

    return gdf[mask].copy()

def _zoom_to_selected_nuts2(ax, gdf=None):
    if isinstance(nuts2, gpd.GeoDataFrame) and not nuts2.empty:
        try:
            target_crs = getattr(gdf, "crs", None)
            if target_crs is None:
                target_crs = "EPSG:4326"
            nuts2_plot = nuts2
            if nuts2_plot.crs is None:
                nuts2_plot = nuts2_plot.set_crs(target_crs, allow_override=True)
            elif str(nuts2_plot.crs) != str(target_crs):
                nuts2_plot = nuts2_plot.to_crs(target_crs)

            if not nuts2_plot.empty:
                nuts2_plot.plot(
                    ax=ax,
                    facecolor="#f3f3f3",
                    edgecolor="#8b8b8b",
                    linewidth=0.6,
                    alpha=0.8,
                    zorder=0,
                )
                nuts2_plot.boundary.plot(ax=ax, color="#8b8b8b", linewidth=0.4, alpha=0.9, zorder=1)

                bb = nuts2_plot.total_bounds
                pad_x = (bb[2] - bb[0]) * 0.10
                pad_y = (bb[3] - bb[1]) * 0.10
                ax.set_xlim(bb[0] - pad_x, bb[2] + pad_x)
                ax.set_ylim(bb[1] - pad_y, bb[3] + pad_y)
        except Exception:
            pass

def plot_mode_industry_disruption_panels(
    disruption_df,
    industry_locs_gdf,
    location_id_col,
    target_rp=100,
    nuts2=None,
    sector_code_map=None,
    corridor_label=None,
    bubble_labels=None,
    bubble_markers=None,
    sector_cmap=None,
    mode_order=None,
    divide_iww_by=1000.0,
    zoom_reference_mode='road',
    figsize_per_col=5.5,
    figsize_per_row=4.5,
    show=True,
):
    """Plot mode-specific failing-industry bubble maps for a target RP.

    Uses exclusive mode assignment from ``prepare_mode_cost_data`` and applies a
    shared zoom across subplots, optionally anchored to a reference mode
    (e.g., ``road``) when present.

    Returns
    -------
    dict
        Contains figure, axes, available modes, mode totals, and plotting data.
    """
    import matplotlib.pyplot as plt

    if sector_cmap is None:
        sector_cmap = {
            'chemicals': '#1f77b4',
            'commercial': '#ff7f0e',
            'food and beverage': '#d62728',
            'intensive livestock': '#9467bd',
            'metals': '#8c564b',
            'minerals': '#e377c2',
            'paper and wood': '#7f7f7f',
        }

    bubble_source_by_mode = prepare_mode_cost_data(
        disruption_df,
        industry_locs_gdf,
        location_id_col,
        target_rp,
        sector_code_map=sector_code_map,
    )

    if divide_iww_by not in (None, 0):
        iww_mask = bubble_source_by_mode['transport_mode'].eq('iww')
        bubble_source_by_mode.loc[iww_mask, 'total_cost_eur'] = (
            pd.to_numeric(bubble_source_by_mode.loc[iww_mask, 'total_cost_eur'], errors='coerce').fillna(0.0) / float(divide_iww_by)
        )

    nuts2_plot = nuts2.copy() if isinstance(nuts2, gpd.GeoDataFrame) else gpd.GeoDataFrame()
    if isinstance(nuts2_plot, gpd.GeoDataFrame) and not nuts2_plot.empty and nuts2_plot.crs != bubble_source_by_mode.crs:
        try:
            nuts2_plot = nuts2_plot.to_crs(bubble_source_by_mode.crs)
        except Exception:
            pass

    available_modes = sorted([m for m in bubble_source_by_mode['transport_mode'].dropna().unique() if m])
    if mode_order:
        rank = {m: i for i, m in enumerate(mode_order)}
        available_modes = sorted(available_modes, key=lambda m: (rank.get(m, 999), m))

    if not available_modes:
        raise ValueError(f'No mode-specific data available for RP{target_rp}')

    mode_totals = (
        bubble_source_by_mode.groupby('transport_mode')['total_cost_eur']
        .sum()
        .sort_values(ascending=False)
    )

    n_modes = len(available_modes)
    n_cols = min(3, n_modes)
    n_rows = (n_modes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_col * n_cols, figsize_per_row * n_rows))
    axes = ([axes] if n_modes == 1 else axes.flatten() if n_modes > 1 else [axes])

    all_costs = pd.to_numeric(bubble_source_by_mode['total_cost_eur'], errors='coerce').fillna(0.0).clip(lower=0.0)
    log_cost_max = float(np.log10(all_costs + 1.0).max()) if len(all_costs) else 1.0

    def _scale_bubble(value):
        scaled = np.log10(max(value, 0.0) + 1.0)
        return 6.0 + (scaled / max(log_cost_max, 1e-9)) * 174.0

    global_plot_data = bubble_source_by_mode[bubble_source_by_mode.geometry.notna()].copy()
    ref_data = global_plot_data
    if zoom_reference_mode in set(global_plot_data['transport_mode'].dropna().astype(str)):
        _candidate = global_plot_data[global_plot_data['transport_mode'].eq(zoom_reference_mode)].copy()
        if not _candidate.empty:
            ref_data = _candidate

    shared_bounds = None
    try:
        minx, miny, maxx, maxy = ref_data.total_bounds
        if np.isfinite([minx, miny, maxx, maxy]).all() and maxx > minx and maxy > miny:
            padx = (maxx - minx)*0.1
            pady = (maxy - miny)*0.7
            shared_bounds = (minx - padx, maxx + padx, miny - pady, maxy + pady)
    except Exception:
        shared_bounds = None

    for idx, mode in enumerate(available_modes):
        ax = axes[idx]
        mode_data = bubble_source_by_mode[bubble_source_by_mode['transport_mode'].eq(mode)].copy()
        mode_data = mode_data[mode_data.geometry.notna()].copy()

        if mode_data.empty:
            ax.text(0.5, 0.5, f'No data for {mode}', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        if isinstance(nuts2_plot, gpd.GeoDataFrame) and not nuts2_plot.empty:
            nuts2_plot.plot(ax=ax, color='#f7f7f7', edgecolor='black', linewidth=0.5, zorder=0)

        for sector, sector_df in mode_data.groupby('origin_sector', dropna=False):
            sector_df.plot(
                ax=ax,
                markersize=sector_df['total_cost_eur'].map(_scale_bubble),
                color=sector_cmap.get(str(sector).strip().lower(), 'tab:blue'),
                alpha=0.72,
                edgecolor='white',
                linewidth=0.5,
                zorder=3,
            )

        if isinstance(nuts2_plot, gpd.GeoDataFrame) and not nuts2_plot.empty:
            nuts2_plot.boundary.plot(ax=ax, color='black', linewidth=0.5, zorder=1)

        ax.set_axis_off()
        title = f'Failing Industries — {str(mode).upper()} (RP{target_rp})'
        if corridor_label:
            title += f' ({corridor_label})'
        ax.set_title(title, fontsize=12, pad=10)

        if idx == 0:
            sector_handles = create_sector_legend(sector_cmap)
            ax.legend(
                handles=sector_handles,
                title='Sector',
                loc='upper left',
                bbox_to_anchor=(0.015, 0.985),
                ncol=2,
                frameon=True,
                framealpha=0.95,
                fancybox=True,
                fontsize=8,
                title_fontsize=9,
            )

        if idx == len(available_modes) - 1:
            bubble_handles, legend_labels = create_industry_bubble_legend(
                bubble_labels=bubble_labels,
                bubble_markers=bubble_markers,
            )
            ax.legend(
                handles=bubble_handles,
                labels=legend_labels,
                title='Total cost (EUR)',
                loc='lower right',
                bbox_to_anchor=(0.985, 0.025),
                frameon=True,
                framealpha=0.95,
                fancybox=True,
                fontsize=8,
                title_fontsize=9,
            )

        if shared_bounds is not None:
            ax.set_xlim(shared_bounds[0], shared_bounds[1])
            ax.set_ylim(shared_bounds[2], shared_bounds[3])
        else:
            try:
                bounds = get_robust_bounds(ref_data if not ref_data.empty else mode_data)
                apply_bounds_to_ax(ax, bounds)
            except Exception:
                pass

        ax.set_aspect('equal', adjustable='box')
        ax.set_anchor('C')

    for idx in range(len(available_modes), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if show:
        plt.show()

    return {
        'fig': fig,
        'axes': axes,
        'available_modes': available_modes,
        'mode_totals': mode_totals,
        'bubble_source_by_mode': bubble_source_by_mode,
        'shared_bounds': shared_bounds,
    }


# ── EAL and Cost Calculations ─────────────────────────────────────────────────

def extract_loss_columns(disruption_df):
    """Extract and compute indirect and total loss columns from disruption data.
    
    Handles multiple fallback options for cost columns.
    
    Parameters
    ----------
    disruption_df : DataFrame
        Disruption data with various cost columns.
    
    Returns
    -------
    DataFrame
        Input dataframe with 'indirect_loss_eur' and 'total_loss_eur' columns added.
    """
    df = disruption_df.copy()
    
    # Indirect losses
    if 'disruption_cost_eur' in df.columns:
        df['indirect_loss_eur'] = pd.to_numeric(df['disruption_cost_eur'], errors='coerce').fillna(0.0)
    else:
        fallback_indirect = [c for c in ['rerouting_cost_eur_event', 'isolation_cost_eur_event', 
                                          'rerouting_cost_eur', 'isolation_cost_eur'] if c in df.columns]
        if fallback_indirect:
            df['indirect_loss_eur'] = df[fallback_indirect].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1)
        else:
            raise ValueError("No indirect-cost columns found")
    
    # Total losses
    if 'total_cost_eur' in df.columns:
        df['total_loss_eur'] = pd.to_numeric(df['total_cost_eur'], errors='coerce').fillna(0.0)
    else:
        total_components = [c for c in ['disruption_cost_eur', 'rerouting_cost_eur_event', 'isolation_cost_eur_event',
                                         'direct_reconstruction_cost_eur', 'rerouting_cost_eur', 'isolation_cost_eur',
                                         'direct_damage_cost_eur'] if c in df.columns]
        if total_components:
            df['total_loss_eur'] = df[total_components].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1)
        else:
            df['total_loss_eur'] = df['indirect_loss_eur']
    
    return df


def normalize_infra_types(disruption_df, rp_col_name=None):
    """Normalize infrastructure types and IDs from disruption data.
    
    Parameters
    ----------
    disruption_df : DataFrame
        Disruption data with failed_infr_type, failed_infra_id, and return_period columns.
    rp_col_name : str, optional
        Name of return period column (auto-detected if None).
    
    Returns
    -------
    DataFrame
        Data with normalized 'infra_type', 'infra_id', and 'rp_num' columns.
    """
    df = disruption_df.copy()
    
    if rp_col_name is None:
        rp_col_name = next((c for c in ['return_period', 'rp', 'RP', 'returnperiod'] if c in df.columns), None)
    
    if rp_col_name is None:
        raise ValueError("No return-period column found")
    
    mode_canon = {
        'rail': 'rail edge', 'rails': 'rail edge', 'rail edge': 'rail edge',
        'road': 'road edge', 'roads': 'road edge', 'road edge': 'road edge',
        'iww': 'iww port', 'iww port': 'iww port', 'iww ports': 'iww port', 'iww_port': 'iww port', 'iww_ports': 'iww port',
        'airport': 'airport', 'airports': 'airport',
        'port': 'port', 'ports': 'port',
    }
    
    df['_infra_type_raw'] = (df['failed_infr_type'].fillna('').astype(str).str.split('|').str[0]
                             .str.lower().str.replace('_', ' ', regex=False).str.strip())
    df['infra_type'] = df['_infra_type_raw'].map(mode_canon)
    df['infra_id'] = df['failed_infr_id'].fillna('').astype(str).str.split('|').str[0].str.strip()
    df['rp_num'] = pd.to_numeric(df[rp_col_name], errors='coerce')
    
    return df


def compute_eal_from_rp_curve(df_by_rp, value_col):
    """Compute Expected Annual Loss (EAL) from RP curve via trapezoidal integration.
    
    Parameters
    ----------
    df_by_rp : DataFrame
        Data grouped by return period with 'rp_num' and value column.
    value_col : str
        Column name containing loss values.
    
    Returns
    -------
    float
        Computed EAL.
    """
    tmp = df_by_rp[['rp_num', value_col]].copy()
    tmp = tmp.groupby('rp_num', as_index=False)[value_col].sum()
    tmp = tmp[tmp['rp_num'] > 0].copy()
    
    if tmp.empty:
        return 0.0
    
    tmp['p'] = 1.0 / tmp['rp_num'].astype(float)
    tmp = tmp.sort_values('p')
    
    x = tmp['p'].to_numpy()
    y = tmp[value_col].to_numpy()
    
    x_aug = np.concatenate(([0.0], x, [1.0]))
    y_aug = np.concatenate(([0.0], y, [y[-1]]))
    
    return float(np.trapz(y_aug, x_aug))


def compute_eal_by_infrastructure(disruption_df, group_cols=['infra_type', 'infra_id']):
    """Compute EAL for each infrastructure asset.
    
    Parameters
    ----------
    disruption_df : DataFrame
        Normalized disruption data with 'rp_num', 'indirect_loss_eur', 'total_loss_eur'.
    group_cols : list
        Columns to group by (default: infra_type, infra_id).
    
    Returns
    -------
    DataFrame
        EAL by infrastructure asset.
    """
    eal_rows = []
    
    for key, group in disruption_df.groupby(group_cols):
        key_dict = dict(zip(group_cols, key if isinstance(key, tuple) else [key]))
        key_dict['EAIL_indirect_eur_per_year'] = compute_eal_from_rp_curve(group, 'indirect_loss_eur')
        key_dict['EAL_total_eur_per_year'] = compute_eal_from_rp_curve(group, 'total_loss_eur')
        eal_rows.append(key_dict)
    
    return pd.DataFrame(eal_rows)


def _norm_infra_key_by_type(value, infra_type):
    """Normalize infrastructure IDs with mode-specific rules for joins."""
    if infra_type in ('road edge', 'rail edge'):
        return _canonical_transport_id(value)
    return _norm_key(value)


def build_present_loss_curve(disruption_sectors, plot_order=None):
    """Build present-day loss curves by infrastructure type/id/return period."""
    plot_order = ['rail edge', 'road edge', 'iww port', 'airport', 'port'] if plot_order is None else list(plot_order)

    if disruption_sectors is None or getattr(disruption_sectors, 'empty', True):
        raise ValueError('`disruption_sectors` is missing or empty. Run the enrichment cell first.')

    df = extract_loss_columns(disruption_sectors.copy())
    df = normalize_infra_types(df)
    df = df[df['infra_type'].isin(plot_order)].copy()
    df = df[df['infra_id'].ne('') & df['rp_num'].notna() & (df['rp_num'] > 0)].copy()

    if df.empty:
        raise ValueError('No valid infrastructure records after filtering.')

    present_curve = (
        df.groupby(['infra_type', 'infra_id', 'rp_num'], as_index=False)
        .agg(loss_eur=('total_loss_eur', 'sum'))
    )
    return present_curve


def _supplement_geo_from_hazard(
    geo_layer,
    infra_type_label,
    hazard_file=None,
    hazard_df=None,
    candidate_cols=None,
):
    """Supplement a geometry layer with missing features from a hazard parquet file."""
    from pathlib import Path
    from shapely import wkb
    
    candidate_cols = candidate_cols or ['id', 'edge_id', 'link_id', 'fid']
    
    # If geo_layer is None or empty, return empty GeoDataFrame
    if geo_layer is None or getattr(geo_layer, 'empty', True):
        return gpd.GeoDataFrame(columns=['id', 'geometry'], crs='EPSG:4326')
    
    # Load hazard data if not provided
    if hazard_df is None:
        if hazard_file is None:
            return geo_layer.copy()
        hazard_file = Path(hazard_file)
        if not hazard_file.exists():
            return geo_layer.copy()
        hazard_df = pd.read_parquet(hazard_file)
    
    # Find ID column in hazard_df
    id_col = next((col for col in candidate_cols if col in hazard_df.columns), None)
    if id_col is None or 'geometry' not in hazard_df.columns:
        return geo_layer.copy()
    
    # Normalize IDs in geo_layer
    geo_id_col = geo_layer.columns[0]
    geo_ids_norm = geo_layer[geo_id_col].astype(str).apply(
        lambda v: _norm_infra_key_by_type(v, infra_type_label)
    )
    geo_ids_set = set(geo_ids_norm.dropna())
    
    # Normalize IDs in hazard
    haz_ids_norm = hazard_df[id_col].astype(str).apply(
        lambda v: _norm_infra_key_by_type(v, infra_type_label)
    )
    haz_ids_set = set(haz_ids_norm.dropna())
    
    # Find missing IDs
    missing_ids = haz_ids_set - geo_ids_set
    if not missing_ids:
        return geo_layer.copy()
    
    # Extract supplement rows from hazard
    mask = haz_ids_norm.isin(missing_ids)
    supplement_df = hazard_df.loc[mask, [id_col, 'geometry']].copy()
    
    # Rename ID column to match geo_layer
    supplement_df = supplement_df.rename(columns={id_col: geo_id_col})
    
    # Convert WKB geometry if needed
    if not supplement_df.empty and isinstance(supplement_df['geometry'].iloc[0], bytes):
        supplement_df['geometry'] = supplement_df['geometry'].apply(wkb.loads)
    
    # Create GeoDataFrame for supplement
    supplement_gdf = gpd.GeoDataFrame(
        supplement_df.reset_index(drop=True),
        geometry='geometry',
        crs=geo_layer.crs or 'EPSG:4326'
    )
    
    # Concatenate with original
    result = pd.concat([geo_layer, supplement_gdf], ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=geo_layer.crs)
    
    return result


def build_impacted_infra_geometries_from_curve(
    present_curve,
    road_edges=None,
    rail_edges=None,
    iww_nodes=None,
    ports=None,
    airports=None,
    plot_order=None,
):
    """Build impacted infrastructure geometries that match the present-loss curve IDs."""
    plot_order = ['rail edge', 'road edge', 'iww port', 'airport', 'port'] if plot_order is None else list(plot_order)

    if present_curve is None or getattr(present_curve, 'empty', True):
        raise ValueError('`present_curve` is missing or empty.')

    keys_by_type = {
        t: set(
            present_curve.loc[present_curve['infra_type'].eq(t), 'infra_id']
            .map(lambda v, _t=t: _norm_infra_key_by_type(v, _t))
            .dropna()
        )
        for t in plot_order
    }

    road_geo = _build_geo_keys(road_edges, 'road edge', ('edge_id', 'id', 'link_id', 'fid'))
    rail_geo = _build_geo_keys(rail_edges, 'rail edge', ('edge_id', 'id', 'link_id', 'fid'))
    iww_geo = _build_geo_keys(
        iww_nodes,
        'iww port',
        ('id', 'node_id', 'port_id', 'fid'),
        extra_filter=lambda d: d['feature'].astype(str).str.lower().eq('port') if 'feature' in d.columns else pd.Series(True, index=d.index),
    )
    port_geo = _build_geo_keys(ports, 'port', ('port_code', 'id', 'port_id', 'node_id', 'fid'))
    airport_geo = _build_geo_keys(airports, 'airport', ('id', 'airport_id', 'icao', 'node_id', 'fid'))

    if not road_geo.empty:
        road_geo = road_geo[road_geo['match_key'].isin(keys_by_type.get('road edge', set()))].copy()

    if not rail_geo.empty:
        rail_geo = rail_geo[rail_geo['match_key'].isin(keys_by_type.get('rail edge', set()))].copy()

    if not iww_geo.empty:
        iww_geo = iww_geo[iww_geo['match_key'].isin(keys_by_type.get('iww port', set()))].copy()

    if not port_geo.empty:
        port_geo = port_geo[port_geo['match_key'].isin(keys_by_type.get('port', set()))].copy()

    if not airport_geo.empty:
        airport_geo = airport_geo[airport_geo['match_key'].isin(keys_by_type.get('airport', set()))].copy()

    infra_geo = pd.concat([road_geo, rail_geo, iww_geo, port_geo, airport_geo], ignore_index=True)
    infra_geo = gpd.GeoDataFrame(
        infra_geo,
        geometry='geometry',
        crs=next(
            (
                g.crs
                for g in [road_geo, rail_geo, iww_geo, port_geo, airport_geo]
                if isinstance(g, gpd.GeoDataFrame) and not g.empty and g.crs is not None
            ),
            'EPSG:4326',
        ),
    )

    if infra_geo.empty:
        raise ValueError('No impacted infrastructure geometries could be built.')

    infra_geo['infra_id'] = infra_geo['match_key'].astype(str)
    infra_geo['infra_id_norm'] = infra_geo['infra_id']
    return infra_geo


def compute_present_eal_outputs(
    disruption_sectors,
    road_edges=None,
    rail_edges=None,
    iww_nodes=None,
    ports=None,
    airports=None,
    nuts2=None,
    plot_order=None,
    show=True,
    make_plots=True,
):
    """Compute present EAL outputs and impacted infrastructure geometries."""
    plot_order = ['rail edge', 'road edge', 'iww port', 'airport', 'port'] if plot_order is None else list(plot_order)

    df = extract_loss_columns(disruption_sectors.copy())
    df = normalize_infra_types(df, rp_col_name=None)
    df = df[df['infra_type'].isin(plot_order)].copy()
    df = df[df['infra_id'].ne('') & df['rp_num'].notna() & (df['rp_num'] > 0)].copy()

    if df.empty:
        raise ValueError('No valid infrastructure records after filtering.')

    eal_by_infra = compute_eal_by_infrastructure(df, group_cols=['infra_type', 'infra_id'])
    eal_by_type = (
        eal_by_infra.groupby('infra_type', as_index=False)
        .agg(
            EAIL_indirect_eur_per_year=('EAIL_indirect_eur_per_year', 'sum'),
            EAL_total_eur_per_year=('EAL_total_eur_per_year', 'sum'),
        )
    )
    eal_by_type['infra_type'] = pd.Categorical(eal_by_type['infra_type'], categories=plot_order, ordered=True)
    eal_by_type = eal_by_type.sort_values('infra_type')

    eal_indirect_by_infra = eal_by_infra[['infra_type', 'infra_id', 'EAIL_indirect_eur_per_year']].copy()
    eal_total_by_infra = eal_by_infra[['infra_type', 'infra_id', 'EAL_total_eur_per_year']].copy()

    present_curve = build_present_loss_curve(disruption_sectors, plot_order=plot_order)
    infra_geo = build_impacted_infra_geometries_from_curve(
        present_curve,
        road_edges=road_edges,
        rail_edges=rail_edges,
        iww_nodes=iww_nodes,
        ports=ports,
        airports=airports,
        plot_order=plot_order,
    )

    rendered = []
    if make_plots:
        rendered = plot_present_eal_bubble_maps(
            infra_geo=infra_geo,
            eal_by_infra=eal_by_infra,
            nuts2=nuts2,
            plot_order=plot_order,
            show=show,
        )

    return {
        'plot_order': plot_order,
        'eal_by_infra': eal_by_infra,
        'eal_by_type': eal_by_type,
        'eal_indirect_by_infra': eal_indirect_by_infra,
        'eal_total_by_infra': eal_total_by_infra,
        'present_curve': present_curve,
        'infra_geo': infra_geo,
        'rendered_plots': rendered,
    }


def plot_present_eal_bubble_maps(
    infra_geo,
    eal_by_infra,
    nuts2=None,
    plot_order=None,
    figure_title='Present EAL by Infrastructure Type',
    font_scale=5,
    figsize=(14, 12),
    show=True,
):
    """
    Plot present-day EAL bubble maps organized by infrastructure type.
    
    Creates one figure per infrastructure type, each with a single map showing
    EAL distribution. All figures use the same shared logarithmic color scale
    for cross-infrastructure comparison.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if infra_geo is None or getattr(infra_geo, 'empty', True):
        raise ValueError('`infra_geo` is missing/empty.')
    if eal_by_infra is None or getattr(eal_by_infra, 'empty', True):
        raise ValueError('`eal_by_infra` is missing/empty.')

    plot_order = ['rail edge', 'road edge', 'iww port', 'airport', 'port'] if plot_order is None else list(plot_order)

    eal_geo = infra_geo.merge(
        eal_by_infra[['infra_type', 'infra_id', 'EAL_total_eur_per_year']],
        on=['infra_type', 'infra_id'],
        how='inner',
    )
    eal_geo = gpd.GeoDataFrame(eal_geo, geometry='geometry', crs=infra_geo.crs)
    eal_geo['EAIL_present'] = pd.to_numeric(eal_geo['EAL_total_eur_per_year'], errors='coerce').fillna(0.0)
    eal_geo.loc[eal_geo['infra_type'] == 'iww port', 'EAIL_present'] /= 1000.0

    title_fs = int(round(13 * font_scale))
    ax_title_fs = int(round(11 * font_scale))
    note_fs = int(round(9 * font_scale))
    cbar_fs = int(round(9 * font_scale))
    tick_fs = int(round(8 * font_scale))

    prepared = {}
    all_pos = []

    for itype in plot_order:
        sub = eal_geo[eal_geo['infra_type'] == itype].copy()
        if sub.empty:
            print(f'[SKIP] {itype} – no data')
            continue

        sub = _infer_and_fix_crs(gpd.GeoDataFrame(sub, geometry='geometry', crs=sub.crs))
        vals = sub['EAIL_present']
        pos = vals[vals > 0]
        plot_crs = sub.crs

        nuts2_bg = None
        bounds = None
        if isinstance(nuts2, gpd.GeoDataFrame) and not nuts2.empty:
            try:
                nuts2_bg = nuts2.copy()
                if nuts2_bg.crs is None:
                    nuts2_bg = nuts2_bg.set_crs('EPSG:4326', allow_override=True)
                if plot_crs is not None and nuts2_bg.crs != plot_crs:
                    nuts2_bg = nuts2_bg.to_crs(plot_crs)
                nb = nuts2_bg.total_bounds
                if np.isfinite(nb).all() and nb[2] > nb[0] and nb[3] > nb[1]:
                    px = (nb[2] - nb[0]) * 0.25
                    py = (nb[3] - nb[1]) * 0.25
                    bounds = (nb[0] - px, nb[1] - py, nb[2] + px, nb[3] + py)
            except Exception:
                pass

        bounds = bounds or get_robust_bounds(sub)
        prepared[itype] = {'sub': sub, 'pos': pos, 'nuts2_bg': nuts2_bg, 'bounds': bounds}
        if not pos.empty:
            all_pos.append(pos.values)

    if not prepared:
        raise ValueError('No data available for visualization.')

    if all_pos:
        all_pos_vals = np.concatenate(all_pos)
        vmin = float(np.nanpercentile(all_pos_vals, 5))
        vmax = float(np.nanpercentile(all_pos_vals, 95))
        if not (np.isfinite(vmin) and np.isfinite(vmax) and 0 < vmin < vmax):
            vmin, vmax = float(np.nanmin(all_pos_vals)), float(np.nanmax(all_pos_vals))
        norm = mpl.colors.LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, vmin * 1.01))
    else:
        norm = mpl.colors.LogNorm(vmin=1e-3, vmax=1.0)

    rendered = []
    
    # Create one figure per infrastructure type
    for itype in plot_order:
        if itype not in prepared:
            print(f'[SKIP FIGURE] {itype} – no data')
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        fig.suptitle(f'{figure_title}: {itype.title()}', fontsize=title_fs, fontweight='bold')
        
        pack = prepared[itype]
        sub = pack['sub']
        pos = pack['pos']
        nuts2_bg = pack['nuts2_bg']
        bounds = pack['bounds']
        
        if nuts2_bg is not None and not nuts2_bg.empty:
            nuts2_bg.plot(ax=ax, color='#f0f0f0', edgecolor='black', linewidth=0.6, zorder=0, aspect='equal')
        
        if pos.empty:
            sub.plot(ax=ax, color='#bdbdbd', linewidth=2.2, markersize=50, zorder=2, aspect='equal')
            ax.text(
                0.5,
                0.5,
                'All EAL = 0',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=note_fs,
                color='0.35',
            )
        else:
            bubble_gdf = sub.copy()
            is_pt = bubble_gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])
            if (~is_pt).any():
                bubble_gdf.loc[~is_pt, 'geometry'] = bubble_gdf.loc[~is_pt].geometry.representative_point()
            
            # Use fixed bubble size for all points - only color scales the EAIL value
            fixed_size = 100
            bubble_gdf.plot(ax=ax, color='white', markersize=fixed_size * 1.45, linewidth=0, zorder=3, aspect='equal')
            bubble_gdf.plot(
                ax=ax,
                column='EAIL_present',
                cmap='RdYlGn_r',
                norm=norm,
                markersize=fixed_size,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.95,
                legend=False,
                zorder=4,
                aspect='equal',
            )
        
        if bounds is not None:
            apply_bounds_to_ax(ax, bounds)
        
        ax.set_axis_off()
        
        # Add colorbar to this figure
        sm = mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn_r')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, aspect=50)
        cbar.set_label('EAL (EUR / year)', fontsize=cbar_fs)
        cbar.ax.tick_params(labelsize=tick_fs)
        
        if show:
            plt.show()
        
        rendered.append({
            'infra_type': itype,
            'fig': fig,
            'ax': ax,
            'norm': norm,
        })
    
    return {
        'rendered': rendered,
        'eal_geo': eal_geo,
        'prepared': prepared,
        'norm': norm,
    }


def _eal_from_curve(loss_by_rp_df, rp_col, loss_col):
    """Compute EAL by integrating a loss curve over exceedance probability."""
    tmp = loss_by_rp_df[[rp_col, loss_col]].copy()
    tmp[rp_col] = pd.to_numeric(tmp[rp_col], errors='coerce')
    tmp[loss_col] = pd.to_numeric(tmp[loss_col], errors='coerce').fillna(0.0)
    tmp = tmp[tmp[rp_col] > 0]
    if tmp.empty:
        return 0.0
    tmp = tmp.groupby(rp_col, as_index=False)[loss_col].sum()
    tmp['p'] = 1.0 / tmp[rp_col].astype(float)
    tmp = tmp.sort_values('p')
    x = tmp['p'].to_numpy()
    y = tmp[loss_col].to_numpy()
    x_aug = np.concatenate(([0.0], x, [1.0]))
    y_aug = np.concatenate(([0.0], y, [y[-1]]))
    return float(np.trapz(y_aug, x_aug))


def plot_future_present_eail_ratio_maps(
    ratio_geo,
    nuts2=None,
    plot_order=None,
    scenarios=None,
    show=True,
):
    """
    Plot future/present EAIL ratio bubble maps organized by infrastructure type.
    
    Creates one figure per infrastructure type, with a 2x2 subplot grid showing
    the four climate scenarios (1.5°C, 2.0°C, 3.0°C, 4.0°C).
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if ratio_geo is None or getattr(ratio_geo, 'empty', True):
        raise ValueError('`ratio_geo` is missing/empty.')

    plot_order = ['rail edge', 'road edge', 'iww port', 'airports','ports'] if plot_order is None else list(plot_order)
    scenarios = ['15', '20', '30', '40'] if scenarios is None else [str(s) for s in scenarios]

    nuts2_bg = None
    plot_crs = ratio_geo.crs
    if isinstance(nuts2, gpd.GeoDataFrame) and not nuts2.empty:
        nuts2_bg = nuts2.to_crs(plot_crs) if nuts2.crs != plot_crs else nuts2

    # Fixed shared color scale across all scenarios
    norm = mpl.colors.Normalize(vmin=0.5, vmax=2)

    rendered = []
    
    # Loop over infrastructure types (outer loop)
    for infra_type in plot_order:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
        fig.suptitle(f'Future / Present EAIL Ratio — {infra_type.title()}',
                     fontsize=26, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        # Loop over scenarios (inner loop, one per subplot)
        for idx, scen in enumerate(scenarios):
            ax = axes_flat[idx]
            scen_df = ratio_geo[ratio_geo['scenario'].astype(str) == str(scen)].copy()
            
            if nuts2_bg is not None and not nuts2_bg.empty:
                nuts2_bg.plot(ax=ax, color='#f0f0f0', edgecolor='black', linewidth=0.6, zorder=0, aspect='equal')
            
            # Filter by infrastructure type
            sub = scen_df[scen_df['infra_type'] == infra_type].copy()
            
            if sub.empty:
                ax.text(0.5, 0.5, f'No data\n+{float(scen)/10:.1f}°C', 
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=14, color='gray')
                ax.set_title(f'+{float(scen)/10:.1f}°C', fontsize=16, fontweight='bold')
                ax.set_axis_off()
                continue
            
            plot_gdf = sub.copy()
            non_point = ~plot_gdf.geometry.geom_type.isin(['Point', 'MultiPoint'])
            if non_point.any():
                plot_gdf.loc[non_point, 'geometry'] = plot_gdf.loc[non_point].geometry.representative_point()
            
            vals = pd.to_numeric(plot_gdf['ratio_future_present'], errors='coerce').replace([np.inf, -np.inf], np.nan)
            finite = vals.dropna()
            
            if finite.empty:
                plot_gdf.plot(ax=ax, color='#bdbdbd', markersize=40, edgecolor='black', linewidth=0.35, zorder=3, aspect='equal')
                ax.set_title(f'+{float(scen)/10:.1f}°C', fontsize=16, fontweight='bold')
            else:
                log_vals = np.log10(finite.clip(lower=1e-3))
                log_min = float(log_vals.min())
                log_max = float(log_vals.max())
                
                def _bubble_size(v, min_size=35, max_size=250):
                    v = float(max(v, 1e-3))
                    lv = np.log10(v)
                    if log_max <= log_min:
                        return 120.0
                    return min_size + ((lv - log_min) / (log_max - log_min)) * (max_size - min_size)
                
                plot_gdf['__size'] = vals.map(_bubble_size)
                plot_gdf.plot(
                    ax=ax,
                    column='ratio_future_present',
                    cmap='RdYlGn_r',
                    norm=norm,
                    markersize=plot_gdf['__size'],
                    edgecolor='black',
                    linewidth=0.35,
                    alpha=0.95,
                    legend=False,
                    zorder=3,
                    aspect='equal',
                )
                ax.set_title(f'+{float(scen)/10:.1f}°C', fontsize=16, fontweight='bold')
            
            if infra_type in ('road edge', 'rail edge'):
                ax.text(0.02, 0.02, 'Bubble markers for edges', transform=ax.transAxes, 
                        fontsize=10, color='0.35', ha='left', va='bottom')
            
            # Apply bounds
            bounds = None
            if nuts2_bg is not None and not nuts2_bg.empty:
                nb = nuts2_bg.total_bounds
                if np.isfinite(nb).all() and nb[2] > nb[0] and nb[3] > nb[1]:
                    px = (nb[2] - nb[0]) * 0.25
                    py = (nb[3] - nb[1]) * 0.25
                    bounds = (nb[0] - px, nb[1] - py, nb[2] + px, nb[3] + py)
            if bounds is None:
                bounds = get_robust_bounds(plot_gdf)
            if bounds is not None:
                apply_bounds_to_ax(ax, bounds)
            
            ax.set_axis_off()
        
        # Add colorbar to the figure (spanning all subplots)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn_r')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label('EAIL ratio (future / present)', fontsize=14)
        cbar.ax.tick_params(labelsize=11)
        
        rendered.append({'infra_type': infra_type, 'fig': fig, 'axes': axes, 'norm': norm})
        if show:
            plt.show()
    
    return rendered


def compute_future_present_eail_ratio_outputs(
    present_curve,
    eal_by_infra,
    infra_geo,
    nuts2=None,
    basin_climate_file=None,
    lev07_shp=None,
    lev08_shp=None,
    lev08_id_col='HYBAS_ID',
    scenario_list=None,
    plot_order=None,
    plot_scenarios=None,
    show=True,
):
    """Compute and optionally plot future/present EAIL ratios by scenario."""
    if eal_by_infra is None or getattr(eal_by_infra, 'empty', True):
        raise ValueError('`eal_by_infra` is missing/empty.')
    if infra_geo is None or getattr(infra_geo, 'empty', True):
        raise ValueError('`infra_geo` is missing/empty.')

    if basin_climate_file is None:
        raise ValueError('`basin_climate_file` is required.')
    if lev07_shp is None or lev08_shp is None:
        raise ValueError('`lev07_shp` and `lev08_shp` are required.')

    basin_climate_file = Path(basin_climate_file)
    lev07_shp = Path(lev07_shp)
    lev08_shp = Path(lev08_shp)

    if not basin_climate_file.exists():
        raise FileNotFoundError(f'Basin RP-shift file not found: {basin_climate_file}')
    if not lev07_shp.exists() or not lev08_shp.exists():
        missing = [p for p in [lev07_shp, lev08_shp] if not p.exists()]
        raise FileNotFoundError(f'Missing basin shapefiles: {missing}')

    basins_climate = gpd.read_parquet(basin_climate_file)
    if not isinstance(basins_climate, gpd.GeoDataFrame):
        basins_climate = gpd.GeoDataFrame(basins_climate, geometry='geometry')
    if basins_climate.crs is None:
        basins_climate = basins_climate.set_crs('EPSG:3035', allow_override=True)

    rp_map = build_rp_shift_map(basins_climate, lev08_id_col=lev08_id_col)

    basins_lev07 = gpd.read_file(lev07_shp)
    basins_lev08 = gpd.read_file(lev08_shp)
    lev07_id_col = _hybas_id_col(basins_lev07, 'lev07 shapefile')
    lev08_id_detected = _hybas_id_col(basins_lev08, 'lev08 shapefile')

    infra_lev07 = link_infra_to_basins(infra_geo, basins_lev07, lev07_id_col)
    lev08_bridge = bridge_lev07_to_lev08(basins_lev07, basins_lev08, lev07_id_col, lev08_id_detected)

    infra_clim = (
        infra_lev07.merge(lev08_bridge, on='lev07_HYBAS_ID', how='left')
        [['infra_type', 'infra_id_norm', 'lev08_HYBAS_ID']].drop_duplicates()
    )

    if infra_clim.empty:
        infra_clim = eal_by_infra[['infra_type', 'infra_id']].drop_duplicates().copy()
        infra_clim.columns = ['infra_type', 'infra_id_norm']
        infra_clim['lev08_HYBAS_ID'] = np.nan

    pc = present_curve.copy() if present_curve is not None else eal_by_infra.copy()
    if 'infra_id_norm' not in pc.columns:
        pc['infra_id_norm'] = pc.apply(
            lambda r: _norm_infra_key_by_type(r.get('infra_id'), r.get('infra_type')),
            axis=1,
        )

    for candidate in ['lev08_HYBAS_ID', 'lev08_HYBAS_ID_x', 'lev08_HYBAS_ID_y']:
        if candidate in pc.columns:
            pc = pc.drop(columns=[candidate])

    present_curve_with_lev08 = pc.merge(infra_clim, on=['infra_type', 'infra_id_norm'], how='left')
    if 'lev08_HYBAS_ID' not in present_curve_with_lev08.columns:
        lev_candidates = [c for c in ['lev08_HYBAS_ID_x', 'lev08_HYBAS_ID_y'] if c in present_curve_with_lev08.columns]
        if lev_candidates:
            present_curve_with_lev08['lev08_HYBAS_ID'] = present_curve_with_lev08[lev_candidates].bfill(axis=1).iloc[:, 0]
    for candidate in ['lev08_HYBAS_ID_x', 'lev08_HYBAS_ID_y']:
        if candidate in present_curve_with_lev08.columns:
            present_curve_with_lev08 = present_curve_with_lev08.drop(columns=[candidate])

    if present_curve_with_lev08.empty:
        raise ValueError('No present loss curves remain after lev08 assignment.')

    future_base = present_curve_with_lev08.copy().rename(columns={'rp_num': 'rp_present'})
    future_base['rp_present'] = pd.to_numeric(future_base['rp_present'], errors='coerce').round(0).astype('Int64')
    future_base['lev08_HYBAS_ID'] = pd.to_numeric(future_base['lev08_HYBAS_ID'], errors='coerce')

    if scenario_list is None:
        scenario_list = sorted(rp_map['scenario'].dropna().astype(str).unique().tolist())
    scen_df = pd.DataFrame({'scenario': [str(s) for s in scenario_list]})
    future_base['__k'] = 1
    scen_df['__k'] = 1
    future_base = future_base.merge(scen_df, on='__k', how='left').drop(columns='__k')

    rp_map_join = rp_map[['lev08_HYBAS_ID', 'rp_present', 'scenario', 'rp_future']].copy()
    rp_map_join['lev08_HYBAS_ID'] = pd.to_numeric(rp_map_join['lev08_HYBAS_ID'], errors='coerce')
    rp_map_join['rp_present'] = pd.to_numeric(rp_map_join['rp_present'], errors='coerce').round(0).astype('Int64')
    rp_map_join['scenario'] = rp_map_join['scenario'].astype(str)
    future_base['scenario'] = future_base['scenario'].astype(str)
    future_base = future_base.merge(rp_map_join, on=['lev08_HYBAS_ID', 'rp_present', 'scenario'], how='left')

    future_agg = (
        future_base.groupby(['infra_type', 'infra_id_norm', 'rp_present', 'scenario'], as_index=False)
        .agg(loss_eur=('loss_eur', 'first'), rp_future=('rp_future', 'mean'))
    )
    future_agg['rp_eff'] = future_agg['rp_future'].fillna(future_agg['rp_present'].astype(float))

    future_rows = []
    for (itype, iid, scen), group in future_agg.groupby(['infra_type', 'infra_id_norm', 'scenario']):
        future_rows.append({
            'infra_type': itype,
            'infra_id_norm': iid,
            'scenario': scen,
            'EAIL_future': _eal_from_curve(group.rename(columns={'rp_eff': '_rp'}), '_rp', 'loss_eur'),
        })

    future_eal = pd.DataFrame(future_rows)
    if future_eal.empty:
        raise ValueError('No future EAIL computed.')

    ratio_df = future_eal.merge(
        eal_by_infra.rename(columns={'infra_id': 'infra_id_norm', 'EAIL_indirect_eur_per_year': 'EAIL_present'}),
        on=['infra_type', 'infra_id_norm'],
        how='left',
    )
    ratio_df['ratio_future_present'] = ratio_df['EAIL_future'] / ratio_df['EAIL_present'].replace(0, np.nan)

    ratio_geo = infra_geo.merge(ratio_df, on=['infra_type', 'infra_id_norm'], how='inner')
    ratio_geo = gpd.GeoDataFrame(ratio_geo, geometry='geometry', crs=infra_geo.crs)
    if ratio_geo.empty:
        raise ValueError('No geometries for ratio plotting.')

    plot_order = ['rail edge', 'road edge', 'iww port','port','airport'] if plot_order is None else list(plot_order)
    rendered = plot_future_present_eail_ratio_maps(
        ratio_geo=ratio_geo,
        nuts2=nuts2,
        plot_order=plot_order,
        scenarios=plot_scenarios,
        show=show,
    )

    return {
        'basins_climate': basins_climate,
        'rp_map': rp_map,
        'basins_lev07': basins_lev07,
        'basins_lev08': basins_lev08,
        'infra_lev07': infra_lev07,
        'lev08_bridge': lev08_bridge,
        'infra_clim': infra_clim,
        'present_curve_with_lev08': present_curve_with_lev08,
        'future_agg': future_agg,
        'future_eal': future_eal,
        'ratio_df': ratio_df,
        'ratio_geo': ratio_geo,
        'rendered_plots': rendered,
    }


# ── Climate & Basin Linking ────────────────────────────────────────────────────

def parse_rp_shift_column(col_name):
    """Parse RP shift column name to extract (return_period, scenario).
    
    Matches patterns like '100_rp_change_20' → (100, '20')
    
    Parameters
    ----------
    col_name : str
        Column name to parse.
    
    Returns
    -------
    tuple or None
        (return_period: int, scenario: str) or None if no match.
    """
    c = str(col_name).lower()
    m = re.match(r'^(\d+)_rp_change_(15|20|30|40)$', c)
    if m:
        return int(m.group(1)), m.group(2)
    m = re.match(r'^rp[_-]?(\d+)[_-]?change[_-]?(15|20|30|40)$', c)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def build_rp_shift_map(basins_climate, lev08_id_col='HYBAS_ID'):
    """Build long-format table of RP shifts from climate basin parquet.
    
    Parameters
    ----------
    basins_climate : GeoDataFrame
        Climate data with basin IDs and RP shift columns.
    lev08_id_col : str
        Basin ID column name (default 'HYBAS_ID').
    
    Returns
    -------
    DataFrame
        Long-format table with columns: lev08_HYBAS_ID, rp_present, scenario, rp_future.
    """
    shift_cols = [(c, *parse_rp_shift_column(c)) for c in basins_climate.columns 
                  if parse_rp_shift_column(c) is not None]
    
    if not shift_cols:
        raise ValueError('No scenario RP-shift columns found matching `AA_rp_change_BB`.')
    
    map_rows = []
    for col_name, rp_present, scen in shift_cols:
        tmp = basins_climate[[lev08_id_col, col_name]].copy()
        tmp = tmp.rename(columns={col_name: 'rp_future', lev08_id_col: 'lev08_HYBAS_ID'})
        tmp['rp_present'] = int(rp_present)
        tmp['scenario'] = scen
        map_rows.append(tmp)
    
    rp_map = pd.concat(map_rows, ignore_index=True)
    rp_map['rp_future'] = pd.to_numeric(rp_map['rp_future'], errors='coerce')
    rp_map['lev08_HYBAS_ID'] = pd.to_numeric(rp_map['lev08_HYBAS_ID'], errors='coerce')
    rp_map = rp_map[rp_map['rp_future'].notna() & (rp_map['rp_future'] > 0)].copy()
    
    return rp_map


def link_infra_to_basins(infra_geo, basins_lev07, lev07_id_col):
    """Link infrastructure to lev07 basins via spatial join.
    
    Parameters
    ----------
    infra_geo : GeoDataFrame
        Infrastructure geometries.
    basins_lev07 : GeoDataFrame
        lev07 basin shapes.
    lev07_id_col : str
        Basin ID column in lev07.
    
    Returns
    -------
    DataFrame
        infra_type, infra_id_norm, lev07_HYBAS_ID for each asset.
    """
    infra_pts = infra_geo.copy()
    infra_pts['geometry'] = infra_pts.geometry.representative_point()
    if infra_pts.crs is None:
        infra_pts = infra_pts.set_crs('EPSG:4326', allow_override=True)
    if infra_pts.crs != basins_lev07.crs:
        infra_pts = infra_pts.to_crs(basins_lev07.crs)
    infra_pts = infra_pts[infra_pts.geometry.notna() & ~infra_pts.geometry.is_empty].copy()
    
    infra_lev07 = gpd.sjoin_nearest(
        infra_pts[['infra_type', 'infra_id_norm', 'geometry']],
        basins_lev07[[lev07_id_col, 'geometry']], how='left', distance_col='_d07'
    ).drop(columns=['index_right'], errors='ignore')
    infra_lev07 = infra_lev07.rename(columns={lev07_id_col: 'lev07_HYBAS_ID'})
    infra_lev07['lev07_HYBAS_ID'] = pd.to_numeric(infra_lev07['lev07_HYBAS_ID'], errors='coerce')
    infra_lev07 = infra_lev07.dropna(subset=['lev07_HYBAS_ID']).drop_duplicates(subset=['infra_type', 'infra_id_norm'])
    
    return infra_lev07[['infra_type', 'infra_id_norm', 'lev07_HYBAS_ID']]


def bridge_lev07_to_lev08(basins_lev07, basins_lev08, lev07_id_col, lev08_id_col):
    """Bridge lev07 and lev08 basins via spatial join.
    
    Parameters
    ----------
    basins_lev07, basins_lev08 : GeoDataFrame
        Level 7 and 8 basin shapefiles.
    lev07_id_col, lev08_id_col : str
        ID column names.
    
    Returns
    -------
    DataFrame
        lev07_HYBAS_ID, lev08_HYBAS_ID pairs.
    """
    if basins_lev08.crs != basins_lev07.crs:
        basins_lev08 = basins_lev08.to_crs(basins_lev07.crs)
    
    lev08_pts = basins_lev08[[lev08_id_col, 'geometry']].copy()
    lev08_pts['geometry'] = lev08_pts.geometry.representative_point()
    lev08_pts = lev08_pts[lev08_pts.geometry.notna() & ~lev08_pts.geometry.is_empty].copy()
    
    bridge = gpd.sjoin_nearest(
        lev08_pts, basins_lev07[[lev07_id_col, 'geometry']], how='left', distance_col='_d_bridge'
    ).drop(columns=['index_right'], errors='ignore')
    
    _l7_col = next((c for c in [lev07_id_col, f'{lev07_id_col}_right', f'{lev07_id_col}_r'] 
                    if c in bridge.columns), None)
    _l8_col = next((c for c in [lev08_id_col, f'{lev08_id_col}_left', f'{lev08_id_col}_l'] 
                    if c in bridge.columns), None)
    if _l7_col is None or _l8_col is None:
        raise ValueError(f'Could not resolve lev07/lev08 columns. Available: {bridge.columns.tolist()}')
    
    bridge = bridge.rename(columns={_l8_col: 'lev08_HYBAS_ID', _l7_col: 'lev07_HYBAS_ID'})
    bridge['lev08_HYBAS_ID'] = pd.to_numeric(bridge['lev08_HYBAS_ID'], errors='coerce')
    bridge['lev07_HYBAS_ID'] = pd.to_numeric(bridge['lev07_HYBAS_ID'], errors='coerce')
    bridge = bridge.dropna(subset=['lev08_HYBAS_ID', 'lev07_HYBAS_ID']).drop_duplicates()
    
    return bridge[['lev07_HYBAS_ID', 'lev08_HYBAS_ID']]


def prepare_random_basin_event_data(
    disruption_sectors,
    industry_locs,
    road_edges,
    rail_edges,
    iww_nodes,
    ports,
    airports,
    rp_target=100,
    seed=42,
    region_gdf=None,
    restrict_to_region=False,
    basin_layer=None,
    basin_id_col=None,
):
    """Prepare damaged infrastructure and affected industry layers for one random basin event.

    Returns
    -------
    dict
        Keys: chosen_basin, rp_target, damaged_geo, industry_geo, infra_geo, event_df.
    """
    if disruption_sectors is None or getattr(disruption_sectors, 'empty', True):
        raise ValueError('`disruption_sectors` is missing/empty.')
    if industry_locs is None or getattr(industry_locs, 'empty', True):
        raise ValueError('`industry_locs` is missing/empty.')

    df = extract_loss_columns(disruption_sectors.copy())
    df = normalize_infra_types(df)
    plot_order = ['rail edge', 'road edge', 'iww port', 'airport', 'port']
    df = df[df['infra_type'].isin(plot_order)].copy()

    basin_col = next((c for c in ['basin_id', 'basin', 'BASIN_ID'] if c in df.columns), None)
    if basin_col is None:
        raise ValueError('No basin column found in disruption data.')

    event_df = df[pd.to_numeric(df['rp_num'], errors='coerce').eq(rp_target)].copy()
    if event_df.empty:
        raise ValueError(f'No records found for RP={rp_target}.')

    basins = event_df[basin_col].dropna().unique().tolist()
    if not basins:
        raise ValueError('No basin IDs available after RP filtering.')

    if (
        restrict_to_region
        and isinstance(region_gdf, gpd.GeoDataFrame)
        and not region_gdf.empty
        and isinstance(basin_layer, gpd.GeoDataFrame)
        and not basin_layer.empty
    ):
        region = region_gdf.copy()
        basin_layer_local = basin_layer.copy()

        if region.crs is None:
            region = region.set_crs(basin_layer_local.crs or 'EPSG:4326', allow_override=True)
        if basin_layer_local.crs is None:
            basin_layer_local = basin_layer_local.set_crs(region.crs or 'EPSG:4326', allow_override=True)
        if region.crs != basin_layer_local.crs:
            region = region.to_crs(basin_layer_local.crs)

        basin_id_col = basin_id_col or next((c for c in ['HYBAS_ID', 'id', 'basin_id'] if c in basin_layer_local.columns), None)
        if basin_id_col is not None and 'geometry' in basin_layer_local.columns:
            region_union = region.geometry.union_all()
            basin_layer_local = basin_layer_local[basin_layer_local.geometry.notna() & ~basin_layer_local.geometry.is_empty].copy()
            basin_layer_local['__rep_pt__'] = basin_layer_local.geometry.representative_point()
            basins_in_region = basin_layer_local[basin_layer_local['__rep_pt__'].within(region_union)][basin_id_col].astype(str).tolist()
            if basins_in_region:
                basins = [b for b in basins if str(b) in set(basins_in_region)]

    chosen_basin = basins[int(np.random.default_rng(seed).integers(0, len(basins)))]
    event_df = event_df[event_df[basin_col].eq(chosen_basin)].copy()

    infra_event = (
        event_df.groupby(['infra_type', 'infra_id'], as_index=False)
        .agg(loss_eur=('total_loss_eur', 'sum'))
    )
    infra_event['infra_id_norm'] = infra_event['infra_id'].map(_norm_key)

    industry_id_col = next((c for c in ['origin_id', 'origin_industry_id', 'industry_id_orig'] if c in event_df.columns), None)
    if industry_id_col is None:
        raise ValueError('No origin industry ID column found in disruption data.')

    industry_total = event_df.groupby(industry_id_col, as_index=False).agg(total_loss_eur=('total_loss_eur', 'sum'))
    industry_event = event_df.groupby([industry_id_col, 'infra_type'], as_index=False).agg(loss_eur=('total_loss_eur', 'sum'))
    dominant = (
        industry_event.sort_values('loss_eur', ascending=False)
        .drop_duplicates(subset=[industry_id_col])[[industry_id_col, 'infra_type']]
        .rename(columns={'infra_type': 'dominant_infra_type'})
    )
    industry_event_plot = industry_total.merge(dominant, on=industry_id_col, how='left')

    locs = industry_locs.copy()
    if 'industry_id' not in locs.columns and 'id' in locs.columns:
        locs = locs.rename(columns={'id': 'industry_id'})
    if 'industry_id' not in locs.columns:
        raise ValueError('`industry_locs` must contain `industry_id` (or `id`).')

    locs['industry_id'] = locs['industry_id'].astype(str).str.strip()
    industry_event_plot[industry_id_col] = industry_event_plot[industry_id_col].astype(str).str.strip()

    industry_geo = industry_event_plot.merge(
        locs[['industry_id', 'geometry']],
        left_on=industry_id_col,
        right_on='industry_id',
        how='left',
    )
    industry_geo = gpd.GeoDataFrame(
        industry_geo.drop(columns=['industry_id'], errors='ignore'),
        geometry='geometry',
        crs=locs.crs,
    )
    industry_geo = industry_geo[industry_geo.geometry.notna() & (industry_geo['total_loss_eur'] > 0)].copy()
    industry_geo = _infer_and_fix_crs(industry_geo)

    keys_by_type = {
        t: set(infra_event.loc[infra_event['infra_type'].eq(t), 'infra_id_norm'].astype(str))
        for t in plot_order
    }

    road_geo = _build_geo_keys(road_edges, 'road edge', ('edge_id', 'id', 'link_id', 'fid'), keys_by_type.get('road edge', set()))
    rail_geo = _build_geo_keys(rail_edges, 'rail edge', ('edge_id', 'id', 'link_id', 'fid'), keys_by_type.get('rail edge', set()))
    iww_geo = _build_geo_keys(
        iww_nodes,
        'iww port',
        ('id', 'node_id', 'port_id', 'fid'),
        keys_by_type.get('iww port', set()),
        extra_filter=lambda d: d['feature'].astype(str).str.lower().eq('port') if 'feature' in d.columns else pd.Series(True, index=d.index),
    )
    port_geo = _build_geo_keys(ports, 'port', ('port_code', 'id', 'port_id', 'node_id', 'fid'), keys_by_type.get('port', set()))
    airport_geo = _build_geo_keys(airports, 'airport', ('id', 'airport_id', 'icao', 'node_id', 'fid'), keys_by_type.get('airport', set()))

    infra_geo = pd.concat([road_geo, rail_geo, iww_geo, port_geo, airport_geo], ignore_index=True)
    infra_geo = gpd.GeoDataFrame(
        infra_geo,
        geometry='geometry',
        crs=next(
            (
                g.crs
                for g in [road_geo, rail_geo, iww_geo, port_geo, airport_geo]
                if isinstance(g, gpd.GeoDataFrame) and not g.empty and g.crs is not None
            ),
            'EPSG:4326',
        ),
    )

    if 'infra_id_norm' not in infra_geo.columns and 'match_key' in infra_geo.columns:
        infra_geo = infra_geo.rename(columns={'match_key': 'infra_id_norm'})

    if infra_geo.empty:
        raise ValueError('No infrastructure geometries available for plotting.')

    damaged_geo = infra_geo.merge(
        infra_event[['infra_type', 'infra_id_norm', 'loss_eur']],
        on=['infra_type', 'infra_id_norm'],
        how='inner',
    )
    damaged_geo = gpd.GeoDataFrame(damaged_geo, geometry='geometry', crs=infra_geo.crs)

    return {
        'chosen_basin': chosen_basin,
        'rp_target': rp_target,
        'damaged_geo': damaged_geo,
        'industry_geo': industry_geo,
        'infra_geo': infra_geo,
        'event_df': event_df,
    }


def plot_random_basin_event_map(
    damaged_geo,
    industry_geo,
    chosen_basin,
    rp_target,
    nuts2=None,
    study_area=None,
    basins_lev07=None,
    basins_lev08=None,
    color_map=None,
):
    """Plot random basin event map with damaged assets and affected industries."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    if color_map is None:
        color_map = {
            'road edge': '#d73027',
            'rail edge': '#e91e90',
            'iww port': '#1a9850',
            'port': '#f46d43',
            'airport': '#542788',
        }

    plot_crs = None
    for gdf in [nuts2, study_area, damaged_geo]:
        if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty and gdf.crs is not None:
            plot_crs = gdf.crs
            break
    plot_crs = plot_crs or 'EPSG:4326'

    dgeo = damaged_geo.copy()
    igeo = industry_geo.copy()
    if not dgeo.empty and dgeo.crs != plot_crs:
        dgeo = dgeo.to_crs(plot_crs)
    if not igeo.empty and igeo.crs != plot_crs:
        igeo = igeo.to_crs(plot_crs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    if isinstance(nuts2, gpd.GeoDataFrame) and not nuts2.empty:
        nuts2_bg = nuts2.to_crs(plot_crs) if nuts2.crs != plot_crs else nuts2
        nuts2_bg.plot(ax=ax, color='#f7f7f7', edgecolor='black', linewidth=0.4, zorder=0, aspect='equal')

    if isinstance(study_area, gpd.GeoDataFrame) and not study_area.empty:
        sa = study_area.to_crs(plot_crs) if study_area.crs != plot_crs else study_area
        sa.boundary.plot(ax=ax, color='0.5', linewidth=1.0, linestyle='--', zorder=1, aspect='equal')

    basin_plot_gdf = None
    basin_key = _norm_key(chosen_basin)
    for bdf in [basins_lev08, basins_lev07]:
        if not isinstance(bdf, gpd.GeoDataFrame) or bdf.empty:
            continue
        for bid_col in [c for c in bdf.columns if 'HYBAS' in c or 'basin' in str(c).lower()]:
            mask = bdf[bid_col].map(_norm_key).eq(basin_key)
            if mask.any():
                basin_plot_gdf = bdf.loc[mask].copy()
                break
        if basin_plot_gdf is not None:
            break

    if basin_plot_gdf is not None and not basin_plot_gdf.empty:
        if basin_plot_gdf.crs != plot_crs:
            basin_plot_gdf = basin_plot_gdf.to_crs(plot_crs)
        basin_plot_gdf.plot(ax=ax, facecolor='#deebf7', edgecolor='none', alpha=0.4, zorder=1, aspect='equal')
        basin_plot_gdf.plot(ax=ax, facecolor='none', edgecolor='#2166ac', linewidth=2.2, zorder=2, aspect='equal')

    if not dgeo.empty:
        dgeo['loss_eur'] = pd.to_numeric(dgeo['loss_eur'], errors='coerce').fillna(0.0)
        for infra_type, sub in dgeo.groupby('infra_type'):
            color = color_map.get(infra_type, 'black')
            is_point = sub.geometry.geom_type.isin(['Point', 'MultiPoint'])
            if (~is_point).any():
                lw = 1.0 + 4.0 * (np.log10(sub.loc[~is_point, 'loss_eur'] + 1) / max(np.log10(sub['loss_eur'].max() + 1), 1.0))
                sub.loc[~is_point].plot(ax=ax, color=color, linewidth=lw, alpha=0.8, zorder=3, aspect='equal')
            if is_point.any():
                ms = 20 + 180 * (np.log10(sub.loc[is_point, 'loss_eur'] + 1) / max(np.log10(sub['loss_eur'].max() + 1), 1.0))
                sub.loc[is_point].plot(ax=ax, color=color, markersize=ms, edgecolor='white', linewidth=0.5, alpha=0.9, zorder=4, aspect='equal')

    if not igeo.empty:
        igeo['total_loss_eur'] = pd.to_numeric(igeo['total_loss_eur'], errors='coerce').fillna(0.0)
        for infra_type, sub in igeo.groupby('dominant_infra_type'):
            color = color_map.get(infra_type, '#333333')
            size = 10 + 220 * (np.log10(sub['total_loss_eur'] + 1) / max(np.log10(igeo['total_loss_eur'].max() + 1), 1.0))
            sub.plot(ax=ax, color=color, markersize=size, marker='o', alpha=0.45, edgecolor='black', linewidth=0.25, zorder=5, aspect='equal')

    legend_items = [
        mlines.Line2D([], [], color=color_map[k], marker='o', linestyle='None', markersize=8, label=k)
        for k in ['road edge', 'rail edge', 'iww port', 'port', 'airport']
    ]
    leg1 = ax.legend(handles=legend_items, title='Failing infrastructure type', loc='upper left', frameon=True)
    ax.add_artist(leg1)

    if not igeo.empty:
        max_loss = float(igeo['total_loss_eur'].max())
        for v in [max_loss * 0.1, max_loss * 0.4, max_loss * 0.8]:
            if v > 0:
                s = 10 + 220 * (np.log10(v + 1) / max(np.log10(max_loss + 1), 1.0))
                ax.scatter([], [], s=s, c='none', edgecolors='black', label=f'Industry loss ~ €{v:,.0f}')
        ax.legend(loc='lower right', frameon=True, title='Industry bubble size')

    ax.set_title(f'Basin Event (Basin={chosen_basin}, RP={rp_target})\nDamaged infrastructure + affected industries', fontsize=13)
    ax.set_axis_off()

    zoom_gdf = basin_plot_gdf if basin_plot_gdf is not None and not basin_plot_gdf.empty else None
    if zoom_gdf is None and isinstance(study_area, gpd.GeoDataFrame) and not study_area.empty:
        zoom_gdf = study_area.to_crs(plot_crs) if study_area.crs != plot_crs else study_area
    if zoom_gdf is not None:
        minx, miny, maxx, maxy = zoom_gdf.total_bounds
        if np.isfinite([minx, miny, maxx, maxy]).all() and maxx > minx and maxy > miny:
            padx, pady = (maxx - minx) * 0.15, (maxy - miny) * 0.15
            ax.set_xlim(minx - padx, maxx + padx)
            ax.set_ylim(miny - pady, maxy + pady)

    plt.tight_layout()
    plt.show()

    return {'fig': fig, 'ax': ax}


def build_country_loss_plot_df(
    present_eal,
    nuts2,
    haz_dir,
    infra_cfg,
    infra_order,
    infra_layers,
):
    """Build country-level table with EAD, EAIL, and total per infrastructure type."""
    if nuts2 is None or getattr(nuts2, 'empty', True):
        raise ValueError('`nuts2` is missing/empty.')
    if present_eal is None or getattr(present_eal, 'empty', True):
        raise ValueError('`present_eal` is missing/empty.')
    if 'EAIL_present' not in present_eal.columns:
        raise ValueError('`present_eal` must contain `EAIL_present`.')
    if 'infra_type' not in present_eal.columns:
        raise ValueError('`present_eal` must contain `infra_type`.')

    id_col_eail = _first_existing(present_eal, ['infra_id_norm', 'infra_id', 'match_key', 'id'])
    if id_col_eail is None:
        raise ValueError('Could not find infrastructure ID column in `present_eal`.')

    ead_parts = []
    eail_parts = []

    for itype in infra_order:
        cfg = infra_cfg[itype]
        geo_obj = infra_layers.get(cfg['geo_var'])
        geo_lookup = _build_geo_lookup(geo_obj, cfg['geo_cols'])
        if geo_lookup.empty:
            continue
        geo_lookup = geo_lookup.copy()
        geo_lookup['_match_key'] = geo_lookup['_match_key'].map(lambda value: _norm_infra_key_by_type(value, itype))
        geo_lookup = geo_lookup[geo_lookup['_match_key'].ne('')].drop_duplicates('_match_key').copy()
        if geo_lookup.empty:
            continue

        hfile = Path(haz_dir) / cfg['haz_file']
        if hfile.exists():
            hz = pd.read_parquet(hfile)
            haz_key_col = _first_existing(hz, [cfg['haz_key']])
            if haz_key_col is not None and 'EAD_min_river_current' in hz.columns:
                h = hz[[haz_key_col, 'EAD_min_river_current']].copy()
                h['_match_key'] = h[haz_key_col].map(lambda value: _norm_infra_key_by_type(value, itype))
                h['EAD'] = pd.to_numeric(h['EAD_min_river_current'], errors='coerce').fillna(0.0)
                h = h[h['_match_key'].ne('')].copy()
                hg = h.merge(geo_lookup, on='_match_key', how='inner')
                if not hg.empty:
                    hg = gpd.GeoDataFrame(hg, geometry='geometry', crs=geo_lookup.crs)
                    hg['infra_type'] = itype
                    ead_parts.append(hg[['infra_type', '_match_key', 'EAD', 'geometry']])

        pe = present_eal[present_eal['infra_type'].eq(itype)].copy()
        if pe.empty:
            continue
        pe['_match_key'] = pe[id_col_eail].map(lambda value: _norm_infra_key_by_type(value, itype))
        pe['EAIL'] = pd.to_numeric(pe['EAIL_present'], errors='coerce').fillna(0.0) / (1000.0 if itype == 'iww port' else 1.0)
        pe = pe[pe['_match_key'].ne('')].copy()
        peg = pe[['_match_key', 'EAIL']].merge(geo_lookup, on='_match_key', how='inner')
        if not peg.empty:
            peg = gpd.GeoDataFrame(peg, geometry='geometry', crs=geo_lookup.crs)
            peg['infra_type'] = itype
            eail_parts.append(peg[['infra_type', '_match_key', 'EAIL', 'geometry']])

    if not ead_parts:
        raise ValueError('No EAD assets could be built.')
    if not eail_parts:
        raise ValueError('No EAIL assets could be built.')

    ead_gdf = gpd.GeoDataFrame(pd.concat(ead_parts, ignore_index=True), geometry='geometry', crs=ead_parts[0].crs)
    eail_gdf = gpd.GeoDataFrame(pd.concat(eail_parts, ignore_index=True), geometry='geometry', crs=eail_parts[0].crs)

    ead_gdf = _country_join(ead_gdf, nuts2)
    eail_gdf = _country_join(eail_gdf, nuts2)

    ead_cty = ead_gdf.groupby(['country', 'infra_type'], as_index=False)['EAD'].sum()
    eail_cty = eail_gdf.groupby(['country', 'infra_type'], as_index=False)['EAIL'].sum()
    plot_df = ead_cty.merge(eail_cty, on=['country', 'infra_type'], how='outer').fillna(0.0)
    plot_df['total'] = plot_df['EAD'] + plot_df['EAIL']
    plot_df = plot_df[plot_df['total'] > 0].copy()
    if plot_df.empty:
        raise ValueError('No country/infra data available after aggregation.')

    country_order = (
        plot_df.groupby('country', as_index=False)['total']
        .sum()
        .sort_values('total', ascending=False)['country']
        .tolist()
    )
    plot_df['country'] = pd.Categorical(plot_df['country'], categories=country_order, ordered=True)
    plot_df['infra_type'] = pd.Categorical(plot_df['infra_type'], categories=infra_order, ordered=True)
    plot_df = plot_df.sort_values(['country', 'infra_type']).copy()
    return plot_df


def plot_country_loss_grouped_bars(plot_df, infra_order, infra_colors):
    """Plot grouped-by-country stacked bars (EAD + EAIL) by infrastructure type."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    countries = plot_df['country'].cat.categories.tolist() if hasattr(plot_df['country'], 'cat') else sorted(plot_df['country'].unique())
    n_c = len(countries)
    n_i = len(infra_order)
    x = np.arange(n_c)
    group_width = 0.86
    bar_w = group_width / max(n_i, 1)

    fig, ax = plt.subplots(figsize=(max(14, n_c * 1.1), 8), constrained_layout=True)

    for j, itype in enumerate(infra_order):
        sub = plot_df[plot_df['infra_type'].eq(itype)].set_index('country')
        ead_vals = np.array([sub.loc[c, 'EAD'] if c in sub.index else 0.0 for c in countries], dtype=float)
        eail_vals = np.array([sub.loc[c, 'EAIL'] if c in sub.index else 0.0 for c in countries], dtype=float)
        xpos = x - group_width / 2 + (j + 0.5) * bar_w
        base_color = infra_colors.get(itype, '#999999')

        ax.bar(
            xpos,
            ead_vals,
            width=bar_w * 0.95,
            color=base_color,
            alpha=0.95,
            edgecolor='black',
            linewidth=0.35,
            hatch='///',
        )
        ax.bar(
            xpos,
            eail_vals,
            width=bar_w * 0.95,
            bottom=ead_vals,
            color=base_color,
            alpha=0.45,
            edgecolor='black',
            linewidth=0.35,
            hatch='\\\\',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=0)
    ax.set_ylabel('Annual loss (EUR/year)')
    ax.set_xlabel('Country')
    ax.set_title('Country-level annual losses by infrastructure type\nStacked components per bar: EAD + EAIL')
    ax.grid(axis='y', alpha=0.2, linewidth=0.6)

    infra_handles = [Patch(facecolor=infra_colors[k], edgecolor='black', label=k) for k in infra_order]
    comp_handles = [
        Patch(facecolor='#666666', edgecolor='black', hatch='///', label='EAD'),
        Patch(facecolor='#666666', edgecolor='black', hatch='\\\\', alpha=0.45, label='EAIL'),
    ]
    leg1 = ax.legend(handles=infra_handles, title='Infrastructure type', loc='upper left', frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=comp_handles, title='Stack component', loc='upper right', frameon=True)

    plt.show()
    return {'fig': fig, 'ax': ax}


def _find_rp_columns(haz_df, plot_rps):
    """Map requested return periods to vulnerability columns in a hazard table."""
    rp_col_map = {}
    for rp in plot_rps:
        candidates = [
            f'vuln_ratio_river_rp{rp}',
            f'vuln_ratio_river_rp0{rp}' if rp < 100 else None,
            f'vuln_ratio_river_rp00{rp}' if rp < 10 else None,
        ]
        candidates = [candidate for candidate in candidates if candidate is not None]
        found = next((candidate for candidate in candidates if candidate in haz_df.columns), None)
        if found is None:
            found = next(
                (
                    column
                    for column in haz_df.columns
                    if f'rp{rp}' in str(column).lower() or f'rp0{rp}' in str(column).lower()
                ),
                None,
            )
        if found:
            rp_col_map[rp] = found
    return rp_col_map


def _prepare_plot_background(nuts2, plot_crs, pad_ratio=0.04):
    """Prepare optional background polygons and padded bounds for plotting."""
    if not isinstance(nuts2, gpd.GeoDataFrame) or nuts2.empty:
        return None, None
    try:
        bg = nuts2.copy()
        if bg.crs is None:
            bg = bg.set_crs('EPSG:4326', allow_override=True)
        if plot_crs is not None and bg.crs != plot_crs:
            bg = bg.to_crs(plot_crs)

        bounds = None
        nb = bg.total_bounds
        if np.isfinite(nb).all() and nb[2] > nb[0] and nb[3] > nb[1]:
            px = (nb[2] - nb[0]) * pad_ratio
            py = (nb[3] - nb[1]) * pad_ratio
            bounds = (nb[0] - px, nb[1] - py, nb[2] + px, nb[3] + py)
        return bg, bounds
    except Exception:
        return None, None


def _plot_tent_rp_panel(ax, haz_df, geo_lookup, rp, rp_col_map, bg, bounds, plot_crs, cmap, norm):
    """Plot one RP panel for TENT vulnerability maps."""
    if rp not in rp_col_map:
        ax.text(0.5, 0.5, f'No data\nRP{rp}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        return False

    vuln_col = rp_col_map[rp]
    haz_rp = haz_df[['_match_key', vuln_col]].copy()
    haz_rp['vuln_ratio'] = pd.to_numeric(haz_rp[vuln_col], errors='coerce')
    haz_rp = haz_rp[haz_rp['vuln_ratio'].notna() & haz_rp['_match_key'].ne('')].copy()

    if haz_rp.empty:
        ax.text(0.5, 0.5, f'No valid data\nRP{rp}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        return False

    merged = haz_rp[['_match_key', 'vuln_ratio']].merge(
        geo_lookup[['_match_key', 'geometry']],
        on='_match_key',
        how='inner',
    )
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=plot_crs)

    if merged.empty:
        ax.text(0.5, 0.5, f'No geo match\nRP{rp}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        return False

    if bg is not None and not bg.empty:
        bg.plot(ax=ax, color='#d9d9d9', edgecolor='black', linewidth=0.5, zorder=0, aspect='equal')

    is_pt = merged.geometry.geom_type.isin(['Point', 'MultiPoint'])
    if (~is_pt).any():
        merged.loc[~is_pt].plot(ax=ax, color='white', linewidth=6.0, zorder=2, aspect='equal')
        merged.loc[~is_pt].plot(
            ax=ax,
            column='vuln_ratio',
            cmap=cmap,
            norm=norm,
            linewidth=4.0,
            legend=False,
            zorder=3,
            aspect='equal',
        )
    if is_pt.any():
        merged.loc[is_pt].plot(ax=ax, color='white', markersize=80, linewidth=0, zorder=3, aspect='equal')
        merged.loc[is_pt].plot(
            ax=ax,
            column='vuln_ratio',
            cmap=cmap,
            norm=norm,
            markersize=50,
            edgecolor='white',
            linewidth=0.4,
            legend=False,
            zorder=4,
            aspect='equal',
        )

    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

    ax.set_title(f'RP{rp}  (mean={merged["vuln_ratio"].mean():.3f})', fontsize=12, fontweight='bold')
    ax.set_axis_off()
    return True


def _add_shared_vulnerability_colorbar(fig, axes, cmap, norm, rec_col, recovery_days_table):
    """Add the shared vulnerability colorbar with disrupted-days labels."""
    import matplotlib as mpl

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.02, aspect=35)
    cbar.ax.tick_params(labelsize=11)
    tick_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{value:.1f}' for value in tick_vals])

    cbar_ax2 = cbar.ax.twinx()
    cbar_ax2.set_ylim(0, 1)
    cbar_ax2.set_yticks(tick_vals)
    cbar_ax2.set_yticklabels(
        [f'{_vuln_to_days(value, rec_col, recovery_days_table):.0f}d' for value in tick_vals],
        fontsize=9,
        color='#333333',
    )
    cbar_ax2.tick_params(axis='y', length=0, pad=2)
    cbar_ax2.set_ylabel('Disrupted days', fontsize=11, color='#333333', rotation=270, labelpad=18)
    cbar.set_label('Vulnerability ratio', fontsize=12, labelpad=10)


def plot_tent_vulnerability_panels(
    haz_dir,
    tent_sources,
    geo_layers=None,
    nuts2=None,
    plot_rps=None,
    recovery_days_table=None,
    mode_to_recovery_col=None,
):
    """Plot 2x2 RP vulnerability panels for each infrastructure type from TENT hazard files.

    Parameters
    ----------
    haz_dir : Path-like
        Directory containing TENT hazard parquet files.
    tent_sources : dict
        Mapping like {infra_type: {'file','key','geo_var','geo_cols'}}.
    nuts2 : GeoDataFrame, optional
        Optional background polygons.
    plot_rps : list[int], optional
        Return periods to plot. Defaults to [10, 50, 100, 500].
    recovery_days_table : DataFrame, optional
        Table with frac_damage and mode columns for right-side colorbar labels.
    mode_to_recovery_col : dict, optional
        Mapping from infra type to recovery table mode column.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    haz_dir = Path(haz_dir)
    plot_rps = [10, 50, 100, 500] if plot_rps is None else list(plot_rps)
    cmap = plt.get_cmap('RdYlGn_r')
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    if recovery_days_table is None:
        recovery_days_table = pd.DataFrame(
            {
                'frac_damage': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'port': [0.0, 0.0, 2.0, 3.5, 5.5, 9.0, 14.0, 22.5, 35.0, 55.0, 85.0],
                'road': [0.0, 0.0, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0, 20.0, 27.0],
                'rail': [0.0, 0.0, 2.0, 3.5, 5.0, 8.0, 12.0, 18.5, 28.5, 43.0, 65.0],
                'airport': [0.0, 0.0, 2.5, 4.5, 7.0, 12.0, 20.0, 33.5, 55.0, 91.0, 150.0],
            }
        )
        recovery_days_table['iww_port'] = recovery_days_table['port']
        recovery_days_table['iww_ports'] = recovery_days_table['port']

    if mode_to_recovery_col is None:
        mode_to_recovery_col = {
            'road edge': 'road',
            'rail edge': 'rail',
            'iww port': 'port',
            'port': 'port',
            'airport': 'airport',
        }

    for infra_type, cfg in tent_sources.items():
        haz_file = haz_dir / cfg['file']
        if not haz_file.exists():
            print(f'[SKIP] {infra_type} — TENT file not found: {haz_file.name}')
            continue

        haz_full = _safe_read_parquet_any(haz_file)
        key_col = next((c for c in haz_full.columns if c.lower() == str(cfg['key']).lower()), None)
        if key_col is None:
            print(f"[SKIP] {infra_type} — key column ({cfg['key']}) not found")
            continue

        rp_col_map = _find_rp_columns(haz_full, plot_rps)
        if not rp_col_map:
            print(f'[SKIP] {infra_type} — no vuln_ratio columns found for requested RPs')
            continue

        haz_full['_match_key'] = haz_full[key_col].map(_norm_key)

        geo_gdf = None
        if isinstance(geo_layers, dict):
            geo_gdf = geo_layers.get(cfg['geo_var'])
        if not isinstance(geo_gdf, gpd.GeoDataFrame) or geo_gdf.empty:
            print(f"[SKIP] {infra_type} — geometry variable `{cfg['geo_var']}` not available")
            continue

        geo_lookup = _build_geo_lookup(geo_gdf, cfg['geo_cols'])
        if geo_lookup.empty:
            print(f'[SKIP] {infra_type} — no matching geometry columns')
            continue

        geo_lookup = _infer_and_fix_crs(geo_lookup)
        plot_crs = geo_lookup.crs

        bg, bounds = _prepare_plot_background(nuts2, plot_crs)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'TENT Vulnerability Ratio — {infra_type}\n(all edges in corridor)', fontsize=15, fontweight='bold')
        axes = axes.ravel()

        rec_col = mode_to_recovery_col.get(infra_type, 'road')

        for idx, rp in enumerate(plot_rps):
            _plot_tent_rp_panel(
                ax=axes[idx],
                haz_df=haz_full,
                geo_lookup=geo_lookup,
                rp=rp,
                rp_col_map=rp_col_map,
                bg=bg,
                bounds=bounds,
                plot_crs=plot_crs,
                cmap=cmap,
                norm=norm,
            )

        _add_shared_vulnerability_colorbar(fig, axes, cmap, norm, rec_col, recovery_days_table)

        plt.show()

        avail = ', '.join([f'RP{rp}' for rp in plot_rps if rp in rp_col_map])
        print(f'  {infra_type}: available RPs = {avail}')
        print()


def run_lau_ead_eail_aggregation(
    selected_nuts_dir,
    country_data_dir,
    root,
    code_dir,
    output_dir,
    nuts2,
    globals_dict,
):
    """Run LAU-level EAD/EAIL aggregation, plotting, and exports.

    Parameters
    ----------
    selected_nuts_dir, country_data_dir,  root : Path-like
        Search roots for LAU boundary files.
    code_dir, output_dir : Path-like
        Output roots for CSV and figure/shapefile exports.
    nuts2 : GeoDataFrame
        Selected corridor NUTS2 polygons used to filter LAUs.
    globals_dict : dict
        Notebook globals() to access in-memory EAD/EAIL intermediate layers.

    Returns
    -------
    dict
        Keys include `lau_gdf_plot`, `lau_gdf`, `ead_combined`, `eail_combined`,
        `ead_by_lau`, `eail_by_lau`, `lau_file`, `fig_path`, `shp_path`, `csv_path`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    code_dir = Path(code_dir)
    output_dir = Path(output_dir)

    # ── Step 1: Locate and load LAU boundaries ───────────────────────────────
    lau_search_dirs = [
        selected_nuts_dir,
        country_data_dir,
        Path(root) / 'inputs' / 'country_data' / 'Corridors_NUTS2' / 'rhine-alpine',
    ]

    lau_name_patterns = [
        'LAU_2024.parquet',
        'LAU_2024.shp',
        '*LAU*.parquet',
        '*lau*.parquet',
        '*LAU*.shp',
        '*lau*.shp',
    ]

    lau_candidates = []
    for search_dir in lau_search_dirs:
        if search_dir is None:
            continue
        search_dir = Path(search_dir)
        if not search_dir.exists():
            continue
        for pat in lau_name_patterns:
            for matched_path in search_dir.glob(pat):
                if matched_path.is_file():
                    lau_candidates.append(matched_path)

    seen = set()
    lau_candidates = [path_obj for path_obj in lau_candidates if not (str(path_obj) in seen or seen.add(str(path_obj)))]

    def _lau_candidate_priority(path_obj):
        name = path_obj.name.lower()
        suffix = path_obj.suffix.lower()
        exact_bonus = 0 if name.startswith('lau_2024') else 10
        fmt_bonus = 0 if suffix == '.parquet' else 1
        return (exact_bonus, fmt_bonus, len(name))

    lau_candidates = sorted(lau_candidates, key=_lau_candidate_priority)

    def _load_lau_any(path_obj):
        suffix = path_obj.suffix.lower()
        if suffix == '.parquet':
            try:
                gdf = _read_parquet_as_gdf(path_obj)
                if not isinstance(gdf, gpd.GeoDataFrame):
                    if 'geometry' in gdf.columns:
                        gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
                    else:
                        raise ValueError('Parquet loaded but no geometry column found.')
                return gdf
            except Exception as exc:
                raise RuntimeError(f'Parquet load failed: {exc}')
        return gpd.read_file(path_obj)

    lau_file = None
    lau_gdf = gpd.GeoDataFrame()
    load_errors = []

    if not lau_candidates:
        print('[ERROR] LAU file not found. Checked directories:')
        for search_dir in lau_search_dirs:
            print(f'  - {search_dir}')
    else:
        for candidate in lau_candidates:
            try:
                temp_lau = _load_lau_any(candidate)
                if temp_lau is None or temp_lau.empty:
                    load_errors.append(f'{candidate} -> loaded empty')
                    continue
                if temp_lau.crs is None:
                    temp_lau = temp_lau.set_crs('EPSG:4326', allow_override=True)
                lau_gdf = temp_lau.to_crs('EPSG:3035')
                lau_file = candidate
                break
            except Exception as exc:
                load_errors.append(f'{candidate} -> {exc}')

    if not lau_gdf.empty:
        selected_nuts2 = nuts2
        if isinstance(selected_nuts2, gpd.GeoDataFrame) and not selected_nuts2.empty:
            nuts2_for_match = selected_nuts2.copy()
            if nuts2_for_match.crs is None:
                nuts2_for_match = nuts2_for_match.set_crs('EPSG:3035', allow_override=True)
            if nuts2_for_match.crs != lau_gdf.crs:
                nuts2_for_match = nuts2_for_match.to_crs(lau_gdf.crs)

            lau_before = len(lau_gdf)
            nuts2_union = nuts2_for_match.geometry.union_all()
            lau_mask = lau_gdf.geometry.intersects(nuts2_union)
            lau_gdf = lau_gdf[lau_mask].copy()
        else:
            print('[WARN] `nuts2` not available or empty; using all loaded LAUs without NUTS2 filtering.')

    # ── Step 2: Collect EAD and EAIL asset layers ────────────────────────────
    ead_all = []
    infra_types = ['railway', 'road', 'iww', 'airport', 'port']
    for infra_type in infra_types:
        ead_col = f'{infra_type}_exposed_ead'
        ead_obj = globals_dict.get(ead_col)
        if isinstance(ead_obj, gpd.GeoDataFrame) and not ead_obj.empty:
            ead_gdf = ead_obj.copy()
            if 'EAD' not in ead_gdf.columns:
                numeric_cols = [col for col in ead_gdf.columns if ead_gdf[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
                if numeric_cols:
                    ead_col_name = numeric_cols[-1]
                    ead_gdf['EAD'] = pd.to_numeric(ead_gdf[ead_col_name], errors='coerce')
            if 'EAD' in ead_gdf.columns:
                ead_gdf['infra_type'] = infra_type
                ead_all.append(ead_gdf[['geometry', 'EAD', 'infra_type']])

    if ead_all:
        ead_combined = pd.concat(ead_all, ignore_index=True)
        ead_combined = gpd.GeoDataFrame(ead_combined, crs='EPSG:3035', geometry='geometry')
        ead_combined['EAD'] = pd.to_numeric(ead_combined['EAD'], errors='coerce').fillna(0)
        print(f'Combined EAD assets: {len(ead_combined):,} ({ead_combined["infra_type"].nunique()} types)')
    else:
        ead_combined = gpd.GeoDataFrame()
        print('[WARN] No EAD data found')

    eail_combined = gpd.GeoDataFrame()
    eail_geo = None
    plot_pack_local = globals_dict.get('plot_pack')
    if isinstance(plot_pack_local, dict):
        eail_geo = plot_pack_local.get('eal_geo')

    if isinstance(eail_geo, gpd.GeoDataFrame) and not eail_geo.empty and 'EAIL_present' in eail_geo.columns:
        eail_combined = eail_geo[['geometry', 'infra_type', 'infra_id', 'EAIL_present']].copy()
        eail_combined = eail_combined.rename(columns={'EAIL_present': 'EAIL'})
    elif isinstance(globals_dict.get('present_eal'), pd.DataFrame) and not globals_dict['present_eal'].empty:
        present_eal_local = globals_dict['present_eal'].copy()
        infra_geo_local = globals_dict.get('infra_geo')
        if isinstance(infra_geo_local, gpd.GeoDataFrame) and not infra_geo_local.empty:
            cols_needed = ['infra_type', 'infra_id', 'geometry']
            eail_combined = infra_geo_local[cols_needed].merge(
                present_eal_local[['infra_type', 'infra_id', 'EAIL_present']],
                on=['infra_type', 'infra_id'],
                how='inner',
            )
            eail_combined = eail_combined.rename(columns={'EAIL_present': 'EAIL'})
        else:
            print('[WARN] `infra_geo` missing/empty; cannot spatialize `present_eal` for LAU aggregation.')
    else:
        print('[WARN] `present_eal`/`plot_pack["eal_geo"]` not available. Run the present-day EAIL cell first.')

    if isinstance(eail_combined, pd.DataFrame) and not eail_combined.empty:
        eail_combined = gpd.GeoDataFrame(eail_combined, geometry='geometry', crs=getattr(eail_combined, 'crs', None))
        if eail_combined.crs is None:
            eail_combined = eail_combined.set_crs('EPSG:3035', allow_override=True)
        elif str(eail_combined.crs) != 'EPSG:3035':
            eail_combined = eail_combined.to_crs('EPSG:3035')
        eail_combined['EAIL'] = pd.to_numeric(eail_combined['EAIL'], errors='coerce').fillna(0)
        print(f'Combined EAIL assets (from present-day EAIL): {len(eail_combined):,} ({eail_combined["infra_type"].nunique()} types)')
    else:
        eail_combined = gpd.GeoDataFrame()
        print('[WARN] No EAIL data found from present-day EAIL outputs')

    ead_by_lau = pd.Series(dtype='float64')
    eail_by_lau = pd.Series(dtype='float64')
    lau_gdf_plot = gpd.GeoDataFrame()

    if lau_gdf.empty or (ead_combined.empty and eail_combined.empty):
        print(f'[ERROR] Cannot proceed: LAU={len(lau_gdf)}, EAD={len(ead_combined)}, EAIL={len(eail_combined)}')
    else:
        lau_gdf_indexed = lau_gdf.copy()
        lau_gdf_indexed['lau_idx'] = range(len(lau_gdf_indexed))

        if not ead_combined.empty:
            ead_lau = gpd.sjoin(ead_combined, lau_gdf_indexed[['geometry', 'lau_idx']], how='left', predicate='intersects')
            ead_by_lau = ead_lau.groupby('lau_idx')['EAD'].sum()
            print(f'\nEAD by LAU: {len(ead_by_lau):,} LAUs with assets (sum aggregation)')
        else:
            print('\nEAD by LAU: [skipped, no data]')

        if not eail_combined.empty:
            eail_lau = gpd.sjoin(eail_combined, lau_gdf_indexed[['geometry', 'lau_idx']], how='left', predicate='intersects')
            eail_by_lau = eail_lau.groupby('lau_idx')['EAIL'].sum()
            print(f'EAIL by LAU: {len(eail_by_lau):,} LAUs with assets')
        else:
            print('EAIL by LAU: [skipped, no data]')

        lau_gdf_plot = lau_gdf_indexed[['geometry', 'lau_idx']].copy()
        if not ead_by_lau.empty:
            lau_gdf_plot = lau_gdf_plot.merge(ead_by_lau.rename('EAD').reset_index(), on='lau_idx', how='left')
            lau_gdf_plot['EAD'] = lau_gdf_plot['EAD'].fillna(0)
        else:
            lau_gdf_plot['EAD'] = 0.0

        if not eail_by_lau.empty:
            lau_gdf_plot = lau_gdf_plot.merge(eail_by_lau.rename('EAIL').reset_index(), on='lau_idx', how='left')
            lau_gdf_plot['EAIL'] = lau_gdf_plot['EAIL'].fillna(0)
        else:
            lau_gdf_plot['EAIL'] = 0.0

        shared_cmap = 'YlOrRd'
        shared_scale_min = 0.0
        shared_scale_max = 7_000_000.0
        bubble_size_min = 10.0
        bubble_size_max = 400.0

        def _bubble_sizes(value_series):
            clipped = pd.to_numeric(value_series, errors='coerce').fillna(0).clip(lower=0, upper=shared_scale_max)
            return bubble_size_min + (clipped / shared_scale_max) * (bubble_size_max - bubble_size_min)

        lau_points = lau_gdf_plot[['lau_idx', 'EAD', 'EAIL', 'geometry']].copy()
        lau_points['EAD'] = pd.to_numeric(lau_points['EAD'], errors='coerce')
        lau_points['EAIL'] = pd.to_numeric(lau_points['EAIL'], errors='coerce')
        lau_points['geometry'] = lau_points.geometry.representative_point()
        lau_points['x'] = lau_points.geometry.x
        lau_points['y'] = lau_points.geometry.y

        epsilon = 1e-9
        ead_points = lau_points[lau_points['EAD'] > epsilon].copy()
        eail_points = lau_points[lau_points['EAIL'] > epsilon].copy()

        print(f'Non-zero EAD bubbles to plot: {len(ead_points):,}')
        print(f'Non-zero EAIL bubbles to plot: {len(eail_points):,}')

        zoom_sources = []
        if not ead_points.empty:
            zoom_sources.append(ead_points[['x', 'y']])
        if not eail_points.empty:
            zoom_sources.append(eail_points[['x', 'y']])

        if zoom_sources:
            zoom_xy = pd.concat(zoom_sources, ignore_index=True)
            if len(zoom_xy) >= 20:
                x_min, x_max = zoom_xy['x'].quantile([0.02, 0.98]).tolist()
                y_min, y_max = zoom_xy['y'].quantile([0.02, 0.98]).tolist()
            else:
                x_min, x_max = zoom_xy['x'].min(), zoom_xy['x'].max()
                y_min, y_max = zoom_xy['y'].min(), zoom_xy['y'].max()
        else:
            x_min, y_min, x_max, y_max = lau_gdf_plot.total_bounds

        span_x = max(x_max - x_min, 1.0)
        span_y = max(y_max - y_min, 1.0)
        pad_x = max(span_x * 0.05, 5_000)
        pad_y = max(span_y * 0.05, 5_000)
        xlim = (x_min - pad_x, x_max + pad_x)
        ylim = (y_min - pad_y, y_max + pad_y)

        fig, (ax_ead, ax_eail) = plt.subplots(1, 2, figsize=(22, 10))
        for ax in [ax_ead, ax_eail]:
            lau_gdf_plot.boundary.plot(ax=ax, color='lightgrey', linewidth=0.25, zorder=1)
            lau_gdf_plot.plot(ax=ax, color='whitesmoke', edgecolor='none', zorder=0)
            ax.set_xlabel('Easting (EPSG:3035)', fontsize=11)
            ax.set_ylabel('Northing (EPSG:3035)', fontsize=11)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        if not ead_points.empty:
            ax_ead.scatter(
                ead_points['x'],
                ead_points['y'],
                c=ead_points['EAD'],
                s=_bubble_sizes(ead_points['EAD']),
                cmap=shared_cmap,
                vmin=shared_scale_min,
                vmax=shared_scale_max,
                alpha=0.85,
                edgecolors='black',
                linewidths=0.2,
                zorder=3,
            )
            ax_ead.set_title('EAD by LAU (colour bubbles)', fontsize=14, fontweight='bold', pad=15)
        else:
            ax_ead.text(0.5, 0.5, 'No EAD data available', ha='center', va='center', fontsize=12, transform=ax_ead.transAxes)
            ax_ead.set_title('EAD by LAU (colour bubbles)', fontsize=14, fontweight='bold')

        if not eail_points.empty:
            ax_eail.scatter(
                eail_points['x'],
                eail_points['y'],
                c=eail_points['EAIL'],
                s=_bubble_sizes(eail_points['EAIL']),
                cmap=shared_cmap,
                vmin=shared_scale_min,
                vmax=shared_scale_max,
                alpha=0.85,
                edgecolors='black',
                linewidths=0.2,
                zorder=3,
            )
            ax_eail.set_title('EAIL by LAU (colour bubbles)', fontsize=14, fontweight='bold', pad=15)
        else:
            ax_eail.text(0.5, 0.5, 'No EAIL data available', ha='center', va='center', fontsize=12, transform=ax_eail.transAxes)
            ax_eail.set_title('EAIL by LAU (colour bubbles)', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        norm = Normalize(vmin=shared_scale_min, vmax=shared_scale_max)
        sm = ScalarMappable(norm=norm, cmap=shared_cmap)
        sm.set_array([])

        cax = fig.add_axes([0.90, 0.15, 0.02, 0.90])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Loss value (€)', fontsize=11)

        fig_path = output_dir / 'lau_ead_eail_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'\nFigure saved: {fig_path}')
        plt.show()

        print('\n=== LAU Aggregation Summary ===')
        print('EAD by LAU:')
        print(f'  Total EAD: €{lau_gdf_plot["EAD"].sum():,.0f}')
        print(f'  Mean per LAU: €{lau_gdf_plot["EAD"].mean():,.0f}')
        print(f'  Max per LAU: €{lau_gdf_plot["EAD"].max():,.0f}')
        print(f'  LAUs with exposure: {(lau_gdf_plot["EAD"] > 0).sum():,}')
        print('\nEAIL by LAU (sum):')
        print(f'  Total: €{lau_gdf_plot["EAIL"].sum():,.0f}')
        print(f'  Mean per LAU: €{lau_gdf_plot["EAIL"].mean():,.0f}')
        print(f'  Max per LAU: €{lau_gdf_plot["EAIL"].max():,.0f}')
        print(f'  LAUs with exposure: {(lau_gdf_plot["EAIL"] > 0).sum():,}')

        shp_path = output_dir / 'lau_ead_eail_aggregated.shp'
        csv_path = output_dir / 'lau_ead_eail_summary.csv'

        try:
            lau_gdf_plot[['geometry', 'lau_idx', 'EAD', 'EAIL']].to_file(shp_path)
            print(f'\nAggregated results saved: {shp_path}')
        except Exception as exc:
            print(f'[WARN] Could not save shapefile: {exc}')

        try:
            lau_gdf_plot[['lau_idx', 'EAD', 'EAIL']].to_csv(csv_path, index=False)
            print(f'Summary table saved: {csv_path}')
        except Exception as exc:
            print(f'[WARN] Could not save CSV: {exc}')

        return {
            'lau_gdf_plot': lau_gdf_plot,
            'lau_gdf': lau_gdf,
            'ead_combined': ead_combined,
            'eail_combined': eail_combined,
            'ead_by_lau': ead_by_lau,
            'eail_by_lau': eail_by_lau,
            'lau_file': lau_file,
            'fig_path': fig_path,
            'shp_path': shp_path,
            'csv_path': csv_path,
        }

    return {
        'lau_gdf_plot': lau_gdf_plot,
        'lau_gdf': lau_gdf,
        'ead_combined': ead_combined,
        'eail_combined': eail_combined,
        'ead_by_lau': ead_by_lau,
        'eail_by_lau': eail_by_lau,
        'lau_file': lau_file,
        'fig_path': None,
        'shp_path': None,
        'csv_path': None,
    }


def _to_4326(gdf):
    if gdf is None or (isinstance(gdf, gpd.GeoDataFrame) and gdf.empty):
        return gdf
    if gdf.crs is None or str(gdf.crs) == "EPSG:4326":
        return gdf
    try:
        return gdf.to_crs("EPSG:4326")
    except Exception:
        return gdf


def _filter_has_L_in_corridor(gdf):
    """Keep rows where CORRIDOR(S) includes token L (alone or with other letters)."""
    if gdf is None or (isinstance(gdf, gpd.GeoDataFrame) and gdf.empty):
        return gdf

    corridor_cols = [c for c in ("CORRIDOR", "CORRIDORS", "corridor", "corridors") if c in gdf.columns]
    if not corridor_cols:
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)

    # Accept forms like: L, A|L, L|B, A;L, A,L
    token_pat = r"(^|[|,;\s])L($|[|,;\s])"

    mask = pd.Series(False, index=gdf.index)
    for col in corridor_cols:
        ser = (
            gdf[col]
            .astype(str)
            .str.upper()
            .str.replace("/", "|", regex=False)
            .str.replace("-", "|", regex=False)
        )
        mask = mask | ser.str.contains(token_pat, regex=True, na=False)

    return gdf[mask].copy()
def _ensure_gdf_crs(gdf, target_crs="EPSG:4326"):
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs(target_crs, allow_override=True)
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf
def _zoom_to_selected_nuts2(ax, gdf=None):
    if isinstance(nuts2, gpd.GeoDataFrame) and not nuts2.empty:
        try:
            target_crs = getattr(gdf, "crs", None)
            if target_crs is None:
                target_crs = "EPSG:4326"
            nuts2_plot = nuts2
            if nuts2_plot.crs is None:
                nuts2_plot = nuts2_plot.set_crs(target_crs, allow_override=True)
            elif str(nuts2_plot.crs) != str(target_crs):
                nuts2_plot = nuts2_plot.to_crs(target_crs)

            if not nuts2_plot.empty:
                # Draw the NUTS2 silhouette behind the risk layers.
                nuts2_plot.plot(
                    ax=ax,
                    facecolor="#f3f3f3",
                    edgecolor="#8b8b8b",
                    linewidth=0.6,
                    alpha=0.8,
                    zorder=0,
                )
                nuts2_plot.boundary.plot(ax=ax, color="#8b8b8b", linewidth=0.4, alpha=0.9, zorder=1)

                bb = nuts2_plot.total_bounds
                pad_x = (bb[2] - bb[0]) * 0.10
                pad_y = (bb[3] - bb[1]) * 0.10
                ax.set_xlim(bb[0] - pad_x, bb[2] + pad_x)
                ax.set_ylim(bb[1] - pad_y, bb[3] + pad_y)
        except Exception:
            pass

def _ensure_gdf_crs(gdf, target_crs="EPSG:4326"):
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs(target_crs, allow_override=True)
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf


def _zoom_to_selected_nuts2(ax, gdf=None):
    if isinstance(nuts2, gpd.GeoDataFrame) and not nuts2.empty:
        try:
            target_crs = getattr(gdf, "crs", None)
            if target_crs is None:
                target_crs = "EPSG:4326"
            nuts2_plot = nuts2
            if nuts2_plot.crs is None:
                nuts2_plot = nuts2_plot.set_crs(target_crs, allow_override=True)
            elif str(nuts2_plot.crs) != str(target_crs):
                nuts2_plot = nuts2_plot.to_crs(target_crs)

            if not nuts2_plot.empty:
                nuts2_plot.plot(
                    ax=ax,
                    facecolor="#f3f3f3",
                    edgecolor="#8b8b8b",
                    linewidth=0.6,
                    alpha=0.8,
                    zorder=0,
                )
                nuts2_plot.boundary.plot(ax=ax, color="#8b8b8b", linewidth=0.4, alpha=0.9, zorder=1)

                bb = nuts2_plot.total_bounds
                pad_x = (bb[2] - bb[0]) * 0.10
                pad_y = (bb[3] - bb[1]) * 0.10
                ax.set_xlim(bb[0] - pad_x, bb[2] + pad_x)
                ax.set_ylim(bb[1] - pad_y, bb[3] + pad_y)
        except Exception:
            pass


def plot_lau_choropleth(ax, lau_gdf, column, title, cmap='Purples', log=False,
                         legend_label=None, vmin=None, vmax=None, europe_countries=None):
    """Plot LAU polygons colored by *column*. Returns the plotted GeoDataFrame
    (with any log-transformed helper column added), or None if there was no
    data to plot (a 'no data' placeholder is still drawn on *ax* in that case)."""
    setup_ax(ax, europe_countries, None, title)
    if not isinstance(lau_gdf, gpd.GeoDataFrame) or lau_gdf.empty or column not in lau_gdf.columns:
        ax.text(0.5, 0.5, f'No LAU data for {column}', ha='center', va='center', transform=ax.transAxes)
        return None
    plot_gdf = _ensure_gdf_crs(lau_gdf.copy(), target_crs='EPSG:4326')
    _zoom_to_selected_nuts2(ax, plot_gdf)
    values = pd.to_numeric(plot_gdf[column], errors='coerce')
    plot_col = column
    if log:
        plot_col = f'_log_{column}'
        plot_gdf[plot_col] = np.log10(values.replace(0, np.nan))
        values = plot_gdf[plot_col]
    if values.notna().sum() == 0:
        ax.text(0.5, 0.5, f'No non-zero LAU data for {column}', ha='center', va='center', transform=ax.transAxes)
        return None
    _vmin = vmin if vmin is not None else values.quantile(0.05)
    _vmax = vmax if vmax is not None else values.quantile(0.95)
    plot_gdf.plot(
        ax=ax,
        column=plot_col,
        cmap=cmap,
        edgecolor='#666666',
        linewidth=0.05,
        alpha=0.8,
        legend=True,
        legend_kwds={'label': legend_label or column, 'shrink': 0.8},
        vmin=_vmin,
        vmax=_vmax,
        missing_kwds={'color': '#eeeeee'},
    )
    if log:
        cbar = ax.collections[-1].colorbar
        format_log_colorbar(cbar)
    return plot_gdf

"""
Utility functions for TEN-T Corridor Analysis

This module provides helper functions for:
- Corridor extraction and parsing
- Map visualization setup
- Flow-based linewidth and markersize calculations
- Raster reprojection and handling
- Data aggregation and loading
- Legend creation
- Risk data merging
"""

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import tempfile
import numpy as np
from matplotlib.lines import Line2D
from typing import List, Tuple, Optional, Dict


def extract_corridors(corridors_str) -> List[str]:
    """
    Extract list of corridor names from CORRIDORS string.
    
    Parameters:
    -----------
    corridors_str : str
        String containing corridor letter codes (e.g., 'ABC')
    
    Returns:
    --------
    list
        List of individual corridor letter codes
    """
    if pd.isna(corridors_str) or corridors_str == '':
        return []
    
    corridors_str = str(corridors_str).strip()
    
    # Split into individual characters (each letter represents a corridor)
    return [c for c in corridors_str if c.isalpha()]


def setup_ax(ax, europe_countries=None, infrastructure=None, title=None):
    """
    Add country boundaries and set map extent for European visualization.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to configure
    europe_countries : GeoDataFrame, optional
        European country boundaries
    infrastructure : dict, optional
        Dictionary of infrastructure GeoDataFrames for CRS matching
    title : str, optional
        Title for the subplot
    """
    if europe_countries is not None:
        europe_countries_projected = europe_countries
        if infrastructure and len(infrastructure) > 0:
            first_gdf = list(infrastructure.values())[0]
            if europe_countries.crs != first_gdf.crs:
                europe_countries_projected = europe_countries.to_crs(first_gdf.crs)
        
        europe_countries_projected.plot(
            ax=ax, 
            color='none', 
            edgecolor='#333333', 
            linewidth=0.8, 
            alpha=0.8, 
            zorder=1
        )
    
    # Zoom to continental Europe (EPSG:3035 coordinates)
    ax.set_xlim(2200000, 6500000)
    ax.set_ylim(1400000, 5500000)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def get_linewidth_freight(flow: float, percentiles: Tuple[float, float, float, float]) -> float:
    """Calculate linewidth for railway freight flows based on percentiles."""
    p05, p35, p65, p95 = percentiles
    
    if flow <= p05:
        return 0.5
    elif flow <= p35:
        return 1.75
    elif flow <= p65:
        return 3
    elif flow <= p95:
        return 4.25
    else:
        return 5.5


def get_linewidth_passenger(flow: float, percentiles: Tuple[float, float, float, float]) -> float:
    """Calculate linewidth for passenger flows based on percentiles."""
    p05, p35, p65, p95 = percentiles
    
    if flow <= p05:
        return 0.5
    elif flow <= p35:
        return 1.75
    elif flow <= p65:
        return 3
    elif flow <= p95:
        return 4.25
    else:
        return 5.5


def get_markersize_freight(flow: float, percentiles: Tuple[float, float, float, float]) -> float:
    """Calculate markersize for freight flows (ports, airports) based on percentiles."""
    p05, p35, p65, p95 = percentiles
    
    if flow <= p05:
        return 25
    elif flow <= p35:
        return 100
    elif flow <= p65:
        return 175
    elif flow <= p95:
        return 225
    else:
        return 300


def get_markersize_passenger(flow: float, percentiles: Tuple[float, float, float, float]) -> float:
    """Calculate markersize for passenger flows (ports, airports) based on percentiles."""
    p05, p35, p65, p95 = percentiles
    
    if flow <= p05:
        return 25
    elif flow <= p35:
        return 100
    elif flow <= p65:
        return 175
    elif flow <= p95:
        return 225
    else:
        return 300


def plot_by_corridor_and_visual_attr(ax, gdf, corridor_colors, visual_attr='linewidth', 
                                     marker='o', alpha=0.6, edgecolor='white', edge_linewidth=1.5):
    """
    Plot features grouped by corridor and visual attribute for performance.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    gdf : GeoDataFrame
        Data to plot with 'primary_corridor' and visual attribute columns
    corridor_colors : dict
        Mapping of corridor codes to colors
    visual_attr : str
        Name of the visual attribute column ('linewidth' or 'markersize')
    marker : str
        Marker style for point data
    alpha : float
        Transparency level
    edgecolor : str
        Edge color for markers
    edge_linewidth : float
        Edge linewidth for markers
    """
    is_line = visual_attr == 'linewidth'
    
    for corridor in corridor_colors.keys():
        corridor_mask = gdf['primary_corridor'] == corridor
        if corridor_mask.any():
            corridor_data = gdf[corridor_mask]
            
            # Plot each visual attribute group together
            for attr_value in corridor_data[visual_attr].unique():
                attr_mask = corridor_data[visual_attr] == attr_value
                
                if is_line:
                    corridor_data[attr_mask].plot(
                        ax=ax,
                        color=corridor_colors[corridor],
                        linewidth=attr_value,
                        alpha=alpha,
                        zorder=2
                    )
                else:
                    corridor_data[attr_mask].plot(
                        ax=ax,
                        color=corridor_colors[corridor],
                        marker=marker,
                        markersize=attr_value,
                        alpha=alpha,
                        edgecolor=edgecolor,
                        linewidth=edge_linewidth,
                        zorder=2
                    )


def reproject_raster_to_3035(raster_path, target_crs='EPSG:3035'):
    """
    Load a raster and reproject to EPSG:3035 if needed.
    
    Parameters:
    -----------
    raster_path : Path or str
        Path to the raster file
    target_crs : str
        Target CRS (default: 'EPSG:3035')
    
    Returns:
    --------
    rasterio.DatasetReader or None
        Opened raster dataset in target CRS, or None if error
    """
    try:
        raster = rasterio.open(raster_path)
        
        # Check if reprojection is needed
        if raster.crs and str(raster.crs) != target_crs:
            
            # Calculate transform for target CRS
            transform, width, height = calculate_default_transform(
                raster.crs, target_crs, raster.width, raster.height,
                *raster.bounds)
            
            kwargs = raster.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Write reprojected raster
            with rasterio.open(temp_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(raster, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=raster.transform,
                    src_crs=raster.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear)
            
            raster.close()
            raster = rasterio.open(temp_path)
        else:
            print(f"Raster already in {target_crs}")
        
        return raster
    except Exception as e:
        print(f"Error loading raster: {e}")
        return None


def aggregate_by_location(gdf, location_col, flow_cols, agg_name):
    """
    Aggregate flow data by location (port_code, airport_id, etc.).
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Data to aggregate
    location_col : str
        Column name for grouping (e.g., 'port_code')
    flow_cols : list
        List of column names to sum
    agg_name : str
        Name for the aggregated total column
    
    Returns:
    --------
    GeoDataFrame
        Aggregated data by location
    """
    if not flow_cols or location_col not in gdf.columns:
        return None
    
    gdf_copy = gdf.copy()
    gdf_copy[agg_name] = gdf_copy[flow_cols].fillna(0).sum(axis=1)
    
    agg_result = gdf_copy.groupby(location_col, as_index=False).agg({
        agg_name: 'sum',
        'primary_corridor': 'first',
        'geometry': 'first'
    })
    
    return gpd.GeoDataFrame(agg_result, geometry='geometry', crs=gdf.crs)


def create_flow_legend(percentiles, unit='MT/year', marker=None, is_freight=True):
    """
    Create legend elements for flow visualization.
    
    Parameters:
    -----------
    percentiles : tuple
        (p05, p35, p65, p95) percentile values
    unit : str
        Unit for display (e.g., 'MT/year', 'M trips/year')
    marker : str, optional
        Marker type ('o', '^', 's', etc.) for point data
    is_freight : bool
        Whether this is freight data (affects formatting)
    
    Returns:
    --------
    list
        List of Line2D legend elements
    """
    p05, p35, p65, p95 = percentiles
    
    # Format values based on type
    if is_freight:
        labels = [f'{p/1000:.1f} {unit}' for p in percentiles]
    else:
        labels = [f'{p/1e6:.2f} {unit}' for p in percentiles]
    
    if marker:
        # Point data (ports, airports)
        sizes = [5, 8, 11, 14]
        return [
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                   markersize=size, label=label, markeredgecolor='white')
            for size, label in zip(sizes, labels)
        ]
    else:
        # Line data (railways, IWW)
        widths = [0.5, 1.75, 3, 4.25]
        return [
            Line2D([0], [0], color='gray', lw=width, label=label)
            for width, label in zip(widths, labels)
        ]


def load_infrastructure_parquet(file_path, infra_type):
    """
    Load infrastructure data from parquet with fallback method.
    
    Parameters:
    -----------
    file_path : Path
        Path to parquet file
    infra_type : str
        Type of infrastructure (for logging)
    
    Returns:
    --------
    GeoDataFrame or None
        Loaded infrastructure data
    """
    if not file_path.exists():
        print(f"{infra_type:20s}: File not found")
        return None
    
    try:
        gdf = gpd.read_parquet(file_path)
        print(f"{infra_type:20s}: {len(gdf):6,} features")
        return gdf
    except Exception as e:
        print(f"{infra_type:20s}: Error - {e}")
        print(f"  Trying alternative read method...")
        try:
            import pyarrow.parquet as pq
            from shapely import wkb
            
            table = pq.read_table(str(file_path))
            df = table.to_pandas()
            df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(bytes(x)))
            
            geo_metadata = table.schema.pandas_metadata.get('geo', {}) if hasattr(table.schema, 'pandas_metadata') else {}
            crs = geo_metadata.get('crs', 'EPSG:4326')
            
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
            print(f"{infra_type:20s}: {len(gdf):6,} features (alternative method)")
            return gdf
        except Exception as e2:
            print(f"{infra_type:20s}: Failed with alternative method - {e2}")
            return None


def reproject_infrastructure_dict(infrastructure, target_crs='EPSG:3035'):
    """
    Reproject all GeoDataFrames in infrastructure dictionary to target CRS.
    
    Parameters:
    -----------
    infrastructure : dict
        Dictionary of GeoDataFrames
    target_crs : str
        Target CRS (default: 'EPSG:3035')
    
    Returns:
    --------
    dict
        Dictionary with reprojected GeoDataFrames
    """
    if len(infrastructure) == 0:
        return infrastructure
    
    for name, gdf in infrastructure.items():
        if gdf.crs is None:
            gdf.set_crs('EPSG:4326', inplace=True)
        if str(gdf.crs) != target_crs:
            infrastructure[name] = gdf.to_crs(target_crs)
    
    return infrastructure


def merge_risk_data_preserve_geometry(base_gdf, risk_df, risk_cc_df=None, id_col='id'):
    """
    Merge risk data with infrastructure while preserving GeoDataFrame structure.
    
    Parameters:
    -----------
    base_gdf : GeoDataFrame
        Base infrastructure data
    risk_df : DataFrame
        Risk data to merge
    risk_cc_df : DataFrame, optional
        Climate change risk data
    id_col : str
        Column name for joining (default: 'id')
    
    Returns:
    --------
    GeoDataFrame
        Merged data preserving geometry
    """
    # Store original CRS
    original_crs = base_gdf.crs
    
    # Merge risk data
    result = base_gdf.merge(risk_df, left_on=id_col, right_on=id_col, how='left')
    
    # Merge climate change data if provided
    if risk_cc_df is not None:
        result = result.merge(risk_cc_df, on=id_col, how='left', suffixes=('', '_cc'))
    
    # Ensure it's still a GeoDataFrame with valid geometry
    if not isinstance(result, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result, geometry='geometry', crs=original_crs)
    
    return result


def format_log_colorbar(colorbar, axis='y'):
    """
    Format colorbar with superscript powers of 10.
    
    Parameters:
    -----------
    colorbar : matplotlib.colorbar.Colorbar
        Colorbar to format
    axis : str
        Which axis to format ('x' or 'y')
    
    Returns:
    --------
    None
        Modifies colorbar in place
    """
    if colorbar is None:
        return
    
    ax = colorbar.ax
    tick_locs = ax.get_yticks() if axis == 'y' else ax.get_xticks()
    superscripts = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    labels = [f'10{str(int(val)).translate(superscripts)}' if not np.isnan(val) else ''
              for val in tick_locs]

    if axis == 'y':
        ax.set_yticks(tick_locs)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(labels)


# ── Public aliases (non-underscored names for notebook-facing helpers) ────────

pick_existing = _pick_existing
timed_load = _timed_load
coerce_loaded_geo = _coerce_loaded_geo
bounds_with_padding = _bounds_with_padding
infer_and_fix_crs = _infer_and_fix_crs
read_parquet_as_gdf = _read_parquet_as_gdf
safe_read_parquet_any = _safe_read_parquet_any
compute_disruption_cost_eur = _compute_disruption_cost_eur
norm_infra_key_by_type = _norm_infra_key_by_type
ensure_gdf_crs = _ensure_gdf_crs
zoom_to_selected_nuts2 = _zoom_to_selected_nuts2
