"""
TENT Corridors Disruption Analysis
===================================
Standalone Python script equivalent of TENT_corridors_disruption_analysis.ipynb.
Analyses RIVER FLOOD and COASTAL FLOOD risk for road and rail infrastructure
on TEN-T corridors.  All maps and bar plots are produced for each hazard type.

Outputs (saved to OUT_DIR):
  Maps  [hazard = river | coastal]
  ----------------------------------
  01_road_rail_flows_by_corridor.png
  02_flows_by_commodity_rail.png
  02_flows_by_commodity_road.png
  03_direct_damage_RP100_<hazard>.png
  04_EAD_<hazard>.png
  05_EAD_future_climate_<hazard>.png
  06_disruption_days_RP100_<hazard>.png

  Bar plots  [hazard = river | coastal]
  --------------------------------------
  bar_affected_length_by_RP_<mode>_<hazard>.png
  bar_affected_flow_by_RP_commodity_<mode>_<hazard>.png

  Data tables
  -----------
  rail_disruption_summary_river.parquet
  rail_disruption_summary_coastal.parquet
  road_disruption_summary_river.parquet

Coastal flood data sources
  Baseline : rail_coastal_TENT_risk.parquet
  Future   : rail_coastal_2050_SSP245_TENT_risk.parquet
           : rail_coastal_2050_SSP585_TENT_risk.parquet
           : rail_coastal_2100_SSP245_TENT_risk.parquet
           : rail_coastal_2100_SSP585_TENT_risk.parquet
"""

import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

CODE_DIR     = 'soge-home/mistral/miraca/'
HAZARD_DIR   = CODE_DIR.parent / "incoming_data" / "spatial_data" / "direct_damages_elco" / "Hazards"
RISK_DIR     = CODE_DIR.parent / "incoming_data" / "spatial_data" / "direct_damages_elco" / "Impacts"
OUT_DIR      = CODE_DIR.parent / "incoming_data" / "spatial_data" / "direct_damages_elco" / "tent_corridors_disruption_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COUNTRIES_SHP = CODE_DIR / "incoming_data" / "spatial_data" / "admin" / "ne_10m" / "ne_10m_admin_0_countries.shp"

# Infrastructure parquet files
INPUTS_DIR   = CODE_DIR.parent / "processed_data" / "europe_corridors_with_flows"
INFRA_FILES = {
    "railway_edges": INPUTS_DIR / "europe_railway_edges_corridors_with_flows.parquet",
    "road_edges":    INPUTS_DIR / "europe_road_edges_corridors_with_flows.parquet",
}

# Risk files — River flood (current + climate change)
RAIL_RISK_PATH    = RISK_DIR / "rail_risk.parquet"
RAIL_RISK_CC_PATH = RISK_DIR / "rail_risk_CC.parquet"
ROAD_RISK_PATH    = RISK_DIR / "road_risk.parquet"
ROAD_RISK_CC_PATH = RISK_DIR / "road_risk_CC.parquet"

# Risk files — Coastal flood (current + future SSP245/SSP585 × 2050/2100)
# Only rail files are provided for coastal flooding
RAIL_COASTAL_RISK_PATH = RISK_DIR / "rail_coastal_TENT_risk.parquet"

# Future coastal risk files keyed by scenario label
RAIL_COASTAL_CC_FILES = {
    'SSP245_2050': RISK_DIR / "rail_coastal_2050_SSP245_TENT_risk.parquet",
    'SSP585_2050': RISK_DIR / "rail_coastal_2050_SSP585_TENT_risk.parquet",
    'SSP245_2100': RISK_DIR / "rail_coastal_2100_SSP245_TENT_risk.parquet",
    'SSP585_2100': RISK_DIR / "rail_coastal_2100_SSP585_TENT_risk.parquet",
}
COASTAL_CC_SCENARIOS = list(RAIL_COASTAL_CC_FILES.keys())

# ============================================================================
# TEN-T CORRIDOR CONFIGURATION
# ============================================================================

CORRIDOR_ORDER = ['A', 'B', 'C', 'E', 'G', 'I', 'J', 'K', 'L', 'U']
CORRIDOR_NAMES = {
    'A': 'Baltic-Adriatic',
    'B': 'North Sea-Baltic',
    'C': 'Mediterranean',
    'E': 'Scandinavian-Mediterranean',
    'G': 'Atlantic',
    'I': 'Rhine-Danube',
    'J': 'Baltic-Aegean',
    'K': 'W. Balkans-E. Mediterranean',
    'L': 'North Sea-Alpine',
    'U': 'Not-in-corridor',
}
CORRIDOR_PALETTE = [
    '#0080C0', '#E91E8C', '#00A651', '#FF69B4', '#FFD700',
    '#00BFFF', '#8B4789', '#8B4513', '#228B22', '#b3b3b3',
]
CORRIDOR_COLORS = dict(zip(CORRIDOR_ORDER, CORRIDOR_PALETTE))

RETURN_PERIODS = [10, 20, 30, 40, 50, 75, 100, 200, 500]
CC_SCENARIOS   = ['1.5C', '2.0C', '3.0C', '4.0C']

# Map extent (EPSG:3035)
MAP_XLIM = (2_500_000, 6_500_000)
MAP_YLIM = (1_400_000, 5_500_000)

# ============================================================================
# DAMAGE RATIO → DAYS OF DISRUPTION
# ============================================================================

def damage_ratio_to_days(damage_ratio: float) -> int:
    """
    Classify a damage ratio (0–100 %) into disruption days.

      < 10  → 0  days  (no damage)
      < 25  → 1  day   (mild)
      < 50  → 3  days  (medium)
      < 75  → 7  days  (severe)
      ≥ 75  → 31 days  (collapse)
    """
    if damage_ratio < 10:
        return 0
    elif damage_ratio < 25:
        return 1
    elif damage_ratio < 50:
        return 3
    elif damage_ratio < 75:
        return 7
    else:
        return 31

DISRUPTION_LABEL = {0: 'No damage', 1: 'Mild (1 d)', 3: 'Medium (3 d)',
                    7: 'Severe (7 d)', 31: 'Collapse (31 d)'}
DISRUPTION_COLORS = {0: '#d4edda', 1: '#fff3cd', 3: '#ffc107', 7: '#fd7e14', 31: '#dc3545'}

# ============================================================================
# HELPERS
# ============================================================================

def _extract_primary_corridor(corridors_str) -> str:
    if pd.isna(corridors_str) or str(corridors_str).strip() in ('', 'NULL', 'None', 'NA'):
        return 'U'
    for ch in str(corridors_str).upper():
        if ch in CORRIDOR_COLORS:
            return ch
    return 'U'

def _load_countries():
    try:
        gdf = gpd.read_file(COUNTRIES_SHP)
        gdf = gdf[gdf['CONTINENT'] == 'Europe'].to_crs('EPSG:3035')
        print(f"  Loaded {len(gdf)} European countries")
        return gdf
    except Exception as e:
        print(f"  Could not load country boundaries: {e}")
        return None

def _load_infra(path: Path, label: str) -> gpd.GeoDataFrame | None:
    if not path.exists():
        print(f"  [SKIP] {label}: file not found ({path})")
        return None
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf.set_crs('EPSG:4326', inplace=True)
    gdf = gdf.to_crs('EPSG:3035')
    if 'CORRIDORS' in gdf.columns:
        gdf['primary_corridor'] = gdf['CORRIDORS'].apply(_extract_primary_corridor)
    else:
        gdf['primary_corridor'] = 'U'
    print(f"  {label}: {len(gdf)} features")
    return gdf

def _setup_ax(ax, countries, title=''):
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    if countries is not None:
        countries.boundary.plot(ax=ax, color='#888888', linewidth=0.4, zorder=1)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_axis_off()

def _corridor_legend_handles():
    return [mlines.Line2D([], [], color=CORRIDOR_COLORS[c], lw=2.5,
                          label=CORRIDOR_NAMES[c])
            for c in CORRIDOR_ORDER if c != 'U']

def _flow_to_lw(value, p05, p95, lw_min=0.4, lw_max=4.0):
    if p95 <= p05:
        return lw_min
    t = np.clip((value - p05) / (p95 - p05), 0, 1)
    return lw_min + t * (lw_max - lw_min)

def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")

def _merge_risk(infra_gdf: gpd.GeoDataFrame,
                risk_df: pd.DataFrame,
                risk_cc_df: pd.DataFrame | None) -> gpd.GeoDataFrame:
    """Left-join risk tables onto the infrastructure GeoDataFrame by index."""
    gdf = infra_gdf.copy()
    if risk_df is not None and not risk_df.empty:
        gdf = gdf.merge(risk_df, left_index=True, right_index=True, how='left', suffixes=('', '_risk'))
    if risk_cc_df is not None and not risk_cc_df.empty:
        gdf = gdf.merge(risk_cc_df, left_index=True, right_index=True, how='left', suffixes=('', '_cc'))
    return gdf

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n" + "="*70)
print("1. LOADING DATA")
print("="*70)

countries = _load_countries()

rail = _load_infra(INFRA_FILES['railway_edges'], 'railway_edges')
road = _load_infra(INFRA_FILES['road_edges'],    'road_edges')

# Risk data
def _load_parquet_safe(path: Path, label: str):
    if not path.exists():
        print(f"  [SKIP] {label} not found ({path})")
        return None
    df = pd.read_parquet(path)
    print(f"  {label}: {len(df)} rows, columns: {list(df.columns[:8])}...")
    return df

# --- River flood ---
print("\n  [River flood]")
rail_risk         = _load_parquet_safe(RAIL_RISK_PATH,    'rail_risk (river)')
rail_risk_cc      = _load_parquet_safe(RAIL_RISK_CC_PATH, 'rail_risk_CC (river)')
road_risk         = _load_parquet_safe(ROAD_RISK_PATH,    'road_risk (river)')
road_risk_cc      = _load_parquet_safe(ROAD_RISK_CC_PATH, 'road_risk_CC (river)')

# --- Coastal flood ---
print("\n  [Coastal flood]")
rail_coastal = _load_parquet_safe(RAIL_COASTAL_RISK_PATH, 'rail_risk (coastal baseline)')

# Load all four future coastal scenarios and merge into a single wide DataFrame
# Each file is expected to contain EAD and damage columns; we suffix-join by scenario label.
rail_coastal_cc_combined: pd.DataFrame | None = None
for _scen, _path in RAIL_COASTAL_CC_FILES.items():
    _df = _load_parquet_safe(_path, f'rail_risk_CC (coastal {_scen})')
    if _df is None:
        continue
    # Rename all non-index columns to include the scenario suffix so they don't collide
    _df = _df.rename(columns={c: f'{c}_{_scen}' if not c.endswith(_scen) else c
                               for c in _df.columns})
    if rail_coastal_cc_combined is None:
        rail_coastal_cc_combined = _df
    else:
        rail_coastal_cc_combined = rail_coastal_cc_combined.join(_df, how='outer', rsuffix=f'_dup_{_scen}')
        # Drop accidental duplicate columns
        dup_cols = [c for c in rail_coastal_cc_combined.columns if '_dup_' in c]
        rail_coastal_cc_combined.drop(columns=dup_cols, inplace=True)
# No road coastal files provided — road coastal will be skipped gracefully

# Merge risk onto geometry
rail_wr_river   = _merge_risk(rail, rail_risk,    rail_risk_cc)              if rail is not None else None
road_wr_river   = _merge_risk(road, road_risk,    road_risk_cc)              if road is not None else None
rail_wr_coastal = _merge_risk(rail, rail_coastal, rail_coastal_cc_combined)  if rail is not None else None
road_wr_coastal = None   # no coastal road data

# Bundle into a dict keyed by hazard name for looped processing
# Structure: HAZARD_DATA[hazard] = {'rail': gdf_or_None, 'road': gdf_or_None,
#                                    'cc_scenarios': [...], 'label': str}
HAZARD_DATA = {
    'river': {
        'rail':         rail_wr_river,
        'road':         road_wr_river,
        'cc_scenarios': CC_SCENARIOS,           # ['1.5C', '2.0C', '3.0C', '4.0C']
        'ead_cc_prefix': 'EAD_',                # EAD_{scenario} columns
        'label':        'River Flood',
    },
    'coastal': {
        'rail':         rail_wr_coastal,
        'road':         road_wr_coastal,
        'cc_scenarios': COASTAL_CC_SCENARIOS,   # ['SSP245_2050','SSP585_2050','SSP245_2100','SSP585_2100']
        'ead_cc_prefix': 'EAD_',
        'label':        'Coastal Flood',
    },
}

# ============================================================================
# COMPUTE DAMAGE RATIO AND DISRUPTION DAYS
# ============================================================================

def _add_damage_ratio_and_days(gdf: gpd.GeoDataFrame,
                                rp: int = 100,
                                unit_cost_per_m: float = 5_000.0) -> gpd.GeoDataFrame:
    """
    Compute damage ratio (%) and disruption days for a given return period.
    damage_ratio = mean_damage_RP / (exposure_RP * unit_cost_per_m) * 100
    """
    gdf = gdf.copy()
    exp_col = f'exposure_{rp}'
    dmg_col = f'mean_damage_{rp}'
    if exp_col in gdf.columns and dmg_col in gdf.columns:
        asset_value = gdf[exp_col].fillna(0) * unit_cost_per_m
        gdf['damage_ratio'] = np.where(
            asset_value > 0,
            (gdf[dmg_col].fillna(0) / asset_value * 100).clip(0, 100),
            0.0
        )
        gdf['days_disrupted'] = gdf['damage_ratio'].apply(damage_ratio_to_days)
    else:
        gdf['damage_ratio']  = np.nan
        gdf['days_disrupted'] = 0
    return gdf

for _haz, _hdata in HAZARD_DATA.items():
    for _mode in ('rail', 'road'):
        if _hdata[_mode] is not None:
            _hdata[_mode] = _add_damage_ratio_and_days(_hdata[_mode])

# Convenience aliases kept for MAP 1 & 2 (flow maps, hazard-independent)
rail_wr = HAZARD_DATA['river']['rail']
road_wr = HAZARD_DATA['river']['road']

# ============================================================================
# MAP 1 — ROAD AND RAIL FLOWS (corridor colour, flow thickness)
# ============================================================================

print("\n" + "="*70)
print("MAP 1 — Road & Rail flows by corridor and flow value")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(26, 14))

for ax, gdf, flow_col, title, mode in [
    (axes[0], rail_wr, 'flow_rail_freight', 'Rail Freight Flows', 'rail'),
    (axes[1], road_wr, 'flow_road_freight', 'Road Freight Flows', 'road'),
]:
    _setup_ax(ax, countries, title)
    if gdf is None or flow_col not in gdf.columns:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=13)
        continue

    gdf_f = gdf[gdf[flow_col] > 0].copy()
    if gdf_f.empty:
        continue

    p05  = gdf_f[flow_col].quantile(0.05)
    p95  = gdf_f[flow_col].quantile(0.95)
    gdf_f['lw'] = gdf_f[flow_col].apply(lambda v: _flow_to_lw(v, p05, p95))

    for corr in CORRIDOR_ORDER:
        sub = gdf_f[gdf_f['primary_corridor'] == corr]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            gpd.GeoDataFrame([row], crs=gdf.crs).plot(
                ax=ax, color=CORRIDOR_COLORS[corr],
                linewidth=row['lw'], alpha=0.75, zorder=2
            )

    # Legends
    leg1 = ax.legend(handles=_corridor_legend_handles(),
                     title='TEN-T Corridor', loc='lower left',
                     fontsize=7, title_fontsize=8)
    ax.add_artist(leg1)
    lw_vals  = [p05, np.percentile(gdf_f[flow_col], 50), p95]
    lw_hdls  = [mlines.Line2D([], [], color='gray',
                               lw=_flow_to_lw(v, p05, p95),
                               label=f'{v/1e3:.0f} kT/yr') for v in lw_vals]
    ax.legend(handles=lw_hdls, title='Flow', loc='upper left',
              fontsize=7, title_fontsize=8)

plt.suptitle('TEN-T Road & Rail Freight Flows', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
_save(fig, '01_road_rail_flows_by_corridor.png')

# ============================================================================
# MAP 2 — FLOWS BY COMMODITY
# ============================================================================

print("\n" + "="*70)
print("MAP 2 — Flows by commodity")
print("="*70)

for gdf, mode_label, fname_suffix in [
    (rail_wr, 'Rail', 'rail'),
    (road_wr, 'Road', 'road'),
]:
    if gdf is None:
        print(f"  [SKIP] {mode_label}: no data")
        continue

    comm_cols = [c for c in gdf.columns
                 if c.startswith('flow_') and 'commodity' in c.lower()
                 or c.startswith('road_share_') or c.startswith('share_')]
    # Also accept columns like flow_<commodity>
    if not comm_cols:
        comm_cols = [c for c in gdf.columns
                     if c.startswith('flow_') and c not in
                     ('flow_rail_freight', 'flow_road_freight',
                      'flow_rail_passenger', 'flow_road_passenger')]

    if not comm_cols:
        print(f"  [SKIP] {mode_label}: no commodity columns found")
        continue

    n = len(comm_cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 8 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(comm_cols):
        ax = axes[i]
        label = col.replace('flow_', '').replace('_', ' ').title()
        _setup_ax(ax, countries, label)
        gdf_f = gdf[gdf[col].fillna(0) > 0].copy()
        if gdf_f.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        p05 = gdf_f[col].quantile(0.05)
        p95 = gdf_f[col].quantile(0.95)
        gdf_f['lw'] = gdf_f[col].apply(lambda v: _flow_to_lw(v, p05, p95))
        for corr in CORRIDOR_ORDER:
            sub = gdf_f[gdf_f['primary_corridor'] == corr]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                gpd.GeoDataFrame([row], crs=gdf.crs).plot(
                    ax=ax, color=CORRIDOR_COLORS[corr],
                    linewidth=row['lw'], alpha=0.75, zorder=2)

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{mode_label} Freight Flows by Commodity', fontsize=15,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, f'02_flows_by_commodity_{fname_suffix}.png')

# ============================================================================
# MAP 3 — DIRECT DAMAGES (DAMAGE RATIO) FOR RP100  [per hazard]
# ============================================================================

print("\n" + "="*70)
print("MAP 3 — Direct damage ratio for RP100 (river & coastal)")
print("="*70)

for haz, hdata in HAZARD_DATA.items():
    fig, axes = plt.subplots(1, 2, figsize=(26, 14))
    for ax, gdf, mode_label in [
        (axes[0], hdata['rail'], 'Rail'),
        (axes[1], hdata['road'], 'Road'),
    ]:
        title = f'{mode_label} — Damage Ratio RP100 (%)'
        _setup_ax(ax, countries, title)
        if gdf is None or 'damage_ratio' not in gdf.columns:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=13)
            continue
        gdf_f = gdf[gdf['damage_ratio'].fillna(0) > 0].copy()
        if gdf_f.empty:
            ax.text(0.5, 0.5, 'No exposed assets', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        gdf_f.plot(ax=ax, column='damage_ratio', cmap='Reds',
                   linewidth=1.5, alpha=0.8, legend=True,
                   legend_kwds={'label': 'Damage ratio (%)', 'shrink': 0.7},
                   vmin=0, vmax=100, zorder=2)

    plt.suptitle(f'Direct Damage Ratio — RP100 {hdata["label"]}', fontsize=16,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, f'03_direct_damage_RP100_{haz}.png')

# ============================================================================
# MAP 4 — EAD  [per hazard]
# ============================================================================

print("\n" + "="*70)
print("MAP 4 — Expected Annual Damage (EAD) (river & coastal)")
print("="*70)

for haz, hdata in HAZARD_DATA.items():
    fig, axes = plt.subplots(1, 2, figsize=(26, 14))
    for ax, gdf, mode_label in [
        (axes[0], hdata['rail'], 'Rail'),
        (axes[1], hdata['road'], 'Road'),
    ]:
        title = f'{mode_label} — Expected Annual Damage (EAD)'
        _setup_ax(ax, countries, title)
        if gdf is None or 'EAD' not in gdf.columns:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=13)
            continue
        gdf_f = gdf[gdf['EAD'].fillna(0) > 0].copy()
        if gdf_f.empty:
            continue
        gdf_f['log_EAD'] = np.log10(gdf_f['EAD'])
        vmin = gdf_f['log_EAD'].quantile(0.05)
        vmax = gdf_f['log_EAD'].quantile(0.95)
        gdf_f.plot(ax=ax, column='log_EAD', cmap='Purples',
                   linewidth=1.5, alpha=0.8, legend=True,
                   legend_kwds={'label': 'EAD (€/yr, log₁₀)', 'shrink': 0.7},
                   vmin=vmin, vmax=vmax, zorder=2)

    plt.suptitle(f'Expected Annual Damage (EAD) — {hdata["label"]}', fontsize=16,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, f'04_EAD_{haz}.png')

# ============================================================================
# MAP 5 — EAD VARIATION FOR FUTURE CLIMATE  [per hazard]
# ============================================================================

print("\n" + "="*70)
print("MAP 5 — EAD variation for future climate scenarios (river & coastal)")
print("="*70)

for haz, hdata in HAZARD_DATA.items():
    scenarios   = hdata['cc_scenarios']
    n_scenarios = len(scenarios)
    fig, axes   = plt.subplots(n_scenarios, 2,
                               figsize=(26, 14 * n_scenarios),
                               squeeze=False)

    for row_idx, scenario in enumerate(scenarios):
        ead_col   = f"{hdata['ead_cc_prefix']}{scenario}"
        ratio_col = f'EAD_ratio_{scenario}'

        for col_idx, (gdf, mode_label) in enumerate([
            (hdata['rail'], 'Rail'), (hdata['road'], 'Road')
        ]):
            ax = axes[row_idx, col_idx]
            _setup_ax(ax, countries, f'{mode_label} — EAD ratio {scenario}')

            if gdf is None or ead_col not in gdf.columns or 'EAD' not in gdf.columns:
                ax.text(0.5, 0.5, f'No data for {ead_col}', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            gdf_f = gdf[gdf['EAD'].fillna(0) > 0].copy()
            if gdf_f.empty:
                continue
            gdf_f[ratio_col] = gdf_f[ead_col].fillna(0) / gdf_f['EAD'].replace(0, np.nan)

            gdf_f.plot(ax=ax, column=ratio_col, cmap='YlOrRd',
                       linewidth=1.5, alpha=0.8, legend=True,
                       legend_kwds={'label': 'EAD ratio vs baseline', 'shrink': 0.7},
                       vmin=1, vmax=5, zorder=2)

            avg = gdf_f[ratio_col].mean()
            ax.text(0.02, 0.98, f'Mean: {avg:.2f}× baseline',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'EAD Variation — {hdata["label"]} Future Climate Scenarios',
                 fontsize=16, fontweight='bold', y=1.005)
    plt.tight_layout()
    _save(fig, f'05_EAD_future_climate_{haz}.png')

# ============================================================================
# MAP 6 — DISRUPTION DAYS PER ASSET (RP100)  [per hazard]
# ============================================================================

print("\n" + "="*70)
print("MAP 6 — Duration of disruption (days) per asset for RP100 (river & coastal)")
print("="*70)

for haz, hdata in HAZARD_DATA.items():
    fig, axes = plt.subplots(1, 2, figsize=(26, 14))
    for ax, gdf, mode_label in [
        (axes[0], hdata['rail'], 'Rail'),
        (axes[1], hdata['road'], 'Road'),
    ]:
        title = f'{mode_label} — Disruption Duration RP100'
        _setup_ax(ax, countries, title)
        if gdf is None or 'days_disrupted' not in gdf.columns:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=13)
            continue
        for days, color in DISRUPTION_COLORS.items():
            sub = gdf[gdf['days_disrupted'] == days]
            if sub.empty:
                continue
            sub.plot(ax=ax, color=color, linewidth=1.5 if days > 0 else 0.5,
                     alpha=0.8 if days > 0 else 0.3, zorder=2 if days > 0 else 1)
        handles = [mpatches.Patch(facecolor=DISRUPTION_COLORS[d],
                                   label=DISRUPTION_LABEL[d])
                   for d in sorted(DISRUPTION_COLORS)]
        ax.legend(handles=handles, title='Disruption class', loc='lower left',
                  fontsize=8, title_fontsize=9)

    plt.suptitle(f'Disruption Duration by Asset — RP100 {hdata["label"]}', fontsize=16,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, f'06_disruption_days_RP100_{haz}.png')

# ============================================================================
# MAP 6b — DISRUPTION DAYS BY COMMODITY  [per hazard, per mode]
# Only edges that carry each commodity (flow_XXX > 0) are shown.
# Colour = disruption class derived from damage_ratio (same scale as Map 6).
# ============================================================================

print("\n" + "="*70)
print("MAP 6b — Disruption days by commodity (river & coastal)")
print("="*70)

def _commodity_cols(gdf: gpd.GeoDataFrame) -> list[str]:
    """Return flow_XXX columns, excluding known aggregate/passenger cols."""
    EXCLUDE = {
        'flow_rail_freight', 'flow_road_freight',
        'flow_rail_passenger', 'flow_road_passenger',
    }
    return [c for c in gdf.columns
            if c.startswith('flow_') and c not in EXCLUDE
            and pd.api.types.is_numeric_dtype(gdf[c])]

for haz, hdata in HAZARD_DATA.items():
    for mode_label, gdf in [('Rail', hdata['rail']), ('Road', hdata['road'])]:
        if gdf is None or 'days_disrupted' not in gdf.columns:
            print(f"  [SKIP] {mode_label} {haz}: no disruption data")
            continue

        comm_cols = _commodity_cols(gdf)
        if not comm_cols:
            print(f"  [SKIP] {mode_label} {haz}: no commodity flow columns found")
            continue

        n      = len(comm_cols)
        ncols  = min(3, n)
        nrows  = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(13 * ncols, 11 * nrows),
                                  squeeze=False)
        axes_flat = axes.flatten()

        for i, col in enumerate(comm_cols):
            ax    = axes_flat[i]
            comm_label = col.replace('flow_', '').replace('_', ' ').title()
            _setup_ax(ax, countries,
                      f'{mode_label} — {comm_label}\nDisruption days RP100 ({hdata["label"]})')

            # Only edges that carry this commodity
            gdf_c = gdf[gdf[col].fillna(0) > 0].copy()
            if gdf_c.empty:
                ax.text(0.5, 0.5, 'No flow', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11)
                continue

            # Use line thickness proportional to commodity flow
            p05 = gdf_c[col].quantile(0.05)
            p95 = gdf_c[col].quantile(0.95)
            gdf_c['lw'] = gdf_c[col].apply(lambda v: _flow_to_lw(v, p05, p95, 0.4, 3.5))

            # Plot by disruption class (ascending so damaged edges draw on top)
            for days in sorted(DISRUPTION_COLORS):
                sub = gdf_c[gdf_c['days_disrupted'] == days]
                if sub.empty:
                    continue
                for _, row in sub.iterrows():
                    gpd.GeoDataFrame([row], crs=gdf.crs).plot(
                        ax=ax,
                        color=DISRUPTION_COLORS[days],
                        linewidth=row['lw'],
                        alpha=0.85 if days > 0 else 0.25,
                        zorder=2 + days,   # damaged edges on top
                    )

            # Disruption class legend
            dis_handles = [
                mpatches.Patch(facecolor=DISRUPTION_COLORS[d], label=DISRUPTION_LABEL[d])
                for d in sorted(DISRUPTION_COLORS)
            ]
            leg1 = ax.legend(handles=dis_handles, title='Disruption',
                             loc='lower left', fontsize=7, title_fontsize=8)
            ax.add_artist(leg1)

            # Flow thickness legend
            lw_vals = [p05, np.percentile(gdf_c[col], 50), p95]
            lw_hdls = [mlines.Line2D([], [], color='#555555',
                                      lw=_flow_to_lw(v, p05, p95, 0.4, 3.5),
                                      label=f'{v/1e3:.0f} kT/yr')
                       for v in lw_vals if np.isfinite(v)]
            ax.legend(handles=lw_hdls, title='Flow', loc='upper left',
                      fontsize=7, title_fontsize=8)

        # Hide unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.suptitle(
            f'{mode_label} Disruption Days by Commodity — RP100 {hdata["label"]}',
            fontsize=15, fontweight='bold', y=1.005)
        plt.tight_layout()
        _save(fig, f'06b_disruption_days_by_commodity_{mode_label.lower()}_{haz}.png')

# ============================================================================
# BAR PLOTS — AFFECTED LENGTH (KM) BY RETURN PERIOD
# ============================================================================

print("\n" + "="*70)
print("BAR — Affected length (km) by return period")
print("="*70)

def _affected_length_by_rp(gdf: gpd.GeoDataFrame) -> dict[int, float]:
    """Sum of exposure_RP (metres → km) across all exposed assets."""
    result = {}
    for rp in RETURN_PERIODS:
        col = f'exposure_{rp}'
        if col in gdf.columns:
            result[rp] = gdf[col].fillna(0).sum() / 1000.0
        else:
            result[rp] = 0.0
    return result

for haz, hdata in HAZARD_DATA.items():
    for mode_label, gdf in [('Rail', hdata['rail']), ('Road', hdata['road'])]:
        fname = f'bar_affected_length_by_RP_{mode_label.lower()}_{haz}.png'
        if gdf is None:
            print(f"  [SKIP] {mode_label} {haz}")
            continue

        lengths = _affected_length_by_rp(gdf)
        rps  = [rp for rp in RETURN_PERIODS if lengths[rp] > 0]
        vals = [lengths[rp] for rp in rps]

        if not rps:
            print(f"  [SKIP] {mode_label} {haz}: no exposure data")
            continue

        bar_color = '#2c7fb8' if haz == 'river' else '#17a2b8'
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar([str(rp) for rp in rps], vals, color=bar_color, edgecolor='white')
        ax.bar_label(bars, fmt='%.0f km', padding=4, fontsize=9)
        ax.set_xlabel('Return Period (years)', fontsize=12)
        ax.set_ylabel('Affected Length (km)', fontsize=12)
        ax.set_title(f'{mode_label} — Affected Length by RP ({hdata["label"]}, present climate)',
                     fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        _save(fig, fname)

# ============================================================================
# BAR PLOTS — AFFECTED FLOW (TONS) BY RP AND COMMODITY
# ============================================================================

print("\n" + "="*70)
print("BAR — Affected flow (tons) by return period and commodity")
print("="*70)

def _affected_flow_by_rp_commodity(gdf: gpd.GeoDataFrame,
                                    commodity_flow_cols: list[str]) -> pd.DataFrame:
    """
    For each RP, sum flows on exposed (exposure_RP > 0) edges per commodity.
    Returns a DataFrame (rows=RP, columns=commodity).
    """
    records = []
    for rp in RETURN_PERIODS:
        exp_col = f'exposure_{rp}'
        if exp_col not in gdf.columns:
            continue
        exposed = gdf[gdf[exp_col].fillna(0) > 0]
        row = {'return_period': rp}
        for col in commodity_flow_cols:
            row[col] = exposed[col].fillna(0).sum() if col in exposed.columns else 0.0
        records.append(row)
    return pd.DataFrame(records).set_index('return_period')

for haz, hdata in HAZARD_DATA.items():
    for mode_label, gdf in [('Rail', hdata['rail']), ('Road', hdata['road'])]:
        fname = f'bar_affected_flow_by_RP_commodity_{mode_label.lower()}_{haz}.png'
        if gdf is None:
            print(f"  [SKIP] {mode_label} {haz}")
            continue

        # Detect commodity flow columns
        comm_flow_cols = [c for c in gdf.columns
                          if ('commodity' in c.lower() or 'share_' in c)
                          and gdf[c].fillna(0).sum() > 0]
        if not comm_flow_cols:
            comm_flow_cols = [c for c in gdf.columns
                              if c.startswith('flow_')
                              and c not in ('flow_rail_freight', 'flow_road_freight',
                                            'flow_rail_passenger', 'flow_road_passenger')]
        if not comm_flow_cols:
            print(f"  [SKIP] {mode_label} {haz}: no commodity flow columns")
            continue

        flow_df = _affected_flow_by_rp_commodity(gdf, comm_flow_cols)
        if flow_df.empty or flow_df.sum().sum() == 0:
            print(f"  [SKIP] {mode_label} {haz}: zero commodity flows on exposed assets")
            continue

        flow_df   = flow_df.loc[:, (flow_df > 0).any()]
        n_comm    = len(flow_df.columns)
        x         = np.arange(len(flow_df.index))
        width     = 0.8 / n_comm
        cmap_comm = plt.cm.get_cmap('tab20', n_comm)

        fig, ax = plt.subplots(figsize=(max(12, 2 * len(flow_df.index)), 6))
        for i, col in enumerate(flow_df.columns):
            label = col.replace('flow_', '').replace('_', ' ').title()
            ax.bar(x + i * width - (n_comm - 1) * width / 2,
                   flow_df[col], width, label=label,
                   color=cmap_comm(i), edgecolor='white', alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([f'RP{rp}' for rp in flow_df.index], fontsize=10)
        ax.set_xlabel('Return Period', fontsize=12)
        ax.set_ylabel('Affected Flow (tons/yr)', fontsize=12)
        ax.set_title(
            f'{mode_label} — Affected Freight Flow by RP & Commodity ({hdata["label"]})',
            fontsize=13, fontweight='bold')
        ax.legend(title='Commodity', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v/1e6:.1f}MT' if v >= 1e6 else f'{v/1e3:.0f}kT'))
        ax.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        _save(fig, fname)

# ============================================================================
# SAVE DATA TABLES
# ============================================================================

print("\n" + "="*70)
print("SAVING DATA TABLES")
print("="*70)

for haz, hdata in HAZARD_DATA.items():
    all_scenarios = hdata['cc_scenarios']
    summary_cols  = (
        ['geometry', 'primary_corridor', 'damage_ratio', 'days_disrupted', 'EAD']
        + [f'exposure_{rp}'    for rp in RETURN_PERIODS]
        + [f'mean_damage_{rp}' for rp in RETURN_PERIODS]
        + [f"{hdata['ead_cc_prefix']}{s}" for s in all_scenarios]
    )
    for mode_label, gdf in [('rail', hdata['rail']), ('road', hdata['road'])]:
        if gdf is None:
            continue
        keep = [c for c in summary_cols if c in gdf.columns]
        fname = f'{mode_label}_disruption_summary_{haz}.parquet'
        out_path = OUT_DIR / fname
        gdf[keep].to_parquet(out_path, index=False)
        print(f"  Saved → {out_path}")

# ============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print(f"All outputs saved to: {OUT_DIR}")
print("="*70)
