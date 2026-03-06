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
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    labels = [f'10{str(int(val)).translate(superscripts)}' if not np.isnan(val) else '' 
              for val in tick_locs]
    
    if axis == 'y':
        ax.set_yticklabels(labels)
    else:
        ax.set_xticklabels(labels)
