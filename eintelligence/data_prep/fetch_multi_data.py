from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Sequence
from shapely.geometry import Point, mapping, shape, box
from pystac_client import Client
import planetary_computer

stac_api_url = "https://planetarycomputer.microsoft.com/api/stac/v1"

def search_s2_items(
        aoi_geojson_or_geom,
        start_date: str | date,
        end_date: str | date,
        max_cloud: int = 20,
        limit: Optional[int] = None,
        same_mgrs_tile: bool = True
) -> List:
    
    """Search for Sentinel-2 items in a given area and date range.

    Args:
        stac_api_url (str): URL of the STAC API.
        aoi_geojson: GeoJSON geometry of the area of interest.
        start_date (str | date): Start date for the search (YYYY-MM-DD).
        end_date (str | date): End date for the search (YYYY-MM-DD).
        max_cloud (int, optional): Maximum cloud cover percentage. Defaults to 20.
        limit (Optional[int], optional): Maximum number of items to return. Defaults to None.
        same_mgrs_tile (bool, optional): If True, only return items from the same MGRS tile. Defaults to True.

    Returns:
        List: List of STAC items matching the search criteria.
    """
    catalog = Client.open(stac_api_url, modifier=planetary_computer.sign_inplace)
    
    geom = (aoi_geojson_or_geom.get("geometry") if isinstance(aoi_geojson_or_geom, dict)
            and "type" in aoi_geojson_or_geom and aoi_geojson_or_geom["type"] != "Polygon"
            else aoi_geojson_or_geom)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geom,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(search.items())
    if not items:
        return []
    
    if same_mgrs_tile:
        counts = {}
        for item in items:
            t = item.properties.get("s2:mgrs_tile")
            counts[t] = counts.get(t, 0) + 1
        winner = max(counts, key=counts.get)
        items = [item for item in items if item.properties.get("s2:mgrs_tile") == winner]

    # Sort by time
    items.sort(key=lambda i: i.properties.get("datetime"))

    if limit: items = items[:limit]
    return items

def search_s1_items(
        aoi_geojson_or_geom,
        start_date: str | date,
        end_date: str | date,
        orbit: Optional[str] = None,
        limit: Optional[int] = None,
        widen_days: int = 14,
    ):
    """
    Search Sentinel-1 **RTC** items over AOI/time. Returns items sorted by datetime.

    Notes:
      - Queries the `sentinel-1-rtc` collection (geocoded COGs with valid CRS/transform).
      - Keeps the IW instrument mode filter.
      - Optional orbit filter: 'ascending' or 'descending'.
      - Expands the time window by `widen_days` if initial search yields no items.
    """
    catalog = Client.open(stac_api_url, modifier=planetary_computer.sign_inplace)

    # Accept either a GeoJSON Feature/Geometry or a bare geometry dict
    if isinstance(aoi_geojson_or_geom, dict):
        geom = aoi_geojson_or_geom.get("geometry", aoi_geojson_or_geom)
    else:
        geom = aoi_geojson_or_geom

    def _query(range_start, range_end):
        q = {"sar:instrument_mode": {"eq": "IW"}}
        if orbit:
            q["sat:orbit_state"] = {"eq": orbit}  # 'ascending' or 'descending'
        return list(
            catalog.search(
                collections=["sentinel-1-rtc"],
                intersects=geom,
                datetime=f"{range_start}/{range_end}",
                query=q,
            ).items()
        )

    # 1) Try the requested range
    items = _query(start_date, end_date)

    # 2) Widen the window if needed
    if not items and widen_days > 0:
        start_dt = datetime.fromisoformat(str(start_date))
        end_dt   = datetime.fromisoformat(str(end_date))
        wstart   = (start_dt - timedelta(days=widen_days)).date().isoformat()
        wend     = (end_dt   + timedelta(days=widen_days)).date().isoformat()
        items    = _query(wstart, wend)

    # Nothing found
    if not items:
        return []

    # Sort + limit
    items.sort(key=lambda i: i.properties.get("datetime"))
    if limit:
        items = items[:limit]
    return items

# def search_s1_items(
#         aoi_geojson_or_geom,
#         start_date: str | date,
#         end_date: str | date,
#         orbit: Optional[str] = None,
#         limit: Optional[int] = None,
#         widen_days: int = 14,
#     ):

#     """
#     Search Sentinel-1 GRD IW items over AOI/time. Returns items sorted by datetime.
#     """
#     catalog = Client.open(stac_api_url, modifier=planetary_computer.sign_inplace)
#     geom = (aoi_geojson_or_geom.get("geometry") if isinstance(aoi_geojson_or_geom, dict)
#             and "type" in aoi_geojson_or_geom and aoi_geojson_or_geom["type"] != "Polygon"
#             else aoi_geojson_or_geom)
    
#     def _query(range_start, range_end, collections, with_grd_filter=True):
#         q = {"sar:instrument_mode": {"eq": "IW"}}
#         if with_grd_filter:
#             q["sar:product_type"] = {"eq": "GRD"}
#         if orbit:
#             q["sat:orbit_state"] = {"eq": orbit}
#         return list(catalog.search(
#             collections=collections,
#             intersects=geom,
#             datetime=f"{range_start}/{range_end}",
#             query=q
#         ).items())
    
#     # try GRD + filters
#     items = _query(start_date, end_date, ["sentinel-1-grd"], with_grd_filter=True)
#     # relax product filter
#     if not items:
#         items = _query(start_date, end_date, ["sentinel-1-grd"], with_grd_filter=False)
#     # widen the window
#     if not items and widen_days > 0:
#         start_dt = datetime.fromisoformat(start_date)
#         end_dt   = datetime.fromisoformat(end_date)
#         wstart   = (start_dt - timedelta(days=widen_days)).date().isoformat()
#         wend     = (end_dt   + timedelta(days=widen_days)).date().isoformat()
#         items    = _query(wstart, wend, ["sentinel-1-grd"], with_grd_filter=False)
#     # try RTC as a last resort (coverage is limited regionally)
#     if not items:
#         items = _query(start_date, end_date, ["sentinel-1-rtc"], with_grd_filter=False)

#     if not items:
#         return []

#     items.sort(key=lambda i: i.properties.get("datetime"))
#     if limit:
#         items = items[:limit]
#     return items
    