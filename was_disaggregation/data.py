"""Lazy input handling and complete, calendar-aligned seasonal samples.

All meteorological arrays use T,Y,X. Seasons are identified by the year of
**their first month**; February 29 is omitted, including during generation.
"""
from __future__ import annotations

import calendar
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

VARIABLES = ("PRCP", "TMIN", "TMAX", "HUMIN", "HUMAX", "WIND", "SOLAR")
CANONICAL_UNITS = {"PRCP": "mm d-1", "TMIN": "degC", "TMAX": "degC",
                   "HUMIN": "%", "HUMAX": "%", "WIND": "m s-1", "SOLAR": "MJ m-2 d-1"}
_DIM_ALIASES = {"time": "T", "latitude": "Y", "lat": "Y", "longitude": "X", "lon": "X"}


def _rename_dimensions(obj):
    rename = {old: new for old, new in _DIM_ALIASES.items()
              if old in obj.dims and new not in obj.dims}
    return obj.rename(rename)


def _unit_key(unit):
    return (str(unit).strip().lower().replace("**", "").replace("^", "")
            .replace(" ", "").replace("_", "").replace("°", "deg"))


def _convert_units(da, name, unit_override=None, assume_units=False):
    units = unit_override if unit_override is not None else da.attrs.get("units")
    if not units:
        if not assume_units:
            raise ValueError(f"{name} has no units; supply unit_overrides={{'{name}': '<source units>'}}.")
        units = CANONICAL_UNITS[name]
    key = _unit_key(units)
    scale, offset = 1., 0.
    if name == "PRCP":
        if key in {"mmd-1", "mm/day", "mmday-1", "mm/d", "mm", "kgm-2d-1", "kgm-2day-1", "kg/m2/day"}:
            pass
        elif key in {"kgm-2s-1", "kg/m2/s", "mms-1", "mm/s"}:
            scale = 86400.
        elif key in {"m", "md-1", "m/day", "mday-1"}:
            scale = 1000.
        elif key in {"ms-1", "m/s"}:
            scale = 86400000.
        else:
            raise ValueError(f"Unsupported precipitation units {units!r}; expected daily amount or known water flux.")
    elif name in {"TMIN", "TMAX"}:
        if key in {"degc", "c", "celsius", "degreecelsius", "degreescelsius", "degreec", "degreesc"}:
            pass
        elif key in {"k", "kelvin", "degreekelvin", "degreeskelvin"}:
            offset = -273.15
        elif key in {"degf", "f", "fahrenheit"}:
            scale, offset = 5. / 9., -32. * 5. / 9.
        else:
            raise ValueError(f"Unsupported {name} units {units!r}.")
    elif name in {"HUMIN", "HUMAX"}:
        if key in {"%", "percent", "percentage", "pct"}:
            pass
        elif key in {"1", "fraction", "dimensionless", "0-1"}:
            scale = 100.
        else:
            raise ValueError(f"Unsupported relative humidity units {units!r}.")
    elif name == "WIND":
        if key in {"ms-1", "m/s", "msec-1", "meter/second", "metrespersecond"}:
            pass
        elif key in {"kmh-1", "km/h", "kph"}:
            scale = 1. / 3.6
        elif key in {"knot", "knots", "kt", "kts"}:
            scale = 0.5144444444444445
        else:
            raise ValueError(f"Unsupported wind units {units!r}; wind force categories cannot be treated as speed.")
    else:
        if key in {"mjm-2d-1", "mj/m2/day", "mjm-2day-1", "mjm-2", "mj/m2"}:
            pass
        elif key in {"wm-2", "w/m2"}:
            scale = 0.0864  # Daily mean flux -> daily radiant energy.
        elif key in {"jm-2", "j/m2", "jm-2d-1", "j/m2/day"}:
            scale = 1.e-6
        elif key in {"kjm-2", "kj/m2", "kjm-2d-1"}:
            scale = 1.e-3
        else:
            raise ValueError(f"Unsupported solar radiation units {units!r}.")
    attrs = dict(da.attrs)
    result = da if scale == 1. and offset == 0. else da * scale + offset
    result = result.rename(name)
    result.attrs = attrs | {"units": CANONICAL_UNITS[name], "source_units": str(units)}
    if assume_units and not da.attrs.get("units") and unit_override is None:
        result.attrs["units_assumed"] = "true"
    return result


def _validate_grid(obj, *, time=False):
    needed = ("T", "Y", "X") if time else ("Y", "X")
    for dim in needed:
        if dim not in obj.dims or dim not in obj.coords:
            raise ValueError(f"Missing required coordinate/dimension {dim!r}.")
        if obj[dim].dims != (dim,):
            raise ValueError("Only one-dimensional rectilinear grid coordinates are supported.")
        vals = obj[dim].values
        if len(vals) == 0:
            raise ValueError(f"Coordinate {dim} is empty.")
        if len(np.unique(vals)) != len(vals):
            raise ValueError(f"Duplicate {dim} coordinates are not allowed.")
        if dim != "T" and (not np.issubdtype(vals.dtype, np.number) or not np.isfinite(vals).all()):
            raise ValueError(f"{dim} coordinates must be finite numbers.")
    if time:
        # DataArray.T is its transpose, whereas Dataset.T may resolve a
        # coordinate. Bracket access is unambiguous for both container types.
        if not np.issubdtype(obj["T"].dtype, np.datetime64):
            raise ValueError("Only Gregorian datetime64 calendars are currently supported; convert explicitly first.")
        idx = pd.DatetimeIndex(obj["T"].values)
        if idx.hasnans or idx.normalize().has_duplicates:
            raise ValueError("T must contain unique daily dates without NaT.")


def canonicalize_observations(observations: xr.Dataset, unit_overrides=None,
                              assume_units=False) -> xr.Dataset:
    """Validate canonical variables, convert known units lazily, and mask invalid values.

    Unknown variables are rejected: map source variable names to canonical names
    before this function. Invalid individual values become NaN, not dry days.
    """
    if not isinstance(observations, xr.Dataset):
        raise TypeError("observations must be an xarray.Dataset.")
    obj = _rename_dimensions(observations)
    _validate_grid(obj, time=True)
    time_was_sorted = not pd.DatetimeIndex(obj.T.values).is_monotonic_increasing
    unknown = set(obj.data_vars) - set(VARIABLES)
    if unknown:
        raise ValueError(f"Unknown observation variables {sorted(unknown)}; use canonical names {VARIABLES}.")
    if "PRCP" not in obj:
        raise ValueError("PRCP is required to condition the daily generator.")
    overrides = unit_overrides or {}
    # Creating a Dask graph around file-backed arrays avoids accidental whole-grid
    # reads during unit conversion, masking, and subsequent seasonal concatenation.
    if obj.chunks is None or not obj.chunks:
        obj = obj.chunk({d: n for d, n in {"T": 366, "Y": 32, "X": 32}.items() if d in obj.dims})
    if time_was_sorted:
        obj = obj.sortby("T")
    result = {}
    for name, da in obj.data_vars.items():
        if set(da.dims) != {"T", "Y", "X"}:
            raise ValueError(f"{name} must have exactly T,Y,X dimensions; found {da.dims}.")
        da = _convert_units(da.transpose("T", "Y", "X"), name, overrides.get(name), assume_units)
        valid = np.isfinite(da)
        if name in {"PRCP", "WIND", "SOLAR", "HUMIN", "HUMAX"}:
            valid = valid & (da >= 0)
        if name in {"HUMIN", "HUMAX"}:
            valid = valid & (da <= 100)
        result[name] = da.where(valid)
    output = xr.Dataset(result, attrs=dict(obj.attrs))
    output = output.assign_coords(T=pd.DatetimeIndex(output.T.values).normalize())
    if time_was_sorted:
        output.attrs["time_coordinate_repair"] = "Unique daily T coordinates sorted chronologically before analysis."
    output.attrs["invalid_value_policy"] = "Nonfinite, negative nonnegative variables and humidity outside [0,100] masked."
    return output


def open_observations(paths: Mapping[str, str | Path], chunks=None,
                      unit_overrides=None) -> xr.Dataset:
    """Open one file per canonical variable without loading meteorological grids.

    A source file may contain a canonical variable, or one sole data variable.
    Files must have identical T,Y,X coordinates; no silent temporal/spatial joins.
    ``chunks=None`` chooses conservative Dask chunks (T=366,Y=32,X=32).
    """
    if not paths:
        raise ValueError("Provide at least a PRCP observation file.")
    arrays, opened = [], []
    try:
        for name, path in paths.items():
            if name not in VARIABLES:
                raise ValueError(f"Unknown canonical variable {name!r}; allowed {VARIABLES}.")
            # Read only metadata first so aliases can be renamed before chunking.
            ds = xr.open_dataset(path, chunks={})
            opened.append(ds)
            ds = _rename_dimensions(ds)
            if name in ds.data_vars:
                da = ds[name]
            elif len(ds.data_vars) == 1:
                da = next(iter(ds.data_vars.values()))
            else:
                raise ValueError(f"{path} has multiple variables without {name}; select/rename its intended variable.")
            da = da.rename(name)
            da = da.chunk(chunks if chunks is not None else {"T": 366, "Y": 32, "X": 32})
            arrays.append(da)
        aligned = xr.align(*arrays, join="exact", copy=False)
        result = canonicalize_observations(xr.Dataset({a.name: a for a in aligned}),
                                          unit_overrides=unit_overrides)
        def close_sources():
            for source in opened:
                source.close()
        result.set_close(close_sources)
        return result
    except Exception:
        for source in opened:
            source.close()
        raise


def prepare_probabilities(probabilities: xr.DataArray, target=None,
                          method="nearest") -> xr.DataArray:
    """Validate PB,PN,PA, normalize rounded fractions/percentages, align grids.

    Entire triples are masked when partly missing, negative, or with a sum more
    than 2% from 1 or 100. Interpolation never extrapolates past the source extent.
    Nearest-neighbor interpolation is the default; linear is explicitly optional.
    """
    if not isinstance(probabilities, xr.DataArray):
        raise TypeError("probabilities must be an xarray.DataArray.")
    p = _rename_dimensions(probabilities)
    if "T" in p.dims:
        if p.sizes["T"] != 1:
            raise ValueError("Select exactly one forecast time before preparing probabilities.")
        p = p.isel(T=0, drop=True)
    if set(p.dims) != {"probability", "Y", "X"}:
        raise ValueError("Forecast must have probability,Y,X dimensions, optionally a singleton T.")
    _validate_grid(p)
    if "probability" not in p.coords:
        raise ValueError("Probability coordinate must explicitly label PB,PN,PA; order is never guessed.")
    labels = [str(v.decode() if isinstance(v, bytes) else v).upper() for v in p.probability.values]
    if len(labels) != 3 or set(labels) != {"PB", "PN", "PA"}:
        raise ValueError("Probability labels must be exactly PB,PN,PA.")
    p = p.assign_coords(probability=labels).sel(probability=["PB", "PN", "PA"])
    attrs = dict(p.attrs)
    total = p.sum("probability", skipna=False)
    valid = (np.isfinite(p).all("probability") & (p >= 0).all("probability")
             & ((abs(total - 1.) <= .020000001) | (abs(total - 100.) <= 2.0000001)))
    p = (p / total.where(valid)).where(valid)
    if target is not None:
        target = _rename_dimensions(target)
        _validate_grid(target)
        if method not in {"nearest", "linear"}:
            raise ValueError("Probability interpolation method must be 'nearest' or 'linear'.")
        same = all(np.array_equal(p[d].values, target[d].values) for d in ("Y", "X"))
        if not same:
            p = p.sortby("Y").sortby("X").interp(Y=target.Y, X=target.X,
                method=method, kwargs={"bounds_error": False, "fill_value": np.nan})
        # Both methods preserve sums in valid regions; normalize roundoff.
        total = p.sum("probability", skipna=False)
        p = p / total.where(total > 0)
    p = p.transpose("probability", "Y", "X").rename("forecast_probability")
    p.attrs = attrs | {"units": "1", "category_order": "PB PN PA",
        "invalid_probability_policy": "Mask entire triple; accept totals within 2% of 1 or 100 then normalize.",
        "spatial_alignment": method + "; no extrapolation" if target is not None else "source grid"}
    return p


def load_probabilities(path, target=None, method="nearest", time_index=0):
    """Read a small forecast file and return a canonical probability DataArray."""
    with xr.open_dataset(path) as ds:
        candidates = [a for a in ds.data_vars.values() if "probability" in a.dims]
        if len(candidates) != 1:
            raise ValueError("Forecast file must contain one variable with a probability dimension.")
        p = _rename_dimensions(candidates[0])
        if "T" in p.dims:
            p = p.isel(T=time_index, drop=True)
        # Forecast grids are small; detach from the closed file. Observation grids
        # are never loaded by this function.
        p = p.load()
    return prepare_probabilities(p, target=target, method=method)


def validate_months(months):
    months = tuple(months)
    if not months or len(months) > 12 or any(isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or m < 1 or m > 12 for m in months):
        raise ValueError("months must contain 1–12 integer month numbers.")
    if len(set(months)) != len(months) or any(b != a % 12 + 1 for a, b in zip(months[:-1], months[1:])):
        raise ValueError("months must be unique and contiguous in chronological order, e.g. (12,1,2).")
    return months


def season_dates(year, months=(7, 8, 9)) -> pd.DatetimeIndex:
    """Dates for a season starting in ``year``, excluding February 29."""
    months = validate_months(months)
    if isinstance(year, (bool, np.bool_)) or not isinstance(year, (int, np.integer)):
        raise ValueError("year must be an integer representing the first season month's year.")
    dates = []
    year_now = int(year)
    previous = months[0]
    for month in months:
        if month < previous:
            year_now += 1
        length = 28 if month == 2 else calendar.monthrange(year_now, month)[1]
        dates.extend(pd.date_range(f"{year_now:04d}-{month:02d}-01", periods=length, freq="D"))
        previous = month
    return pd.DatetimeIndex(dates, name="T")


def seasonal_cube(observations: xr.Dataset, months=(7, 8, 9)) -> xr.Dataset:
    """Build a lazy season_year,day,Y,X cube; omit globally incomplete seasons.

    Calendar completeness is determined from T only. A missing value at one cell
    stays missing and does not discard the season at other cells. Downstream sums
    must use ``skipna=False`` so missing days cannot become low rainfall totals.
    """
    months = validate_months(months)
    obs = _rename_dimensions(observations)
    _validate_grid(obs, time=True)
    if not obs.chunks:
        obs = obs.chunk({"T": 366, "Y": 32, "X": 32})
    if not pd.DatetimeIndex(obs.T.values).is_monotonic_increasing:
        obs = obs.sortby("T")
    times = pd.DatetimeIndex(obs.T.values).normalize()
    slices, years, dropped = [], [], []
    # Starting one year earlier also permits an all-incomplete first cross-year season.
    first_year, last_year = times[0].year - 1, times[-1].year
    for year in range(first_year, last_year + 1):
        wanted = season_dates(year, months)
        if wanted[-1] < times[0] or wanted[0] > times[-1]:
            continue
        indices = times.get_indexer(wanted)
        if (indices < 0).any():
            dropped.append(year)
            continue
        season = obs.isel(T=indices).rename(T="day").assign_coords(day=np.arange(len(wanted)))
        slices.append(season)
        years.append(year)
    if not slices:
        raise ValueError("No complete seasons found for requested months; check dates and daily calendar completeness.")
    result = xr.concat(slices, dim=xr.IndexVariable("season_year", years),
                       coords="minimal", compat="override", join="exact")
    representative = season_dates(years[0], months)
    result = result.assign_coords(month=("day", representative.month.to_numpy()),
                                  day_of_month=("day", representative.day.to_numpy()))
    result = result.transpose("season_year", "day", "Y", "X")
    result.attrs = dict(obs.attrs) | {"season_months": ",".join(map(str, months)),
        "season_year_convention": "calendar year of first month", "leap_day_policy": "February 29 omitted",
        "dropped_incomplete_seasons": ",".join(map(str, dropped))}
    return result
