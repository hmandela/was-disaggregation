"""Agro-climatic season attributes computed from daily rainfall.

Each attribute turns daily PRCP into one number per season and site: seasonal
total, onset date, cessation date or maximum dry-spell length. These are the
quantities forecast by PRESASS / AGRHYMET, and each one can become a
constraint on the historical year weights (see :mod:`was_disaggregation.mre`).

Windows are given as ``("MM-DD", "MM-DD")`` in the calendar year of the season
(``end < start`` crosses into the next year). February 29 is omitted, matching
the generators' 365-day Gregorian policy. Date attributes are returned as
**days after the start of their search window** (0 = search start), which is
counted on this calendar and orders early -> late. A season whose event is not found
inside the search window is *censored* at the window length (i.e. classed as
late); ``censored_`` reports how often this happened.

Any missing day inside an attribute's data span makes that season's value NaN
(missing is never treated as dry).

The functions accept a DataArray with a ``T`` dimension and any other
dimensions (Y, X, member ...), stacked internally into ``site``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


def _md(text):
    month, day = (int(v) for v in str(text).split("-"))
    return month, day


def _window(year, start, end):
    sm, sd = _md(start)
    em, ed = _md(end)
    first = pd.Timestamp(int(year), sm, sd)
    last = pd.Timestamp(int(year) + (1 if (em, ed) < (sm, sd) else 0), em, ed)
    return first, last


def _window_dates(year, start, end, pad_after=0):
    """Window plus look-ahead on the package's February-29-excluded calendar."""
    first, last = _window(year, start, end)
    dates = pd.date_range(first, last, freq="D")
    dates = dates[~((dates.month == 2) & (dates.day == 29))]
    extra, current = [], last
    while len(extra) < pad_after:
        current += pd.Timedelta(days=1)
        if (current.month, current.day) != (2, 29):
            extra.append(current)
    return dates.append(pd.DatetimeIndex(extra)) if extra else dates


def _window_length(year, start, end):
    return len(_window_dates(year, start, end))


def _stack(prcp):
    """DataArray (T, ...) -> (DataArray (T, site), template of the other dims)."""
    if "T" not in prcp.dims:
        raise ValueError("daily rainfall must have a T dimension")
    if prcp.sizes["T"] == 0:
        raise ValueError("daily rainfall must have a nonempty T dimension")
    other = [d for d in prcp.dims if d != "T"]
    template = prcp.isel(T=0, drop=True)
    if other:
        stacked = prcp.transpose("T", *other).stack(site=other)
    else:
        stacked = prcp.expand_dims(site=[0], axis=1)
    return stacked, template


def extract_windows(prcp, years, start, end, pad_after=0):
    """Daily values for each season window -> (year, day, site), NaN padded.

    ``pad_after`` extra days are appended after ``end`` (look-ahead for
    false-start checks or post-onset dry-spell windows).
    """
    stacked, _ = _stack(prcp)
    years = list(years)
    if not years or not isinstance(pad_after, (int, np.integer)) or pad_after < 0:
        raise ValueError("years must be nonempty and pad_after a nonnegative integer")
    if not np.issubdtype(stacked["T"].dtype, np.datetime64):
        raise ValueError("T must use Gregorian datetime64 dates")
    times = pd.DatetimeIndex(stacked["T"].values).normalize()
    if times.hasnans or times.has_duplicates:
        raise ValueError("T must contain unique daily dates without NaT")
    stacked = stacked.assign_coords(T=times)
    spans = [_window_dates(y, start, end, pad_after) for y in years]
    lengths = [len(dates) for dates in spans]
    n = max(lengths)
    out = np.full((len(years), n, stacked.sizes["site"]), np.nan)
    for i, dates in enumerate(spans):
        values = stacked.reindex(T=dates).values
        out[i, :lengths[i]] = values
    return out


def _dry_run_ending(wet):
    """Length of the dry run ending at each day (0 on wet days), axis 1 = day."""
    run = np.zeros(wet.shape, dtype=np.int32)
    current = np.zeros(wet.shape[:1] + wet.shape[2:], dtype=np.int32)
    for d in range(wet.shape[1]):
        current = np.where(wet[:, d], 0, current + 1)
        run[:, d] = current
    return run


def _reshape(values, template, leading):
    """(year, site) -> DataArray (leading, *template.dims)."""
    name, coord = leading
    shape = (values.shape[0],) + template.shape
    return xr.DataArray(values.reshape(shape), dims=(name,) + template.dims,
                        coords={name: coord, **{d: template[d] for d in template.dims if d in template.coords}})


class Attribute:
    """Base class. Subclasses implement ``values(prcp, years) -> (year, site)``."""
    name = "attribute"
    units = ""

    def compute(self, prcp: xr.DataArray, years) -> xr.DataArray:
        """Attribute per season year, as DataArray (season_year, <other dims>)."""
        years = [int(y) for y in np.atleast_1d(years)]
        _, template = _stack(prcp)
        values = self.values(prcp, years)
        out = _reshape(values, template, ("season_year", years))
        out.name = self.name
        out.attrs.update(units=self.units, definition=repr(self), calendar_policy="Gregorian; February 29 omitted")
        return out

    def span(self):
        """(start 'MM-DD', end 'MM-DD', extra days after end) of data needed."""
        raise NotImplementedError

    def values(self, prcp, years):
        raise NotImplementedError


@dataclass
class SeasonalTotal(Attribute):
    """Rainfall total over ``window`` (mm). Requires every day of the window."""
    window: tuple = ("07-01", "09-30")
    name: str = "total"
    units: str = "mm"

    def span(self):
        return self.window[0], self.window[1], 0

    def values(self, prcp, years):
        x = extract_windows(prcp, years, *self.window)
        valid_len = np.array([_window_length(y, *self.window) for y in years])
        out = np.full((len(years), x.shape[2]), np.nan)
        for i, n in enumerate(valid_len):
            block = x[i, :n]
            out[i] = np.where(np.isfinite(block).all(axis=0), np.nansum(block, axis=0), np.nan)
        return out


@dataclass
class OnsetDate(Attribute):
    """Agronomic onset (Sivakumar 1988, as used by AGRHYMET / PRESASS).

    First day ``d`` of the search window such that
      * rain accumulated over days d .. d+accumulation_days-1 >= accumulation_mm,
      * at least ``min_wet_days`` of those days are wet (>= wet_threshold),
      * no dry spell longer than ``dry_spell_days`` occurs in the following
        ``check_days`` days (false-start check).
    Returned as days after ``search[0]``; censored at the window length.
    """
    search: tuple = ("05-01", "09-30")
    accumulation_mm: float = 20.0
    accumulation_days: int = 3
    min_wet_days: int = 1
    dry_spell_days: int = 7
    check_days: int = 30
    wet_threshold: float = 1.0
    name: str = "onset"
    units: str = "days after search start"

    def __post_init__(self):
        _window(2001, *self.search)
        for name in ("accumulation_days", "min_wet_days", "dry_spell_days", "check_days"):
            value = getattr(self, name)
            minimum = 1 if name in ("accumulation_days", "min_wet_days") else 0
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.min_wet_days > self.accumulation_days:
            raise ValueError("min_wet_days cannot exceed accumulation_days")
        if not np.isfinite(self.accumulation_mm) or self.accumulation_mm <= 0:
            raise ValueError("accumulation_mm must be positive")
        if not np.isfinite(self.wet_threshold) or self.wet_threshold <= 0:
            raise ValueError("wet_threshold must be positive")

    def span(self):
        return self.search[0], self.search[1], self.check_days + self.accumulation_days - 1

    def values(self, prcp, years):
        onset, _ = self._onset(prcp, years)
        return onset

    def _onset(self, prcp, years, x=None):
        if x is None:
            x = extract_windows(prcp, years, *self.span()[:2], pad_after=self.span()[2])
        ny, nd, ns = x.shape
        length = np.array([_window_length(y, *self.search) for y in years])
        missing = np.zeros((ny, ns), dtype=bool)
        for i, n in enumerate(length):
            missing[i] = ~np.isfinite(x[i, :n + self.span()[2]]).all(axis=0)
        xf = np.nan_to_num(x)
        wet = xf >= self.wet_threshold
        k = self.accumulation_days
        csum = np.concatenate([np.zeros((ny, 1, ns)), np.cumsum(xf, axis=1)], axis=1)
        cwet = np.concatenate([np.zeros((ny, 1, ns)), np.cumsum(wet, axis=1)], axis=1)
        idx = np.arange(nd - k + 1)
        accum = csum[:, idx + k] - csum[:, idx]
        nwet = cwet[:, idx + k] - cwet[:, idx]
        candidate = (accum >= self.accumulation_mm) & (nwet >= self.min_wet_days)
        # longest dry spell strictly after the accumulation window, within check_days
        run = _dry_run_ending(wet)
        longest = np.zeros(candidate.shape, dtype=np.int32)
        for j in range(1, self.check_days + 1):
            pos = idx + k - 1 + j
            ok = pos < nd
            r = np.zeros(candidate.shape, dtype=np.int32)
            r[:, ok] = np.minimum(run[:, pos[ok]], j)
            longest = np.maximum(longest, r)
        good = candidate & (longest <= self.dry_spell_days)
        onset = np.full((ny, ns), np.nan)
        censored = np.zeros((ny, ns), dtype=bool)
        for i, n in enumerate(length):
            g = good[i, :n]
            found = g.any(axis=0)
            first = np.argmax(g, axis=0)
            onset[i] = np.where(found, first, n)
            censored[i] = ~found
        onset[missing] = np.nan
        self.censored_ = float(np.mean(censored[~missing])) if (~missing).any() else np.nan
        return onset, x


@dataclass
class CessationDate(Attribute):
    """End of season from a bucket water balance (AGRHYMET-style).

    Soil water S starts at 0 on ``balance_start`` and evolves as
    S = clip(S + P - et_mm, 0, capacity_mm). Cessation is the first day of
    ``search`` on which S = 0 (days after ``search[0]``); censored if never.
    """
    search: tuple = ("09-01", "11-30")
    balance_start: str = "05-01"
    capacity_mm: float = 70.0
    et_mm: float = 5.0
    name: str = "cessation"
    units: str = "days after search start"

    def __post_init__(self):
        _window(2001, *self.search)
        _window(2001, self.balance_start, self.search[1])
        if not np.isfinite(self.capacity_mm) or self.capacity_mm <= 0:
            raise ValueError("capacity_mm must be positive")
        if not np.isfinite(self.et_mm) or self.et_mm < 0:
            raise ValueError("et_mm must be finite and nonnegative")

    def span(self):
        return self.balance_start, self.search[1], 0

    def values(self, prcp, years):
        x = extract_windows(prcp, years, self.balance_start, self.search[1])
        ny, nd, ns = x.shape
        out = np.full((ny, ns), np.nan)
        for i, y in enumerate(years):
            b0, end = _window(y, self.balance_start, self.search[1])
            s0 = _window(y, *self.search)[0]
            if s0 < b0:
                s0 = s0.replace(year=s0.year + 1)
            dates = _window_dates(y, self.balance_start, self.search[1])
            n = len(dates)
            offset = int((dates < s0).sum())
            block = x[i, :n]
            miss = ~np.isfinite(block).all(axis=0)
            soil = np.zeros(ns)
            result = np.full(ns, float(n - offset))
            found = np.zeros(ns, dtype=bool)
            for d in range(n):
                soil = np.clip(soil + np.nan_to_num(block[d]) - self.et_mm, 0, self.capacity_mm)
                if d >= offset:
                    hit = (soil <= 0) & ~found
                    result[hit] = d - offset
                    found |= hit
            out[i] = np.where(miss, np.nan, result)
        return out


@dataclass
class MaxDrySpell(Attribute):
    """Longest run of dry days (< wet_threshold) inside a window.

    Fixed window: ``window=("MM-DD","MM-DD")``. Onset-relative window (PRESASS
    early-season dry spells): ``after_onset=OnsetDate(...)`` and ``length_days``
    -> days [onset, onset + length_days). Seasons without onset are censored
    at ``length_days`` (failed season = longest spell).
    Spells are cut at the window edges.
    """
    window: tuple | None = None
    after_onset: OnsetDate | None = None
    length_days: int = 50
    wet_threshold: float = 1.0
    name: str = "dry_spell"
    units: str = "days"

    def __post_init__(self):
        if (self.window is None) == (self.after_onset is None):
            raise ValueError("give exactly one of window or after_onset")
        if not isinstance(self.length_days, (int, np.integer)) or self.length_days < 1:
            raise ValueError("length_days must be a positive integer")
        if not np.isfinite(self.wet_threshold) or self.wet_threshold <= 0:
            raise ValueError("wet_threshold must be positive")

    def span(self):
        if self.window is not None:
            return self.window[0], self.window[1], 0
        s, e, pad = self.after_onset.span()
        return s, e, max(pad, self.length_days - 1)

    def values(self, prcp, years):
        if self.window is not None:
            x = extract_windows(prcp, years, *self.window)
            n = np.array([_window_length(y, *self.window) for y in years])
            start = np.zeros((len(years), x.shape[2]), dtype=int)
            stop = np.broadcast_to(n[:, None], start.shape)
        else:
            s, e, pad = self.span()
            x = extract_windows(prcp, years, s, e, pad_after=pad)
            onset, _ = self.after_onset._onset(prcp, years, x=x)
            n_search = np.array([_window_length(y, *self.after_onset.search) for y in years])
            no_onset = onset >= n_search[:, None]
            start = np.where(np.isfinite(onset), onset, 0).astype(int)
            stop = start + self.length_days
            start = np.where(no_onset, 0, start)
        ny, nd, ns = x.shape
        day = np.arange(nd)[None, :, None]
        inside = (day >= start[:, None, :]) & (day < np.minimum(stop, nd)[:, None, :])
        miss = (~np.isfinite(x) & inside).any(axis=1)
        dry = inside & (np.nan_to_num(x) < self.wet_threshold)
        run = _dry_run_ending(~dry)
        out = np.where(miss, np.nan, run.max(axis=1).astype(float))
        if self.after_onset is not None:
            # No onset in the search window: a failed season, censored as the
            # longest possible spell (keeps such years in the "long" class).
            out = np.where(np.isfinite(onset), np.where(no_onset, float(self.length_days), out), np.nan)
        return out


__all__ = ["Attribute", "SeasonalTotal", "OnsetDate", "CessationDate", "MaxDrySpell", "extract_windows"]
