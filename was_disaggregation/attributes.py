"""Agro-climatic season attributes computed from daily rainfall.

SeasonalTotal, OnsetDate, CessationDate and MaxDrySpell turn daily PRCP into
one value per season and site, for use as forecasts or reweighting constraints.
Windows use MM-DD dates, omit February 29 and may cross the year boundary.
Date values count days after the search start. Events not found are censored
at the window length; censored_ and per-event diagnostics distinguish this
convention from an observed event.

Missing rainfall remains missing. Totals require complete windows; onset and
bucket cessation inspect the prefix needed to establish the first event.
The public functions accept a T dimension and arbitrary spatial/member axes.

Scientific attribution
----------------------
M. V. K. Sivakumar (1988), "Predicting rainy season potential from the onset
of rains in Southern Sahelian and Sudanian climatic zones of West Africa",
Agricultural and Forest Meteorology 42, 295-305.
DOI: 10.1016/0168-1923(88)90039-1.
    OnsetDate implements the accumulation plus false-start criterion with
    configurable thresholds and dates. CessationDate(method="sivakumar")
    implements the rainfall-only end-of-rains criterion. Changing their
    defaults defines a variant; the 58-location experiment is not reproduced.
    CessationDate(method="bucket") is a package bucket-water-balance
    convention, not Sivakumar's published rainfall-only cessation method.
    SeasonalTotal and MaxDrySpell are descriptive statistics with explicit
    windows. Their implementation is not attributed to a new named article.

Censoring, required-data padding and missing-prefix checks are package
extensions, not demonstrated improvements to agricultural forecast skill.
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
        out[i, :lengths[i]] = np.where(np.isfinite(values) & (values >= 0), values, np.nan)
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
        if not years or len(set(years)) != len(years):
            raise ValueError("years must be nonempty and unique")
        _, template = _stack(prcp)
        self._censored_values = None
        values = self.values(prcp, years)
        out = _reshape(values, template, ("season_year", years))
        out.name = self.name
        out.attrs.update(units=self.units, definition=repr(self), calendar_policy="Gregorian; February 29 omitted")
        if self._censored_values is not None:
            observed = np.where(np.isfinite(values), 1. - self._censored_values, np.nan)
            out = out.assign_coords(event_observed=_reshape(observed, template, ("season_year", years)))
            out.event_observed.attrs.update(description="1 = event found; 0 = right-censored; NaN = unresolved missing data")
            out.attrs["censoring_policy"] = "No event is represented by the window length; this is a capped event variable, not an exact date."
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
      * no dry spell longer than ``dry_spell_days`` occurs in days
        d+1 .. d+``check_days`` (false-start check; the three accumulation
        days are part of these 30 days, as in Sivakumar 1988).
    Returned as days after ``search[0]``; censored at the window length.
    Missing dates are checked only for candidates that could precede the
    first valid onset; an early validated onset needs no data at the far end
    of the search window or its padding.

    References
    ----------
    M. V. K. Sivakumar (1988), DOI: 10.1016/0168-1923(88)90039-1;
    see the module bibliography. Configurable thresholds, min_wet_days,
    calendar policy and censoring are explicit implementation conventions.
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
        return self.search[0], self.search[1], max(self.check_days, self.accumulation_days - 1)

    def values(self, prcp, years):
        onset, _ = self._onset(prcp, years)
        return onset

    def _onset(self, prcp, years, x=None):
        if x is None:
            x = extract_windows(prcp, years, *self.span()[:2], pad_after=self.span()[2])
        ny, nd, ns = x.shape
        length = np.array([_window_length(y, *self.search) for y in years])
        finite = np.isfinite(x)
        xf = np.where(finite, x, 0.)
        # Missing observations must not act as dry days: a known, long dry
        # spell still rules out a candidate, while a gap otherwise leaves it
        # undecidable.
        wet = ~finite | (xf >= self.wet_threshold)
        k = self.accumulation_days
        csum = np.concatenate([np.zeros((ny, 1, ns)), np.cumsum(xf, axis=1)], axis=1)
        cwet = np.concatenate([np.zeros((ny, 1, ns)), np.cumsum(finite & wet, axis=1)], axis=1)
        cmissing = np.concatenate([np.zeros((ny, 1, ns), dtype=int),
                                   np.cumsum(~finite, axis=1)], axis=1)
        idx = np.arange(nd - k + 1)
        accum = csum[:, idx + k] - csum[:, idx]
        nwet = cwet[:, idx + k] - cwet[:, idx]
        candidate = (accum >= self.accumulation_mm) & (nwet >= self.min_wet_days)
        complete_accum = (cmissing[:, idx + k] - cmissing[:, idx]) == 0
        # Sivakumar's 30-day check starts immediately after the candidate
        # onset day, including days 2 and 3 of the accumulation window.
        run = _dry_run_ending(wet)
        longest = np.zeros(candidate.shape, dtype=np.int32)
        for j in range(1, self.check_days + 1):
            pos = idx + j
            ok = pos < nd
            r = np.zeros(candidate.shape, dtype=np.int32)
            r[:, ok] = np.minimum(run[:, pos[ok]], j)
            longest = np.maximum(longest, r)
        check_end = idx + self.check_days + 1
        complete_check = np.zeros(candidate.shape, dtype=bool)
        ok = check_end <= nd
        complete_check[:, ok] = ((cmissing[:, check_end[ok]] - cmissing[:, idx[ok] + 1]) == 0)
        too_dry = longest > self.dry_spell_days
        good = candidate & complete_accum & complete_check & ~too_dry
        unresolved = ~complete_accum | (candidate & ~complete_check & ~too_dry)
        onset = np.full((ny, ns), np.nan)
        censored = np.zeros((ny, ns), dtype=bool)
        for i, n in enumerate(length):
            g = good[i, :n]
            u = unresolved[i, :n]
            found = g.any(axis=0)
            first = np.where(found, np.argmax(g, axis=0), n)
            unknown = u.any(axis=0) & (np.argmax(u, axis=0) <= first)
            onset[i] = np.where(unknown, np.nan, first)
            censored[i] = ~found & ~unknown
        known = np.isfinite(onset)
        self._censored_values = censored.astype(float)
        self.censored_ = float(np.mean(censored[known])) if known.any() else np.nan
        return onset, x


@dataclass
class CessationDate(Attribute):
    """End of season from a bucket water balance (AGRHYMET-style, default).

    Soil water S starts at 0 on ``balance_start`` and evolves as
    S = clip(S + P - et_mm, 0, capacity_mm). Cessation is the first day of
    ``search`` on which S = 0 (days after ``search[0]``); censored if never.

    ``method="sivakumar"`` instead returns the first candidate day after
    which *no rain* falls during the next 20 days, matching the rainfall-only
    ending-of-rains criterion in Sivakumar (1988). For the paper's strictly
    post-September-1 candidates use ``search=("09-02", "11-30")``.

    References
    ----------
    M. V. K. Sivakumar (1988), DOI: 10.1016/0168-1923(88)90039-1,
    for method="sivakumar" only. The default bucket method is a package
    water-balance variant; it is not the cessation equation of that article.
    """
    search: tuple = ("09-01", "11-30")
    balance_start: str = "05-01"
    capacity_mm: float = 70.0
    et_mm: float = 5.0
    name: str = "cessation"
    units: str = "days after search start"
    method: str = "bucket"
    initial_soil_mm: float = 0.0

    def __post_init__(self):
        _window(2001, *self.search)
        if self.method not in {"bucket", "sivakumar"}:
            raise ValueError("method must be 'bucket' or 'sivakumar'")
        _window(2001, self.balance_start, self.search[1])
        if not np.isfinite(self.capacity_mm) or self.capacity_mm <= 0:
            raise ValueError("capacity_mm must be positive")
        if not np.isfinite(self.et_mm) or self.et_mm < 0:
            raise ValueError("et_mm must be finite and nonnegative")
        if not np.isfinite(self.initial_soil_mm) or not 0 <= self.initial_soil_mm <= self.capacity_mm:
            raise ValueError("initial_soil_mm must lie in [0, capacity_mm]")

    def span(self):
        if self.method == "sivakumar":
            return self.search[0], self.search[1], 20
        return self.balance_start, self.search[1], 0

    def values(self, prcp, years):
        if self.method == "sivakumar":
            x = extract_windows(prcp, years, *self.search, pad_after=20)
            ny, nd, ns = x.shape
            result = np.full((ny, ns), np.nan)
            for i, year in enumerate(years):
                n = _window_length(year, *self.search)
                block = x[i]
                # Only following days enter the rainfall-only criterion;
                # a later gap cannot undo an earlier confirmed ending.
                found = np.zeros(ns, dtype=bool)
                unknown = np.zeros(ns, dtype=bool)
                for d in range(n):
                    following = block[d + 1:d + 21]
                    has_rain = (np.isfinite(following) & (following > 0)).any(axis=0)
                    complete = np.isfinite(following).all(axis=0)
                    hit = complete & ~has_rain & ~found & ~unknown
                    result[i, hit] = d
                    found |= hit
                    unknown |= ~complete & ~has_rain & ~found
                result[i, ~found & ~unknown] = n
            lengths = np.asarray([_window_length(y, *self.search) for y in years])[:, None]
            self._censored_values = (result >= lengths).astype(float)
            known = np.isfinite(result)
            self.censored_ = float(self._censored_values[known].mean()) if known.any() else np.nan
            return result
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
            soil = np.full(ns, self.initial_soil_mm)
            result = np.full(ns, float(n - offset))
            found = np.zeros(ns, dtype=bool)
            unknown = np.zeros(ns, dtype=bool)
            for d in range(n):
                unknown |= ~np.isfinite(block[d]) & ~found
                soil = np.clip(soil + np.nan_to_num(block[d]) - self.et_mm, 0, self.capacity_mm)
                if d >= offset:
                    hit = (soil <= 0) & ~found & ~unknown
                    result[hit] = d - offset
                    found |= hit
            out[i] = np.where(unknown, np.nan, result)
        lengths = np.asarray([_window_length(y, *self.search) for y in years])[:, None]
        self._censored_values = (out >= lengths).astype(float)
        known = np.isfinite(out)
        self.censored_ = float(self._censored_values[known].mean()) if known.any() else np.nan
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
