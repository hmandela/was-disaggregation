"""Run with: pixi run python examples/synthetic_smoke.py."""
import numpy as np
import pandas as pd
import xarray as xr
from was_disaggregation import WeatherGenerator, season_dates

dates = pd.DatetimeIndex(np.concatenate([season_dates(y, (7, 8, 9)).values for y in range(2001, 2013)]))
rng = np.random.default_rng(42)
rain = rng.gamma(1.5, 6., (len(dates), 2, 2))
rain[rng.random(rain.shape) < .45] = 0
obs = xr.Dataset({"PRCP": (("T", "Y", "X"), rain)},
                 coords={"T": dates, "Y": [6., 7.], "X": [1., 2.]})
obs.PRCP.attrs["units"] = "mm d-1"
prob = xr.DataArray(np.broadcast_to(np.array([.2, .3, .5])[:, None, None], (3, 2, 2)),
                    dims=("probability", "Y", "X"),
                    coords={"probability": ["PB", "PN", "PA"], "Y": obs.Y, "X": obs.X})
model = WeatherGenerator(climatology=(2001, 2012), spatial="independent", seed=42).fit(obs, prob)
out = model.generate(2013, n_members=10)
print(out)
print("Mean seasonal totals (mm):", out.PRCP.sum("T").mean("member").values)
