"""Dask tiled generation with spatially shared, deterministic innovations."""
from __future__ import annotations
import json
import numpy as np
import xarray as xr
from .data import canonicalize_observations, prepare_probabilities, season_dates
from .model import WeatherGenerator
from ._version import __version__


def multiscale_indices(length, count=8):
    """Select broad anchors and immediate neighbours to cover several distances."""
    length, count = int(length), min(int(count), int(length))
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    if count == 1:
        return np.array([length//2])
    anchors = np.unique(np.linspace(0, length-1, (count+1)//2).astype(int))
    neighbours = np.where(anchors < length-1, anchors+1, anchors-1)
    selected = np.unique(np.concatenate([anchors, neighbours]))
    if len(selected) < count:
        unused = np.setdiff1d(np.arange(length), selected)
        selected = np.sort(np.r_[selected, unused[:count-len(selected)]])
    return selected[:count]


def _tile_generate(obs, probabilities, site_ids, year, n_members, member_start, kwargs, names):
    model = WeatherGenerator(**kwargs).fit(obs, probabilities, site_ids=site_ids)
    output = model.generate(year=year, n_members=n_members, member_start=member_start)
    return (np.stack([output[v].values for v in names]),
            output["tercile_class"].values if "tercile_class" in output else None)


def generate_dask(observations, probabilities, year, n_members=20,
                  months=(7,8,9), climatology=(1991,2020), tile_shape=(10,10),
                  spatial_models=None, seed=42, member_batch=20,
                  calibration_side=8, **model_options):
    """Return a lazy Dataset with bounded spatial and member tasks.

    Spatial kernels are calibrated ONCE from a multiscale domain sample (at most
    calibration_side squared cells), unless supplied. Broad anchors plus adjacent
    grid points sample both short and long distances. All tiles use those same
    kernels and global site IDs. There are no independent tile seams introduced
    by reseeding. Fitted local parameters are recomputed per member batch; this
    conservative memory tradeoff is intentional. Use ``ds.to_netcdf`` or Zarr to
    stream output; ``ds.compute()`` materializes the entire output.

    For NetCDF input prefer Dask's synchronous or threaded scheduler; process
    workers should reopen files themselves if the backend is not serializable.
    Scheduler concurrency times per-task memory determines peak memory.
    """
    import dask.array as da
    from dask import delayed
    for value, name in ((year, "year"), (n_members, "n_members"),
                        (member_batch, "member_batch"), (calibration_side, "calibration_side")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if len(tile_shape) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in tile_shape):
        raise ValueError("tile_shape must contain two positive integers")
    obs = canonicalize_observations(observations)
    if "PRCP" not in obs:
        raise ValueError("PRCP is required")
    prob = prepare_probabilities(probabilities, target=obs)
    ny,nx = obs.sizes["Y"],obs.sizes["X"]
    ty,tx = map(int,tile_shape)
    if min(ty,tx,n_members,member_batch,calibration_side) < 1:
        raise ValueError("Tiles, batches, calibration_side and ensemble sizes must be positive")
    options = dict(months=months,climatology=climatology,seed=seed,**model_options)
    options.pop("max_sites",None)
    options["max_sites"] = max(ty*tx,calibration_side**2)
    spatial = options.get("spatial","distance")
    if spatial == "distance" and spatial_models is None:
        yi = multiscale_indices(ny,calibration_side)
        xi = multiscale_indices(nx,calibration_side)
        calibration = WeatherGenerator(**options).fit(obs.isel(Y=yi,X=xi),prob.isel(Y=yi,X=xi))
        spatial_models = calibration.spatial_models_
    if (spatial == "distance" and options.get("conditioning") == "mixture"
            and options.get("class_draw", "fitted") == "fitted" and "season_class" not in spatial_models):
        raise ValueError("Mixture with class_draw='fitted' needs a shared 'season_class' kernel in spatial_models "
                         "(refit the calibration with conditioning='mixture', or use class_draw='shared').")
    options["spatial_models"] = spatial_models
    names = ["PRCP"] + [v for v in obs.data_vars if v != "PRCP"]
    dates = season_dates(int(year),months)
    nday = len(dates)
    blocks_members=[]
    classes_members=[]
    mixture = options.get("conditioning") == "mixture"
    for start in range(0,int(n_members),int(member_batch)):
        count=min(int(member_batch),int(n_members)-start)
        blocks_y=[]
        classes_y=[]
        for y0 in range(0,ny,ty):
            blocks_x=[]
            classes_x=[]
            ys=slice(y0,min(y0+ty,ny))
            for x0 in range(0,nx,tx):
                xs=slice(x0,min(x0+tx,nx))
                ids=(np.arange(ys.start,ys.stop)[:,None]*nx+np.arange(xs.start,xs.stop)[None,:]).ravel()
                task=delayed(_tile_generate)(obs.isel(Y=ys,X=xs),prob.isel(Y=ys,X=xs),ids,
                    int(year),count,start,options,names)
                blocks_x.append(da.from_delayed(task[0],shape=(len(names),count,nday,ys.stop-ys.start,xs.stop-xs.start),dtype=np.float32))
                if mixture:
                    classes_x.append(da.from_delayed(task[1],shape=(count,ys.stop-ys.start,xs.stop-xs.start),dtype=np.int8))
            blocks_y.append(da.concatenate(blocks_x,axis=4))
            if mixture:
                classes_y.append(da.concatenate(classes_x,axis=2))
        blocks_members.append(da.concatenate(blocks_y,axis=3))
        if mixture:
            classes_members.append(da.concatenate(classes_y,axis=1))
    array=da.concatenate(blocks_members,axis=1)
    result=xr.Dataset({v:(("member","T","Y","X"),array[i]) for i,v in enumerate(names)},
        coords={"member":np.arange(n_members),"T":dates,"Y":obs.Y,"X":obs.X})
    for v in result:
        result[v].attrs.update(obs[v].attrs)
    if mixture:
        result["tercile_class"] = (("member", "Y", "X"), da.concatenate(classes_members,axis=0))
        result["tercile_class"].attrs.update(flag_values="0 1 2", flag_meanings="below near above",
            description="tercile class whose parameters drove each member/cell")
    result.attrs.update(generator=f"was-disaggregation {__version__} Dask tiles",seed=int(seed),
        season_months=",".join(map(str,months)),climatology=f"{climatology[0]}-{climatology[1]}",
        spatial_method=spatial,spatial_models=json.dumps(spatial_models or {},default=lambda x: np.asarray(x).tolist()),
        tile_shape=str(tuple(tile_shape)),member_batch=int(member_batch),
        caveat="Stationary spatial kernels fitted on domain sample; validate regional correlation and seasonal frequencies",
        leap_day_policy="February 29 excluded")
    return result
