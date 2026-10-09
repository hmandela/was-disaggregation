"""Small operational entry point; notebooks describe the statistical choices.

This module is package command-line infrastructure. The --paper-protocol
choices refer to the scientific authors documented in model, rainfall and
srg3; the command-line interface itself does not reproduce an article.
"""
import argparse
from pathlib import Path
from .data import open_observations, load_probabilities, prepare_probabilities
from .scalable import generate_dask


def main(argv=None):
    p=argparse.ArgumentParser(description="Generate daily gridded weather from a seasonal tercile forecast")
    from ._version import __version__
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--obs",required=True,help="Daily PRCP NetCDF")
    p.add_argument("--forecast",required=True,help="NetCDF with PB,PN,PA")
    p.add_argument("--output",required=True)
    p.add_argument("--year",required=True,type=int)
    p.add_argument("--months",nargs="+",type=int,default=[7,8,9])
    p.add_argument("--climatology",nargs=2,type=int,default=[1991,2020])
    p.add_argument("--members",type=int,default=20)
    p.add_argument("--member-batch",type=int,default=20)
    p.add_argument("--tile",nargs=2,type=int,default=[10,10])
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--wet-threshold",type=float,default=1.0)
    p.add_argument("--spatial",choices=["distance","independent"],default="distance")
    p.add_argument("--paper-protocol", choices=["houngnibo_2023_srg1", "houngnibo_2023_srg2",
                                               "houngnibo_2023_srg3", "wilks_2002"],
                   help="Fix the statistical conventions of the selected article; overrides generic rain options")
    p.add_argument("--n-features", type=int, default=128,
                   help="Spatial Fourier frequencies; larger values reduce latent covariance approximation error")
    p.add_argument("--conditioning",choices=["mean","mixture"],default="mean",
                   help="mean = averaged parameters; mixture = per-member parameter class (adds between-class variability)")
    p.add_argument("--class-draw",choices=["fitted","shared"],default="fitted")
    p.add_argument("--weighting",choices=["tercile","pdf_ratio","mre","croley"],default="tercile",
                   help="mre is selected automatically for --total-window/--onset/--dry-spell/--cessation unless mre/croley is explicit")
    p.add_argument("--total-window",nargs=2,metavar=("MM-DD","MM-DD"),
                   help="Window of the main (total) forecast when it differs from --months, e.g. 07-01 09-30")
    p.add_argument("--onset",help="NetCDF: onset tercile probabilities (early, normal, late)")
    p.add_argument("--onset-search",nargs=2,default=["05-01","09-30"],metavar=("MM-DD","MM-DD"))
    p.add_argument("--dry-spell",help="NetCDF: post-onset max dry spell probabilities (short, normal, long)")
    p.add_argument("--dry-spell-days",type=int,default=50,help="Post-onset window for --dry-spell")
    p.add_argument("--cessation",help="NetCDF: cessation probabilities (early, normal, late)")
    p.add_argument("--tolerance",type=float,default=0.01,help="Probability tolerance tau of each constraint")
    p.add_argument("--constraint-mode", choices=["soft", "exact"], default="soft",
                   help="exact enforces feasible historical category masses; soft uses a penalty")
    p.add_argument("--amounts",choices=["gamma","mixed_exponential"],default="gamma")
    p.add_argument("--persistence",choices=["climatology","weighted","independent","yearly"],default="climatology")
    p.add_argument("--wet-rule", choices=["ge", "gt"], default="ge")
    p.add_argument("--amount-basis", choices=["excess", "raw"], default="excess")
    p.add_argument("--reset-each-month", action="store_true")
    p.add_argument("--tmin-forecast", help="Independent seasonal TMIN tercile probabilities NetCDF")
    p.add_argument("--tmax-forecast", help="Independent seasonal TMAX tercile probabilities NetCDF")
    p.add_argument("--occurrence",choices=["markov","spell"],default="markov",
                   help="spell = run-length dependent (semi-Markov) wet/dry sequence; use with --dry-spell")
    p.add_argument("--no-trace",action="store_true",help="Set sub-threshold rain to zero (v0.1 behaviour)")
    p.add_argument("--bbox",nargs=4,type=float,metavar=("WEST","SOUTH","EAST","NORTH"))
    p.add_argument("--scheduler",choices=["synchronous","threads"],default="synchronous")
    p.add_argument("--workers",type=int,default=2)
    for v in ["TMIN","TMAX","HUMIN","HUMAX","WIND","SOLAR"]:
        p.add_argument("--"+v.lower(),help=f"Daily {v} NetCDF on identical grid/time")
    args=p.parse_args(argv)
    if min(args.members, args.member_batch, args.workers, *args.tile) < 1:
        p.error("members, member-batch, workers and tile sizes must be positive")
    import dask
    paths={"PRCP":args.obs}
    paths.update({v:getattr(args,v.lower()) for v in ["TMIN","TMAX","HUMIN","HUMAX","WIND","SOLAR"] if getattr(args,v.lower())})
    obs=open_observations(paths,chunks={"T":366,"Y":args.tile[0],"X":args.tile[1]})
    if args.bbox:
        w,s,e,n=args.bbox
        obs=obs.where((obs.X>=w)&(obs.X<=e)&(obs.Y>=s)&(obs.Y<=n),drop=True)
    forecast=load_probabilities(args.forecast,target=obs)
    temperature_probabilities = {
        name: load_probabilities(path, target=obs)
        for name, path in (("TMIN", args.tmin_forecast), ("TMAX", args.tmax_forecast)) if path}
    constraints=_constraints(args)
    weighting=args.weighting if not constraints or args.weighting in ("mre","croley") else "mre"
    print(f"Generating {args.members} members on {obs.sizes['Y']} x {obs.sizes['X']} cells; baseline {args.climatology} (must match forecast provider).")
    with dask.config.set(scheduler=args.scheduler,num_workers=args.workers):
        result=generate_dask(obs,forecast,year=args.year,n_members=args.members,months=tuple(args.months),
            climatology=tuple(args.climatology),tile_shape=tuple(args.tile),member_batch=args.member_batch,
            seed=args.seed,wet_threshold=args.wet_threshold,spatial=args.spatial,
            paper_protocol=args.paper_protocol,n_features=args.n_features,
            conditioning=args.conditioning,class_draw=args.class_draw,weighting=weighting,
            constraint_mode=args.constraint_mode,temperature_probabilities=temperature_probabilities or None,
            wet_rule=args.wet_rule,amount_basis=args.amount_basis,reset_each_month=args.reset_each_month,
            constraints=constraints or None,occurrence=args.occurrence,
            amount_distribution=args.amounts,persistence=args.persistence,trace_rainfall=not args.no_trace)
        dest=Path(args.output)
        dest.parent.mkdir(parents=True,exist_ok=True)
        result.to_netcdf(dest,encoding={v:{"zlib":True,"complevel":4,
            "dtype":result[v].dtype if result[v].dtype.kind in "iu" else "float32"} for v in result})
    print(f"Written {dest}")

def _constraints(args):
    """Build SeasonalConstraints from the optional forecast files."""
    from .attributes import SeasonalTotal, OnsetDate, MaxDrySpell, CessationDate
    from .mre import SeasonalConstraint
    import xarray as xr
    def read(path):
        from .mre import relabel_probabilities
        with xr.open_dataset(path) as ds:
            candidates = []
            for da in ds.data_vars.values():
                try:
                    candidates.append(prepare_probabilities(relabel_probabilities(da)))
                except (ValueError, IndexError):
                    continue
            if len(candidates) != 1:
                raise ValueError(f"{path}: expected one labelled tercile-probability variable with one forecast time")
            return candidates[0].load()
    out=[]
    if not (args.total_window or args.onset or args.dry_spell or args.cessation):
        return out
    if args.total_window:
        out.append(SeasonalConstraint(SeasonalTotal(tuple(args.total_window)),None,args.tolerance))
    onset=OnsetDate(search=tuple(args.onset_search))
    if args.onset:
        out.append(SeasonalConstraint(onset,read(args.onset),args.tolerance))
    if args.dry_spell:
        out.append(SeasonalConstraint(MaxDrySpell(after_onset=onset,length_days=args.dry_spell_days),read(args.dry_spell),args.tolerance))
    if args.cessation:
        out.append(SeasonalConstraint(CessationDate(),read(args.cessation),args.tolerance))
    return out

if __name__ == "__main__":
    main()
