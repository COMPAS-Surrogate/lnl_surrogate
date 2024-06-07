from .lnl_surrogate import LnLSurrogate
from .sample import run_sampler
from .train import train


def build_surrogate(
    csv: str,
    model_type: str = "gp",
    label: str = "",
    outdir: str = "outdir",
    plots: bool = True,
    lnl_threshold: float = None,
    sample: bool = True,
):
    surrogate = LnLSurrogate.from_csv(
        csv,
        model_type,
        label=label,
        outdir=outdir,
        plot=plots,
        lnl_threshold=lnl_threshold,
    )
    if sample:
        run_sampler(surrogate, outdir=outdir, label=label, verbose=True)
