from lnl_surrogate.surrogate.lnl_surrogate import LnLSurrogate


def test_lnlsurrogate_from_csv(training_csv, tmpdir):
    lnl_surr = LnLSurrogate.from_csv(
        training_csv, model_type="gp", label="test"
    )
    lnl_surr.plot(outdir=tmpdir)
