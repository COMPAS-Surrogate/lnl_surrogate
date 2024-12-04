from lnl_computer.cosmic_integration.star_formation_paramters import (
    DEFAULT_DICT,
)

from lnl_surrogate.surrogate.train import train


def test_lvk_trainer(mock_data, tmpdir):
    train(
        model_type="gp",
        compas_h5_filename=mock_data.compas_filename,
        params=["aSF"],
        mcz_obs_filename="LVK",
        duration=1,
        n_init=3,
        n_rounds=2,
        n_pts_per_round=1,
        outdir=f"{tmpdir}/out_train_lvk",
        verbose=3,
        reference_param=DEFAULT_DICT,
    )
