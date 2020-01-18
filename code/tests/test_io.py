"""
Test the input and output functions
"""
import pathlib

import numpy.testing as npt
from facebias.io import load_bfw_datatable

DATADIR = pathlib.Path(__file__).parent.parent.parent.joinpath("data")


def test_bias_dataframe():
    "Make sure the data is loaded properly"
    fname = DATADIR.joinpath("bfw-datatable.pkl")
    data = load_bfw_datatable(fname)
    assert len(data) == 923898
    npt.assert_allclose(data["score"].mean(), 0.5864211374778945)
