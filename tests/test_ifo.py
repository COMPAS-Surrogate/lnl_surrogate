import bilby


def test_ifo():
    ifos = bilby.gw.detector.InterferometerList(["H1"])
