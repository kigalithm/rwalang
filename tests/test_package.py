import kgt_rwalang

def test_import_package():
    """Testing that the rwalang package can be imported"""
    assert kgt_rwalang is not None
    assert hasattr(kgt_rwalang, 'detector')
    assert hasattr(kgt_rwalang, 'linguistic_features')
    # assert hasattr(rwalang, 'utils')