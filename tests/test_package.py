import rwalang

def test_import_package():
    """Testing that the rwalang package can be imported"""
    assert rwalang is not None
    assert hasattr(rwalang, 'detector')
    assert hasattr(rwalang, 'linguistic_features')
    assert hasattr(rwalang, 'utils')