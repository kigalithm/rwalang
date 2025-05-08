import pytest
from kgt_rwalang import detector


# Get the directory of the current test file
# Using a pytest fixture is a clean way to provide this to multiple tests
@pytest.fixture
def detector_instance():
    """Provides a Decorator instance for tests."""
    return detector.KinyaLangDetector()


def test_detector_instantiation(detector_instance):
    """Test that Detector can be instantiated."""
    assert detector_instance is not None
    assert detector_instance.model is None


# Test loading the model using importlib.resources
# This test assumes the model file is correctly included as package data
# and the MODEL_RESOURCE_PATH is correctly defined in rwalang/config.py
def test_load_model_success(detector_instance):
      """Test that load_model successfully loads the model."""
      # Assuming MODEL_RESOURCE_PATH is defined in rwalang/config
      # Call load_model() with the default model
      success = detector_instance.load_model()

      assert success is True
      assert detector_instance.model is not None


# TODO: More tests need to be added