"""Import sanity checks for quality modules."""


def test_ensemble_classifier_import():
    """Ensure ensemble_classifier module loads without syntax errors."""
    from pdf_yaml_pipeline.quality.ensemble_classifier import (
        ClassifierThresholdConfig,
        EnsembleClassifier,
        ThresholdConfig,
    )

    assert ClassifierThresholdConfig is ThresholdConfig
    assert EnsembleClassifier is not None
