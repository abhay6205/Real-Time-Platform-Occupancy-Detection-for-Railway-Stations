import pytest
from counter.density_classifier import DensityClassifier

def test_density_classifier():
    classifier = DensityClassifier(low_max=15, high_min=41)

    # Test count 0
    label, color = classifier.classify(0)
    assert label == "Low"
    assert color.startswith("#")

    # Test count 14
    label, color = classifier.classify(14)
    assert label == "Low"
    assert color.startswith("#")

    # Test count 15
    label, color = classifier.classify(15)
    assert label == "Medium"
    assert color.startswith("#")

    # Test count 40
    label, color = classifier.classify(40)
    assert label == "Medium"
    assert color.startswith("#")

    # Test count 41
    label, color = classifier.classify(41)
    assert label == "High"
    assert color.startswith("#")

    # Test count 100
    label, color = classifier.classify(100)
    assert label == "High"
    assert color.startswith("#")
