import pytest
from counter.person_counter import PersonCounter

def test_person_counter():
    counter = PersonCounter(ema_alpha=0.3)

    # Initial state
    assert counter.smoothed_count == 0.0

    # Update with 10 detections
    detections = [{} for _ in range(10)]
    result = counter.update(detections)

    # Ensure it returns an integer
    assert isinstance(result, int)

    # Ensure smoothed count is updated properly (0.3*10 + 0.7*0.0 = 3.0)
    assert counter.smoothed_count == 3.0
    assert result == 3

    # Ensure smoothing keeps result between 0 and raw count (10)
    assert 0 <= counter.smoothed_count <= 10

    # Test repeated updates converge toward raw count
    for _ in range(20):
        result = counter.update(detections)

    # Should be close to 10
    assert round(counter.smoothed_count) == 10

    # Test reset
    counter.reset()
    assert counter.smoothed_count == 0.0
