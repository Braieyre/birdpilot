import unittest

import numpy as np
from PIL import Image

from src.close_view_contract import change_gate, change_metrics, roi_pixels, validate_normalized_roi


class CloseViewContractTests(unittest.TestCase):
    def test_roi_rounding_and_validation(self) -> None:
        self.assertEqual(roi_pixels((0.1, 0.2, 0.9, 0.8), 101, 51), (10, 10, 91, 41))
        with self.assertRaises(ValueError):
            validate_normalized_roi((0.5, 0.0, 0.5, 1.0))

    def test_identical_background_stays_closed(self) -> None:
        image = Image.new("RGB", (80, 60), (100, 100, 100))
        metrics = change_metrics(image, image, (0, 0, 1, 1), blur_radius=0)
        self.assertEqual(metrics["mean_absolute_change"], 0)
        self.assertFalse(change_gate(metrics, mean_threshold=0.04, fraction_threshold=0.08))

    def test_local_subject_opens_gate_inside_roi(self) -> None:
        background = Image.new("RGB", (100, 100), (0, 0, 0))
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        array[35:65, 35:65] = 255
        current = Image.fromarray(array)
        metrics = change_metrics(current, background, (0.25, 0.25, 0.75, 0.75), blur_radius=0)
        self.assertGreater(metrics["changed_pixel_fraction"], 0.08)
        self.assertTrue(change_gate(metrics, mean_threshold=0.04, fraction_threshold=0.08))

    def test_change_outside_roi_does_not_open_gate(self) -> None:
        background = Image.new("RGB", (100, 100), (0, 0, 0))
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        array[:20, :20] = 255
        metrics = change_metrics(
            Image.fromarray(array), background, (0.4, 0.4, 1.0, 1.0), blur_radius=0
        )
        self.assertFalse(change_gate(metrics, mean_threshold=0.04, fraction_threshold=0.08))


if __name__ == "__main__":
    unittest.main()
