import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolox_contract import box_iou, decode_birds, nms, padded_box, preprocess_onnx, preprocess_rknn


class YoloXContractTest(unittest.TestCase):
    def test_preprocessing_layout_and_color(self):
        rgb = np.zeros((2, 4, 3), dtype=np.uint8)
        rgb[0, 0] = [10, 20, 30]
        onnx, onnx_scale = preprocess_onnx(rgb, 4)
        rknn, rknn_scale = preprocess_rknn(rgb, 4)
        self.assertEqual(onnx.shape, (1, 3, 4, 4))
        self.assertEqual(rknn.shape, (1, 4, 4, 3))
        np.testing.assert_array_equal(onnx[0, :, 0, 0], [30, 20, 10])
        np.testing.assert_array_equal(rknn[0, 0, 0], [30, 20, 10])
        self.assertEqual(onnx_scale, rknn_scale)
        np.testing.assert_array_equal(rknn[0, 3, 3], [114, 114, 114])

    def test_nms_and_iou(self):
        boxes = np.asarray([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=float)
        self.assertAlmostEqual(float(box_iou(boxes[0], boxes[1:2])[0]), 0.64)
        self.assertEqual(nms(boxes, np.asarray([0.9, 0.8, 0.7]), 0.45), [0, 2])

    def test_decode_and_coordinate_clamp(self):
        output = np.zeros((1, 3549, 85), dtype=np.float32)
        output[0, 0, 4] = 1.0
        output[0, 0, 5 + 14] = 0.9
        output[0, 0, 2:4] = np.log([2.0, 2.0])
        detections = decode_birds(output, 0.25, 0.45, 416, 1.0, 100, 80)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["box_xyxy"][:2], [0.0, 0.0])
        self.assertEqual(padded_box([-2, -3, 102, 83], 100, 80), (0, 0, 100, 80))
        self.assertEqual(padded_box([17, 17, 31, 31], 100, 100, padding=0), (16, 16, 32, 32))

    def test_rejects_unknown_output_shape(self):
        with self.assertRaises(ValueError):
            decode_birds(np.zeros((1, 10, 85)), 0.25, 0.45, 416, 1.0, 100, 100)


if __name__ == "__main__":
    unittest.main()
