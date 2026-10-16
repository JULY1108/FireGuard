import unittest

from fireguard_core import Detection, deduplicate_detections


class DetectionDedupTest(unittest.TestCase):
    def test_overlap_containment_classes_and_separate_targets(self):
        main = Detection("smoke", .9, 0, 0, 100, 100)
        overlap = Detection("smoke", .8, 20, 0, 120, 100)
        contained = Detection("smoke", .7, 30, 30, 50, 50)
        separate = Detection("smoke", .6, 200, 0, 300, 100)
        fire = Detection("fire", .5, 30, 30, 50, 50)
        invalid = Detection("smoke", .99, 10, 10, 10, 20)
        boxes = [contained, separate, overlap, fire, invalid, main]
        self.assertEqual(deduplicate_detections(boxes, .45), [main, separate, fire])
        self.assertEqual(deduplicate_detections([], .45), [])
        self.assertEqual(deduplicate_detections([main, overlap], .9), [main, overlap])


if __name__ == "__main__":
    unittest.main()
