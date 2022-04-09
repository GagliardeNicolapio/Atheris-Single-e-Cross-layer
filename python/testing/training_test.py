import unittest
import numpy as np
from python.training import or_aggregation, and_aggregation


class TestAggregation(unittest.TestCase):
    def test_different_length(self):
        # check different size
        self.assertRaises(ValueError, and_aggregation, np.array([2]), np.array([1, 2, 3]))
        self.assertRaises(ValueError, and_aggregation, np.array([1, 2, 3]), np.array([3]))
        self.assertRaises(ValueError, or_aggregation, np.array([4]), np.array([1, 2, 3]))
        self.assertRaises(ValueError, or_aggregation, np.array([1, 2, 3]), np.array([2]))

    def test_same_size(self):
        try:
            and_aggregation(np.array([1, 2, 3]), np.array([1, 2, 3]))
            or_aggregation(np.array([1, 2, 3]), np.array([1, 2, 3]))
        except ValueError:
            self.fail("check same size failed")

    def test_zero_length(self):
        self.assertRaises(ValueError, and_aggregation, np.array([]), np.array([]))
        self.assertRaises(ValueError, and_aggregation, np.array([1, 2, 3]), np.array([]))
        self.assertRaises(ValueError, and_aggregation, np.array([]), np.array([1, 2, 3]))

        self.assertRaises(ValueError, or_aggregation, np.array([]), np.array([]))
        self.assertRaises(ValueError, or_aggregation, np.array([]), np.array([1, 2, 3]))
        self.assertRaises(ValueError, or_aggregation, np.array([1, 2, 3]), np.array([]))

    def test_and_aggregation(self):
        self.assertEqual(np.array_equal(and_aggregation(np.array([0]), np.array([0])), np.array([0])), True)
        self.assertEqual(np.array_equal(and_aggregation(np.array([0]), np.array([1])), np.array([0])), True)
        self.assertEqual(np.array_equal(and_aggregation(np.array([1]), np.array([0])), np.array([0])), True)
        self.assertEqual(np.array_equal(and_aggregation(np.array([1]), np.array([1])), np.array([1])), True)

        self.assertEqual(np.array_equal(and_aggregation(np.array([0, 0]), np.array([0, 0])), np.array([0, 0])), True)
        self.assertEqual(np.array_equal(and_aggregation(np.array([1, 1]), np.array([1, 1])), np.array([1, 1])), True)
        self.assertEqual(np.array_equal(and_aggregation(np.array([1, 0]), np.array([0, 1])), np.array([0, 0])), True)
        self.assertEqual(np.array_equal(and_aggregation(np.array([0, 1]), np.array([1, 0])), np.array([0, 0])), True)

    def test_or_aggregation(self):
        self.assertEqual(np.array_equal(or_aggregation(np.array([0]), np.array([0])), np.array([0])), True)
        self.assertEqual(np.array_equal(or_aggregation(np.array([0]), np.array([1])), np.array([1])), True)
        self.assertEqual(np.array_equal(or_aggregation(np.array([1]), np.array([0])), np.array([1])), True)
        self.assertEqual(np.array_equal(or_aggregation(np.array([1]), np.array([1])), np.array([1])), True)

        self.assertEqual(np.array_equal(or_aggregation(np.array([0, 0]), np.array([0, 0])), np.array([0, 0])), True)
        self.assertEqual(np.array_equal(or_aggregation(np.array([1, 1]), np.array([1, 1])), np.array([1, 1])), True)
        self.assertEqual(np.array_equal(or_aggregation(np.array([1, 0]), np.array([0, 1])), np.array([1, 1])), True)
        self.assertEqual(np.array_equal(or_aggregation(np.array([0, 1]), np.array([1, 0])), np.array([1, 1])), True)


if __name__ == '__main__':
    unittest.main()
