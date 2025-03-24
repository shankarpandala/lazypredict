"""Tests for helper functions."""

import unittest

import numpy as np
import pandas as pd


class TestHelpers(unittest.TestCase):
    """Test helper functions."""

    def test_get_card_split(self):
        """Test categorical cardinality threshold function."""
        from lazypredict.utils.preprocessing import (
            categorical_cardinality_threshold,
        )

        # Create test data with known cardinality
        data = pd.DataFrame(
            {
                "low_card": [
                    "A",
                    "B",
                    "A",
                    "A",
                    "B",
                    "A",
                    "B",
                    "A",
                    "B",
                    "A",
                ],  # 2 unique values
                "high_card": list("ABCDEFGHIJ"),  # 10 unique values
            }
        )

        # Test with threshold=5 (should split correctly)
        low_card, high_card = categorical_cardinality_threshold(
            data, ["low_card", "high_card"], threshold=5
        )

        # Check results
        self.assertEqual(low_card, ["low_card"])
        self.assertEqual(high_card, ["high_card"])

        # Verify specific values
        self.assertIn("low_card", low_card)
        self.assertIn("high_card", high_card)

    def test_adjusted_rsquared(self):
        """Test adjusted R-squared calculation."""
        from lazypredict.utils.metrics import adjusted_rsquared

        # Test case 1: Perfect prediction
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        n_features = 2
        adj_r2 = adjusted_rsquared(y_true, y_pred, n_features)
        self.assertAlmostEqual(adj_r2, 1.0)

        # Test case 2: Imperfect prediction
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        adj_r2 = adjusted_rsquared(y_true, y_pred, n_features)
        self.assertLess(adj_r2, 1.0)
        self.assertGreater(adj_r2, 0.0)


if __name__ == "__main__":
    unittest.main()
