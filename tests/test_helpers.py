import pytest
import pandas as pd
from lazypredict.Supervised import get_card_split, adjusted_rsquared
import unittest

def test_get_card_split():
    df = pd.DataFrame({
        'A': ['a', 'b', 'c', 'd', 'e'],
        'B': ['f', 'g', 'h', 'i', 'j'],
        'C': [1, 2, 3, 4, 5]
    })
    cols = ['A', 'B']
    card_low, card_high = get_card_split(df, cols, n=3)
    assert len(card_low) == 2
    assert len(card_high) == 0

class TestHelpers(unittest.TestCase):
    def test_get_card_split(self):
        df = pd.DataFrame({
            'A': ['a', 'b', 'c', 'a', 'b', 'c'],
            'B': ['x', 'y', 'z', 'x', 'y', 'z'],
            'C': [1, 2, 3, 4, 5, 6]
        })
        card_low, card_high = get_card_split(df, ['A', 'B'], n=2)
        self.assertEqual(card_low, ['A', 'B'])
        self.assertEqual(card_high, [])

    def test_adjusted_rsquared(self):
        r2 = 0.8
        n = 100
        p = 5
        adj_r2 = adjusted_rsquared(r2, n, p)
        self.assertAlmostEqual(adj_r2, 0.796, places=3)

if __name__ == '__main__':
    unittest.main()