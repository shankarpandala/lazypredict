import pytest
import pandas as pd
from lazypredict.Supervised import get_card_split

def test_get_card_split():
    df = pd.DataFrame({
        'A': ['a', 'b', 'c', 'd', 'e'],
        'B': ['f', 'g', 'h', 'i', 'j'],
        'C': [1, 2, 3, 4, 5]
    })
    cols = ['A', 'B']
    card_low, card_high = get_card_split(df, cols, n=3)
    # Both columns have 5 unique values, which is > 3, so they go to card_high
    assert len(card_low) == 0
    assert len(card_high) == 2