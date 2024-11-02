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
    assert len(card_low) == 2
    assert len(card_high) == 0