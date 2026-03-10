import pandas as pd
from lazypredict.Supervised import get_card_split


def test_get_card_split():
    df = pd.DataFrame({
        'A': ['a', 'b', 'c'],
        'B': ['f', 'g', 'h'],
    })
    # Duplicate rows to have more data but same cardinality
    df = pd.concat([df] * 5, ignore_index=True)
    # Add a high cardinality column
    df['D'] = ['cat_' + str(i) for i in range(len(df))]

    cols = pd.Index(['A', 'B', 'D'])
    card_low, card_high = get_card_split(df, cols, n=5)
    # A (3 unique) and B (3 unique) have cardinality <= 5 => low
    # D (15 unique) has cardinality > 5 => high
    assert len(card_low) == 2
    assert len(card_high) == 1
    assert 'D' in card_high


def test_get_card_split_with_list():
    """Test that get_card_split works with plain list of column names."""
    df = pd.DataFrame({
        'A': ['a', 'b'],
        'B': ['x' + str(i) for i in range(20)][:2],
    })
    df = pd.concat([df] * 10, ignore_index=True)
    df['C'] = ['c_' + str(i) for i in range(len(df))]

    cols = ['A', 'C']
    card_low, card_high = get_card_split(df, cols, n=5)
    assert 'A' in card_low
    assert 'C' in card_high
