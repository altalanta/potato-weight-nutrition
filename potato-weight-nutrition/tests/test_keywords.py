"""Test keyword extraction functionality."""

import pandas as pd
import pytest

from potato_pipeline.features import build_keywords, KEYWORDS


def test_keywords_defined():
    """Test that all expected keywords are defined."""
    expected_keywords = {
        "potato", "beans", "rice", "oats", "bread", 
        "fruit", "vegetables", "dairy", "meat", "egg", "nuts"
    }
    assert set(KEYWORDS.keys()) == expected_keywords


def test_keyword_patterns():
    """Test that keyword patterns match expected food items correctly."""
    # Test data with various food descriptions
    test_data = pd.DataFrame({
        'subject_id': ['S1', 'S1', 'S2', 'S2', 'S3'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03']),
        'food': [
            'baked potato with butter',
            'black beans and rice',
            'Chickpeas with vegetables',
            'Lentil soup',
            'Regular bread and eggs'
        ]
    })
    
    result = build_keywords(test_data)
    
    # Check that we get the expected structure
    assert 'subject_id' in result.columns
    assert 'date' in result.columns
    
    # Check keyword columns exist
    for keyword in KEYWORDS.keys():
        col_name = f'kw_{keyword}'
        assert col_name in result.columns
    
    # Check specific matches
    # Day 1 (S1): should have potato=1, beans=1, rice=1
    day1 = result[(result['subject_id'] == 'S1') & (result['date'] == '2023-01-01')]
    assert len(day1) == 1
    assert day1['kw_potato'].iloc[0] == 1  # "baked potato"
    assert day1['kw_beans'].iloc[0] == 1   # "black beans"
    assert day1['kw_rice'].iloc[0] == 1    # "rice"
    
    # Day 2 (S2): should have beans=1, vegetables=1
    day2 = result[(result['subject_id'] == 'S2') & (result['date'] == '2023-01-02')]
    assert len(day2) == 1
    assert day2['kw_beans'].iloc[0] == 1      # "chickpeas" and "lentil"
    assert day2['kw_vegetables'].iloc[0] == 1 # "vegetables"
    assert day2['kw_potato'].iloc[0] == 0     # No potato
    
    # Day 3 (S3): should have bread=1, egg=1
    day3 = result[(result['subject_id'] == 'S3') & (result['date'] == '2023-01-03')]
    assert len(day3) == 1
    assert day3['kw_bread'].iloc[0] == 1  # "bread"
    assert day3['kw_egg'].iloc[0] == 1    # "eggs"


def test_case_insensitive_matching():
    """Test that keyword matching is case insensitive."""
    test_data = pd.DataFrame({
        'subject_id': ['S1'],
        'date': pd.to_datetime(['2023-01-01']),
        'food': ['BAKED POTATO and Black Beans']
    })
    
    result = build_keywords(test_data)
    
    assert result['kw_potato'].iloc[0] == 1
    assert result['kw_beans'].iloc[0] == 1


def test_plural_and_singular_matching():
    """Test that both plural and singular forms are matched."""
    test_data = pd.DataFrame({
        'subject_id': ['S1', 'S2', 'S3', 'S4'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'food': [
            'potato salad',      # singular
            'potatoes mashed',   # plural
            'black bean soup',   # singular
            'red beans and rice' # plural
        ]
    })
    
    result = build_keywords(test_data)
    
    # Both singular and plural should be detected
    assert result['kw_potato'].sum() == 2  # Both potato and potatoes
    assert result['kw_beans'].sum() == 2   # Both bean and beans


def test_word_boundary_matching():
    """Test that word boundaries are respected in matching."""
    test_data = pd.DataFrame({
        'subject_id': ['S1', 'S2'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'food': [
            'potato chips',        # Should match 'potato'
            'sweet potato bread'   # Should match both 'potato' and 'bread'
        ]
    })
    
    result = build_keywords(test_data)
    
    # Both should match potato
    assert result['kw_potato'].sum() == 2
    
    # Only second should match bread
    assert result['kw_bread'].sum() == 1


def test_empty_input():
    """Test handling of empty input."""
    empty_df = pd.DataFrame()
    result = build_keywords(empty_df)
    
    assert result.empty


def test_missing_food_column():
    """Test handling of missing food column."""
    test_data = pd.DataFrame({
        'subject_id': ['S1'],
        'date': pd.to_datetime(['2023-01-01']),
        'other_column': ['some value']
    })
    
    result = build_keywords(test_data)
    
    # Should return the input dataframe unchanged if no food column
    assert result.equals(test_data)


def test_null_food_values():
    """Test handling of null food values."""
    test_data = pd.DataFrame({
        'subject_id': ['S1', 'S2'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-01']),
        'food': ['potato salad', None]
    })
    
    result = build_keywords(test_data)
    
    # Should handle null values gracefully (both rows processed, one with null food)
    assert len(result) == 2  # Both subject-dates processed
    assert result['kw_potato'].iloc[0] == 1  # First row has potato
    assert result['kw_potato'].iloc[1] == 0  # Second row (null food) has no potato


if __name__ == "__main__":
    pytest.main([__file__])