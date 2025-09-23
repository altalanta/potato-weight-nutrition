"""Test week alignment functionality."""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from potato_pipeline.features import prepare_weight, aggregate_weekly, align_weeks


def test_prepare_weight_basic():
    """Test basic weight trajectory preparation."""
    # Create test tidy data
    tidy_data = pd.DataFrame({
        'subject_id': ['S1', 'S1', 'S1', 'S2', 'S2'],
        'metric': ['weight', 'weight', 'weight', 'weight', 'weight'],
        'phase': ['start', 'end', 'end', 'start', 'end'],
        'week': [0, 4, 8, 0, 4],
        'value': [70.0, 68.5, 67.0, 85.0, 83.5],
        'note': [None, None, None, None, None]
    })
    
    result = prepare_weight(tidy_data)
    
    # Check basic structure
    assert 'subject_id' in result.columns
    assert 'week' in result.columns
    assert 'value' in result.columns
    assert 'start_weight' in result.columns
    assert 'delta_kg' in result.columns
    assert 'pct_change' in result.columns
    
    # Check calculations for S1
    s1_data = result[result['subject_id'] == 'S1'].sort_values('week')
    assert len(s1_data) == 3
    
    # Baseline should be 70.0
    assert s1_data['start_weight'].iloc[0] == 70.0
    
    # Delta calculations
    expected_deltas = [0.0, -1.5, -3.0]  # 70-70, 68.5-70, 67-70
    for i, expected_delta in enumerate(expected_deltas):
        assert abs(s1_data['delta_kg'].iloc[i] - expected_delta) < 0.001
    
    # Percentage changes
    expected_pcts = [0.0, -2.14, -4.29]  # Approximately
    for i, expected_pct in enumerate(expected_pcts):
        assert abs(s1_data['pct_change'].iloc[i] - expected_pct) < 0.1


def test_week_alignment_basic():
    """Test basic week alignment between weight and nutrition data."""
    # Create weight trajectories
    weight_data = pd.DataFrame({
        'subject_id': ['S1', 'S1', 'S1'],
        'week': [0, 4, 8],
        'value': [70.0, 68.5, 67.0],
        'start_weight': [70.0, 70.0, 70.0],
        'delta_kg': [0.0, -1.5, -3.0],
        'pct_change': [0.0, -2.14, -4.29],
        'phase': ['start', 'end', 'end'],
        'note': [None, None, None]
    })
    
    # Create weekly nutrition data
    weekly_nutrition = pd.DataFrame({
        'subject_id': ['S1', 'S1', 'S1'],
        'week_index': [0, 4, 8],
        'fiber_g_total_week_mean': [15.0, 20.0, 25.0],
        'calories_kcal_total_week_mean': [1800, 1750, 1700]
    })
    
    weight_traj, weekly_nutr, analysis_df = align_weeks(weight_data, weekly_nutrition)
    
    # Check that merge worked
    assert len(analysis_df) == 3
    assert 'fiber_g_total_week_mean' in analysis_df.columns
    assert 'calories_kcal_total_week_mean' in analysis_df.columns
    
    # Check that week alignment is correct
    for i in range(3):
        row = analysis_df.iloc[i]
        expected_week = [0, 4, 8][i]
        assert row['week'] == expected_week
        assert row['fiber_g_total_week_mean'] == [15.0, 20.0, 25.0][i]
    
    # Check that delta_kg_next is computed (lead of delta_kg)
    assert 'delta_kg_next' in analysis_df.columns
    assert analysis_df['delta_kg_next'].iloc[0] == -1.5  # Next week's delta
    assert analysis_df['delta_kg_next'].iloc[1] == -3.0  # Next week's delta
    assert pd.isna(analysis_df['delta_kg_next'].iloc[2])  # Last observation


def test_aggregate_weekly_basic():
    """Test weekly aggregation of daily data."""
    # Create daily data spanning multiple weeks
    base_date = datetime(2023, 1, 1)
    daily_data = pd.DataFrame({
        'subject_id': ['S1'] * 14,  # 2 weeks of data
        'date': [base_date + timedelta(days=i) for i in range(14)],
        'fiber_g_total': [10, 12, 15, 8, 20, 18, 16,  # Week 0
                         22, 24, 19, 21, 25, 23, 20], # Week 1
        'calories_kcal_total': [1800] * 7 + [1750] * 7,
        'kw_beans': [1, 0, 0, 1, 0, 0, 0,  # Week 0: 2/7 days
                    0, 1, 0, 0, 1, 1, 0]   # Week 1: 3/7 days
    })
    
    result = aggregate_weekly(daily_data)
    
    # Should have 2 weeks of data
    assert len(result) == 2
    assert 'subject_id' in result.columns
    assert 'week_index' in result.columns
    
    # Check week 0 aggregation
    week0 = result[result['week_index'] == 0].iloc[0]
    expected_fiber_mean_w0 = sum([10, 12, 15, 8, 20, 18, 16]) / 7
    assert abs(week0['fiber_g_total_week_mean'] - expected_fiber_mean_w0) < 0.001
    
    # Check keyword aggregation (share of days)
    expected_beans_share_w0 = 2/7  # 2 days out of 7
    assert abs(week0['share_of_days_kw_beans'] - expected_beans_share_w0) < 0.001
    assert week0['any_kw_beans'] == 1  # Any day had beans
    
    # Check week 1
    week1 = result[result['week_index'] == 1].iloc[0]
    expected_beans_share_w1 = 3/7  # 3 days out of 7
    assert abs(week1['share_of_days_kw_beans'] - expected_beans_share_w1) < 0.001


def test_alignment_with_missing_nutrition():
    """Test alignment when nutrition data is missing for some weeks."""
    weight_data = pd.DataFrame({
        'subject_id': ['S1', 'S1', 'S1'],
        'week': [0, 4, 8],
        'value': [70.0, 68.5, 67.0],
        'delta_kg': [0.0, -1.5, -3.0]
    })
    
    # Only have nutrition for week 4
    weekly_nutrition = pd.DataFrame({
        'subject_id': ['S1'],
        'week_index': [4],
        'fiber_g_total_week_mean': [20.0],
        'calories_kcal_total_week_mean': [1750]
    })
    
    weight_traj, weekly_nutr, analysis_df = align_weeks(weight_data, weekly_nutrition)
    
    # Should still have all 3 weight observations
    assert len(analysis_df) == 3
    
    # Week 4 should have nutrition data
    week4_row = analysis_df[analysis_df['week'] == 4].iloc[0]
    assert week4_row['fiber_g_total_week_mean'] == 20.0
    
    # Weeks 0 and 8 should have NaN for nutrition
    week0_row = analysis_df[analysis_df['week'] == 0].iloc[0]
    week8_row = analysis_df[analysis_df['week'] == 8].iloc[0]
    assert pd.isna(week0_row['fiber_g_total_week_mean'])
    assert pd.isna(week8_row['fiber_g_total_week_mean'])


def test_multiple_subjects():
    """Test alignment with multiple subjects."""
    weight_data = pd.DataFrame({
        'subject_id': ['S1', 'S1', 'S2', 'S2'],
        'week': [0, 4, 0, 4],
        'value': [70.0, 68.5, 80.0, 79.0],
        'delta_kg': [0.0, -1.5, 0.0, -1.0]
    })
    
    weekly_nutrition = pd.DataFrame({
        'subject_id': ['S1', 'S1', 'S2', 'S2'],
        'week_index': [0, 4, 0, 4],
        'fiber_g_total_week_mean': [15.0, 20.0, 12.0, 18.0]
    })
    
    weight_traj, weekly_nutr, analysis_df = align_weeks(weight_data, weekly_nutrition)
    
    # Should have 4 observations (2 per subject)
    assert len(analysis_df) == 4
    
    # Check that subjects are handled separately
    s1_data = analysis_df[analysis_df['subject_id'] == 'S1']
    s2_data = analysis_df[analysis_df['subject_id'] == 'S2']
    
    assert len(s1_data) == 2
    assert len(s2_data) == 2
    
    # Check delta_kg_next is computed within subjects
    s1_sorted = s1_data.sort_values('week')
    assert s1_sorted['delta_kg_next'].iloc[0] == -1.5  # S1 week 0 -> week 4
    assert pd.isna(s1_sorted['delta_kg_next'].iloc[1])  # S1 week 4 (last)


def test_empty_inputs():
    """Test handling of empty inputs."""
    empty_df = pd.DataFrame()
    
    # Empty weight data
    weight_traj, weekly_nutr, analysis_df = align_weeks(empty_df, empty_df)
    assert analysis_df.empty
    
    # Empty nutrition data but valid weight data
    weight_data = pd.DataFrame({
        'subject_id': ['S1'],
        'week': [0],
        'value': [70.0],
        'delta_kg': [0.0]
    })
    
    weight_traj, weekly_nutr, analysis_df = align_weeks(weight_data, empty_df)
    assert len(analysis_df) == 1
    assert analysis_df['subject_id'].iloc[0] == 'S1'


if __name__ == "__main__":
    pytest.main([__file__])