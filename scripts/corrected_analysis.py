#!/usr/bin/env python3
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_participant_data(file_path):
    """Extract weight data from a single participant's Sheet1 CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Get participant info from metadata columns
        participant_id = df['participant_id'].iloc[0] if 'participant_id' in df.columns else None
        category = df['category'].iloc[0] if 'category' in df.columns else None
        
        # Find the weight row
        weight_row_idx = None
        for idx, row in df.iterrows():
            # Check if any cell in the row contains "Weight in the Morning"
            if row.astype(str).str.contains('Weight in the Morning', na=False).any():
                weight_row_idx = idx
                break
        
        if weight_row_idx is None:
            return None
        
        # Extract weights from the weight row
        weight_row = df.iloc[weight_row_idx]
        weights = []
        
        # Start from column 4 (after the first few descriptive columns)
        for col_idx in range(4, len(weight_row) - 3):  # Exclude the last 3 metadata columns
            value = weight_row.iloc[col_idx]
            if pd.notna(value):
                try:
                    weight_val = float(value)
                    # Validate weight is in reasonable range
                    if 80 <= weight_val <= 400:  # Reasonable human weight range
                        weights.append(weight_val)
                    else:
                        # Stop when we hit unreasonable values (likely end of data)
                        break
                except (ValueError, TypeError):
                    # Stop when we hit non-numeric values
                    break
        
        if len(weights) < 2:  # Need at least 2 measurements
            return None
        
        # Calculate metrics
        weight_change = weights[-1] - weights[0]
        weight_change_percent = (weight_change / weights[0]) * 100
        
        return {
            'participant_id': participant_id,
            'category': category,
            'starting_weight': weights[0],
            'ending_weight': weights[-1],
            'weight_change': weight_change,
            'weight_change_percent': weight_change_percent,
            'num_measurements': len(weights),
            'weights': weights
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def analyze_all_participants():
    """Analyze all participants from the complete dataset"""
    csv_dir = "/Users/artemisfolle/Documents/Coding_Projects/potato_csv_data_complete"
    sheet1_files = glob.glob(os.path.join(csv_dir, "*_Sheet1.csv"))
    
    print(f"Processing {len(sheet1_files)} participant files...")
    
    participants = []
    errors = 0
    
    for file_path in sheet1_files:
        result = extract_participant_data(file_path)
        if result:
            participants.append(result)
        else:
            errors += 1
    
    print(f"Successfully processed {len(participants)} participants ({errors} errors)")
    return participants

def create_corrected_visualization(participants):
    """Create a corrected visualization with proper data"""
    
    if not participants:
        print("No valid participant data found!")
        return None
    
    # Extract data
    weight_changes = [p['weight_change'] for p in participants]
    starting_weights = [p['starting_weight'] for p in participants]
    categories = [p['category'] for p in participants]
    num_measurements = [p['num_measurements'] for p in participants]
    
    # Create duration categories based on number of measurements
    duration_categories = []
    for num in num_measurements:
        if num >= 28:  # 4+ weeks
            duration_categories.append('4+ Weeks')
        elif num >= 14:  # 2-4 weeks
            duration_categories.append('2-4 Weeks')
        elif num >= 7:   # 1-2 weeks
            duration_categories.append('1-2 Weeks')
        else:            # < 1 week
            duration_categories.append('< 1 Week')
    
    # Create visualization
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Potato Diet Study Results - Weight & Nutrition Analysis', fontsize=16, fontweight='bold')
    
    # Color palette
    colors = ['#8B4513', '#CD853F', '#DEB887', '#F4A460']
    
    # Plot 1: Weight change distribution
    ax1.hist(weight_changes, bins=min(15, len(weight_changes)//2), alpha=0.7, 
             color='#8B4513', edgecolor='black', linewidth=1)
    ax1.set_xlabel('Weight Change (lbs)', fontsize=11)
    ax1.set_ylabel('Number of Participants', fontsize=11)
    ax1.set_title('Distribution of Weight Changes', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight change vs starting weight by duration
    unique_durations = list(set(duration_categories))
    duration_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_durations)}
    
    for dur in unique_durations:
        dur_indices = [i for i, d in enumerate(duration_categories) if d == dur]
        dur_starting = [starting_weights[i] for i in dur_indices]
        dur_changes = [weight_changes[i] for i in dur_indices]
        ax2.scatter(dur_starting, dur_changes, alpha=0.7, 
                   label=dur, color=duration_colors[dur], s=60, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('Starting Weight (lbs)', fontsize=11)
    ax2.set_ylabel('Weight Change (lbs)', fontsize=11)
    ax2.set_title('Weight Change vs Starting Weight by Study Duration', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average weight change by category
    category_stats = {}
    for cat in set(categories):
        cat_changes = [weight_changes[i] for i, c in enumerate(categories) if c == cat]
        if cat_changes:
            category_stats[cat] = {
                'mean': np.mean(cat_changes),
                'count': len(cat_changes)
            }
    
    if category_stats:
        bars = ax3.bar(category_stats.keys(), [stats['mean'] for stats in category_stats.values()], 
                       alpha=0.8, edgecolor='black', linewidth=1, color=['#8B4513', '#CD853F', '#DEB887'][:len(category_stats)])
        ax3.set_ylabel('Average Weight Change (lbs)', fontsize=11)
        ax3.set_title('Average Weight Change by Study Category', fontsize=12, fontweight='bold')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, (cat, stats) in zip(bars, category_stats.items(), strict=False):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height >= 0 else -0.2),
                    f'{height:.1f} lbs\n(n={stats["count"]})',
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Plot 4: Summary statistics
    ax4.axis('off')
    
    # Calculate summary stats
    total_participants = len(participants)
    avg_weight_change = np.mean(weight_changes)
    median_weight_change = np.median(weight_changes)
    std_weight_change = np.std(weight_changes)
    successful_participants = len([w for w in weight_changes if w < 0])
    success_rate = (successful_participants / total_participants) * 100 if total_participants > 0 else 0
    avg_starting_weight = np.mean(starting_weights)
    avg_duration = np.mean(num_measurements)
    
    # Count by original categories
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    summary_text = f"""POTATO DIET STUDY SUMMARY

Total Participants: {total_participants}
Average Study Duration: {avg_duration:.1f} days
Average Starting Weight: {avg_starting_weight:.1f} lbs

WEIGHT LOSS RESULTS:
• Average Change: {avg_weight_change:.1f} ± {std_weight_change:.1f} lbs
• Median Change: {median_weight_change:.1f} lbs
• Participants with Weight Loss: {successful_participants}
• Success Rate: {success_rate:.1f}%

CATEGORY BREAKDOWN:"""
    
    for cat, count in sorted(category_counts.items()):
        summary_text += f"\n• {cat}: {count} participants"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/Users/artemisfolle/Documents/Coding_Projects/potato-weight-nutrition/potato_diet_teaser.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Corrected visualization saved to: {output_path}")
    return output_path

def print_data_validation(participants):
    """Print validation statistics"""
    print("\nDATA VALIDATION:")
    print(f"Weight changes range: {min(p['weight_change'] for p in participants):.1f} to {max(p['weight_change'] for p in participants):.1f} lbs")
    print(f"Starting weights range: {min(p['starting_weight'] for p in participants):.1f} to {max(p['starting_weight'] for p in participants):.1f} lbs")
    print(f"Study durations range: {min(p['num_measurements'] for p in participants)} to {max(p['num_measurements'] for p in participants)} days")
    
    # Show some examples
    print("\nSample participants:")
    for i, p in enumerate(participants[:5]):
        print(f"  {p['participant_id']}: {p['starting_weight']:.1f} → {p['ending_weight']:.1f} lbs ({p['weight_change']:+.1f} lbs over {p['num_measurements']} days)")

if __name__ == "__main__":
    print("Analyzing complete potato diet dataset...")
    participants = analyze_all_participants()
    
    if participants:
        print_data_validation(participants)
        output_file = create_corrected_visualization(participants)
        print(f"\nCorrected visualization complete: {output_file}")
    else:
        print("No valid participant data found!")