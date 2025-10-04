#!/usr/bin/env python3
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_weights_from_csv(file_path):
    """Extract weight data from a potato diet CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Find the weight row by looking for "Weight in the Morning"
        weight_row_idx = None
        for idx, row in df.iterrows():
            if any('Weight in the Morning' in str(cell) for cell in row.values):
                weight_row_idx = idx
                break
        
        if weight_row_idx is None:
            return None, None
        
        # Extract weights from the row
        weight_row = df.iloc[weight_row_idx]
        weights = []
        days = []
        
        # Start from column 4 and extract numeric values that look like weights
        for i, value in enumerate(weight_row.iloc[4:]):
            if pd.notna(value):
                try:
                    # Try to convert to float and check if it's a reasonable weight
                    weight_val = float(value)
                    if 50 <= weight_val <= 500:  # Reasonable weight range
                        weights.append(weight_val)
                        days.append(len(weights))
                except (ValueError, TypeError):
                    continue
        
        # Extract participant info
        participant_id = None
        category = None
        
        for col in df.columns:
            if 'participant_id' in col:
                participant_id = df[col].iloc[0] if not df[col].empty else None
            elif 'category' in col:
                category = df[col].iloc[0] if not df[col].empty else None
        
        if participant_id is None:
            # Try to extract from filename
            filename = os.path.basename(file_path)
            if filename.startswith('combined'):
                return None, None
            participant_id = filename.split('_')[0]
        
        return {
            'participant_id': participant_id,
            'category': category,
            'weights': weights,
            'days': days,
            'weight_change': weights[-1] - weights[0] if len(weights) >= 2 else 0,
            'duration': len(weights)
        }, weights
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def analyze_all_data():
    """Analyze all CSV files and extract weight data"""
    csv_dir = "/Users/artemisfolle/Documents/Coding_Projects/potato_csv_data"
    csv_files = glob.glob(os.path.join(csv_dir, "*_Sheet1.csv"))  # Only process Sheet1 files
    
    participants = []
    
    for file_path in csv_files:
        if 'combined' in file_path:
            continue
            
        participant_data, weights = extract_weights_from_csv(file_path)
        if participant_data and weights:
            participants.append(participant_data)
    
    return participants

def create_improved_visualization(participants):
    """Create an improved visualization"""
    
    if not participants:
        print("No valid participant data found!")
        return None
    
    # Extract data for plotting
    weight_changes = [p['weight_change'] for p in participants]
    starting_weights = [p['weights'][0] for p in participants if p['weights']]
    durations = [p['duration'] for p in participants]
    participant_ids = [p['participant_id'] for p in participants]
    
    # Categorize by duration
    categories = []
    for p in participants:
        if p['duration'] >= 25:
            categories.append('4+ Weeks')
        elif p['duration'] >= 14:
            categories.append('2-4 Weeks')
        elif p['duration'] >= 7:
            categories.append('1-2 Weeks')
        else:
            categories.append('< 1 Week')
    
    # Create the visualization
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Potato Diet Study Results - Weight & Nutrition Analysis', fontsize=16, fontweight='bold')
    
    # Color palette
    colors = ['#8B4513', '#CD853F', '#DEB887', '#F4A460']
    
    # Plot 1: Weight change distribution
    ax1.hist(weight_changes, bins=min(10, len(weight_changes)), alpha=0.7, 
             color='#8B4513', edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Weight Change (lbs)', fontsize=11)
    ax1.set_ylabel('Number of Participants', fontsize=11)
    ax1.set_title('Distribution of Weight Changes', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight change vs starting weight
    unique_categories = list(set(categories))
    category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}
    
    for cat in unique_categories:
        cat_indices = [i for i, c in enumerate(categories) if c == cat]
        cat_starting = [starting_weights[i] for i in cat_indices]
        cat_changes = [weight_changes[i] for i in cat_indices]
        ax2.scatter(cat_starting, cat_changes, alpha=0.8, 
                   label=cat, color=category_colors[cat], s=80, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('Starting Weight (lbs)', fontsize=11)
    ax2.set_ylabel('Weight Change (lbs)', fontsize=11)
    ax2.set_title('Weight Change vs Starting Weight by Duration', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average weight change by duration category
    cat_avg_changes = {}
    cat_counts = {}
    for cat in unique_categories:
        cat_changes = [weight_changes[i] for i, c in enumerate(categories) if c == cat]
        cat_avg_changes[cat] = np.mean(cat_changes) if cat_changes else 0
        cat_counts[cat] = len(cat_changes)
    
    bars = ax3.bar(cat_avg_changes.keys(), cat_avg_changes.values(), 
                   color=[category_colors[cat] for cat in cat_avg_changes.keys()],
                   alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Average Weight Change (lbs)', fontsize=11)
    ax3.set_title('Average Weight Change by Study Duration', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cat in zip(bars, cat_avg_changes.keys(), strict=False):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -0.5),
                f'{height:.1f} lbs\n(n={cat_counts[cat]})',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Plot 4: Individual weight trajectories (sample)
    ax4.set_title('Sample Weight Trajectories', fontsize=12, fontweight='bold')
    
    # Show up to 6 participants with different colors
    sample_participants = participants[:6]
    trajectory_colors = plt.cm.Set3(np.linspace(0, 1, len(sample_participants)))
    
    for i, p in enumerate(sample_participants):
        if p['weights'] and len(p['weights']) > 1:
            ax4.plot(p['days'], p['weights'], 'o-', 
                    color=trajectory_colors[i], alpha=0.8, linewidth=2, markersize=4,
                    label=f"ID: {str(p['participant_id'])[:6]}...")
    
    ax4.set_xlabel('Day', fontsize=11)
    ax4.set_ylabel('Weight (lbs)', fontsize=11)
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add summary statistics text box
    total_participants = len(participants)
    avg_weight_loss = np.mean(weight_changes)
    median_weight_loss = np.median(weight_changes)
    successful_participants = len([w for w in weight_changes if w < 0])
    success_rate = (successful_participants / total_participants) * 100 if total_participants > 0 else 0
    avg_duration = np.mean(durations)
    
    summary_text = f"""STUDY SUMMARY
    
Total Participants: {total_participants}
Average Duration: {avg_duration:.1f} days

Weight Loss Results:
• Average Change: {avg_weight_loss:.1f} lbs
• Median Change: {median_weight_loss:.1f} lbs
• Participants with Weight Loss: {successful_participants}
• Success Rate: {success_rate:.1f}%

Duration Categories:"""
    
    for cat in sorted(unique_categories):
        count = categories.count(cat)
        summary_text += f"\n• {cat}: {count} participants"
    
    # Add text box to the figure
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
             verticalalignment='bottom')
    
    # Save the plot
    output_path = "/Users/artemisfolle/Documents/Coding_Projects/potato-weight-nutrition/potato_diet_teaser.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Improved teaser visualization saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    print("Analyzing potato diet data...")
    participants = analyze_all_data()
    
    print(f"Found valid data for {len(participants)} participants")
    
    if participants:
        # Print some sample data for verification
        print("\nSample participant data:")
        for i, p in enumerate(participants[:3]):
            print(f"  Participant {p['participant_id']}: {len(p['weights'])} weight measurements")
            print(f"    Starting weight: {p['weights'][0]:.1f} lbs")
            print(f"    Weight change: {p['weight_change']:.1f} lbs")
        
        # Create visualization
        output_file = create_improved_visualization(participants)
        print(f"\nVisualization complete: {output_file}")
    else:
        print("No valid participant data found!")