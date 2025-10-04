#!/usr/bin/env python3
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_potato_data(file_path):
    """Parse individual potato data files and extract key metrics"""
    try:
        # Read the file
        df = pd.read_csv(file_path)
        
        # Extract participant ID from filename or data
        filename = os.path.basename(file_path)
        if filename.startswith('combined'):
            return None
        
        participant_id = filename.split('_')[0]
        sheet_type = filename.split('_')[1].replace('.csv', '')
        
        if sheet_type == 'Summary':
            # Parse summary sheet
            return parse_summary_sheet(df, participant_id)
        elif sheet_type == 'Sheet1':
            # Parse daily tracking sheet
            return parse_daily_sheet(df, participant_id)
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def parse_summary_sheet(df, participant_id):
    """Extract summary information from Summary sheet"""
    data = {'participant_id': participant_id, 'sheet_type': 'summary'}
    
    # Look for key fields in the summary
    for idx, row in df.iterrows():
        if pd.notna(row.iloc[1]):  # Check if second column has data
            field_name = str(row.iloc[1]).strip()
            value = row.iloc[2] if len(row) > 2 else None
            
            if 'Subject ID' in field_name:
                data['subject_id'] = value
            elif 'Starting Weight' in field_name:
                data['starting_weight'] = value
            elif 'ending' in field_name.lower() or 'final' in field_name.lower():
                data['ending_weight'] = value
    
    return data

def parse_daily_sheet(df, participant_id):
    """Extract daily tracking data from Sheet1"""
    data = {'participant_id': participant_id, 'sheet_type': 'daily'}
    
    # Find key rows
    weight_row = None
    date_row = None
    day_row = None
    broke_diet_row = None
    energy_row = None
    mood_row = None
    
    for idx, row in df.iterrows():
        if pd.notna(row.iloc[2]):  # Check if third column has data
            field_name = str(row.iloc[2]).strip()
            
            if 'Weight in the Morning' in field_name:
                weight_row = idx
            elif 'Date' in field_name:
                date_row = idx
            elif 'Day' in field_name:
                day_row = idx
            elif 'Broke diet' in field_name:
                broke_diet_row = idx
            elif 'Energy' in field_name:
                energy_row = idx
            elif 'Mood' in field_name:
                mood_row = idx
    
    # Extract weight data
    weights = []
    dates = []
    days = []
    
    if weight_row is not None:
        weight_data = df.iloc[weight_row, 4:].values  # Start from column 4
        for i, weight in enumerate(weight_data):
            if pd.notna(weight) and str(weight).replace('.', '').isdigit():
                weights.append(float(weight))
                days.append(i + 1)
    
    if date_row is not None:
        date_data = df.iloc[date_row, 4:len(weights)+4].values
        for date in date_data:
            if pd.notna(date):
                dates.append(str(date))
    
    data['weights'] = weights
    data['days'] = days[:len(weights)]
    data['dates'] = dates[:len(weights)]
    
    # Calculate weight change
    if len(weights) >= 2:
        data['weight_change'] = weights[-1] - weights[0]
        data['weight_change_percent'] = (data['weight_change'] / weights[0]) * 100
    
    return data

def analyze_all_data():
    """Analyze all potato data files"""
    csv_dir = "/Users/artemisfolle/Documents/Coding_Projects/potato_csv_data"
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    all_participants = {}
    
    for file_path in csv_files:
        if 'combined' in file_path:
            continue
            
        result = parse_potato_data(file_path)
        if result:
            pid = result['participant_id']
            if pid not in all_participants:
                all_participants[pid] = {}
            
            sheet_type = result['sheet_type']
            all_participants[pid][sheet_type] = result
    
    return all_participants

def create_weight_loss_visualization(participants_data):
    """Create a visualization showing weight loss across participants"""
    
    # Extract weight change data
    weight_changes = []
    starting_weights = []
    categories = []
    participant_ids = []
    
    for pid, data in participants_data.items():
        if 'daily' in data and 'weights' in data['daily']:
            daily_data = data['daily']
            weights = daily_data['weights']
            
            if len(weights) >= 2:
                weight_change = weights[-1] - weights[0]
                weight_changes.append(weight_change)
                starting_weights.append(weights[0])
                participant_ids.append(pid)
                
                # Determine category based on file structure
                # This is a simplified categorization
                if len(weights) >= 25:  # Likely 4 weeks
                    categories.append('4 Weeks')
                elif len(weights) >= 10:
                    categories.append('Closed Early')
                else:
                    categories.append('Dropped')
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Potato Diet Study Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Weight change distribution
    ax1.hist(weight_changes, bins=20, alpha=0.7, color='#8B4513', edgecolor='black')
    ax1.set_xlabel('Weight Change (lbs)')
    ax1.set_ylabel('Number of Participants')
    ax1.set_title('Distribution of Weight Changes')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 2: Weight change by starting weight
    colors = ['#8B4513', '#DEB887', '#D2691E']
    category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(set(categories))}
    
    for cat in set(categories):
        cat_indices = [i for i, c in enumerate(categories) if c == cat]
        cat_starting = [starting_weights[i] for i in cat_indices]
        cat_changes = [weight_changes[i] for i in cat_indices]
        ax2.scatter(cat_starting, cat_changes, alpha=0.7, 
                   label=cat, color=category_colors[cat], s=60)
    
    ax2.set_xlabel('Starting Weight (lbs)')
    ax2.set_ylabel('Weight Change (lbs)')
    ax2.set_title('Weight Change vs Starting Weight')
    ax2.legend()
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 3: Average weight change by category
    cat_avg_changes = {}
    for cat in set(categories):
        cat_changes = [weight_changes[i] for i, c in enumerate(categories) if c == cat]
        cat_avg_changes[cat] = np.mean(cat_changes) if cat_changes else 0
    
    bars = ax3.bar(cat_avg_changes.keys(), cat_avg_changes.values(), 
                   color=[category_colors[cat] for cat in cat_avg_changes.keys()],
                   alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Average Weight Change (lbs)')
    ax3.set_title('Average Weight Change by Category')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # Plot 4: Summary statistics
    ax4.axis('off')
    
    # Calculate summary stats
    total_participants = len(weight_changes)
    avg_weight_loss = np.mean(weight_changes)
    median_weight_loss = np.median(weight_changes)
    successful_participants = len([w for w in weight_changes if w < 0])
    success_rate = (successful_participants / total_participants) * 100
    
    summary_text = f"""
    Study Summary:
    
    Total Participants: {total_participants}
    
    Average Weight Change: {avg_weight_loss:.1f} lbs
    Median Weight Change: {median_weight_loss:.1f} lbs
    
    Participants with Weight Loss: {successful_participants}
    Success Rate: {success_rate:.1f}%
    
    Category Breakdown:
    """
    
    for cat, count in {cat: categories.count(cat) for cat in set(categories)}.items():
        summary_text += f"\n    {cat}: {count} participants"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/Users/artemisfolle/Documents/Coding_Projects/potato_diet_teaser.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Teaser visualization saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    print("Analyzing potato diet data...")
    participants_data = analyze_all_data()
    
    print(f"Found data for {len(participants_data)} participants")
    
    # Create visualization
    output_file = create_weight_loss_visualization(participants_data)
    print(f"Visualization complete: {output_file}")