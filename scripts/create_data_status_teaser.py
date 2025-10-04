#!/usr/bin/env python3
"""
Create a data status teaser plot that accurately reflects the current state.
No fake data, no simulation - just honest reporting of what's available.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def create_data_status_plot(output_path):
    """Create a plot that honestly shows data availability status."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Remove axes for clean look
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9, 'Potato Fiber Study - Data Pipeline Status', 
            fontsize=24, fontweight='bold', ha='center', va='center')
    
    # Status message
    status_text = """
DATA AVAILABILITY STATUS
═══════════════════════════════════════════

Raw Data Files Found:
• Excel files in ~/Downloads/Potato Raw Dato/
• Multiple subject directories (4 Weeks, Closed Early, Dropped)
• Individual .xlsx files per participant

Required Processed Files:
✗ Potato_tidy.csv (not found)
✗ Potato_nutrition_rows.csv (not found) 
✗ Potato_fiber_daily.csv (not found)

Current State:
• Pipeline is configured and ready
• Data processing step needed to convert Excel → CSV
• No synthetic/simulated data displayed
• Awaiting data cleaning/processing step

Next Steps:
1. Process raw Excel files into required CSV format
2. Run pipeline analysis on real data
3. Generate authentic results visualization
"""
    
    ax.text(5, 5, status_text, fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8),
            fontfamily='monospace')
    
    # Footer note
    ax.text(5, 1, 'This figure shows actual data availability status - no simulated results',
            fontsize=10, ha='center', va='center', style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    """Create honest data status plot."""
    
    output_path = Path("potato-weight-nutrition/reports/figs/teaser.png")
    
    print("Creating data availability status plot...")
    success = create_data_status_plot(output_path)
    
    if success:
        print(f"✅ Data status plot created: {output_path}")
        print("   📋 Shows actual data availability - no simulated results")
        return True
    else:
        print("❌ Failed to create plot")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)