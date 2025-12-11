#!/usr/bin/env python3
"""
Merge two descriptor AP files into a scatter-ready format for pgfplots.

Usage:
    python make_scatter_dat.py <descriptor1> <descriptor2>

Example:
    python make_scatter_dat.py SIFT ASV_SIFT_1M_15
    
Output:
    Creates combined_dat/<descriptor1>vs<descriptor2>.dat
"""

import sys
import csv
import os

def load_ap(path):
    """Load AP data from a descriptor file."""
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=" ")
        for row in reader:
            data[row["pairId"]] = float(row["AP"])
    return data

def main():
    if len(sys.argv) != 3:
        print("Usage: python make_scatter_dat.py <descriptor1> <descriptor2>")
        print("Example: python make_scatter_dat.py SIFT ASV_SIFT_1M_15")
        sys.exit(1)
    
    desc1_name = sys.argv[1]
    desc2_name = sys.argv[2]
    
    # Load data files
    desc1_file = f"ap_pairs/{desc1_name}.dat"
    desc2_file = f"ap_pairs/{desc2_name}.dat"
    
    try:
        desc1_data = load_ap(desc1_file)
        desc2_data = load_ap(desc2_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure {desc1_file} and {desc2_file} exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs("combined_dat", exist_ok=True)
    
    # Create output filename
    output_file = f"combined_dat/{desc1_name}vs{desc2_name}.dat"
    
    # Write merged data
    with open(output_file, "w") as f:
        f.write(f"pairId AP_{desc1_name} AP_{desc2_name}\n")
        for pid in sorted(desc1_data.keys()):
            if pid in desc2_data:
                f.write(f"{pid} {desc1_data[pid]:.6f} {desc2_data[pid]:.6f}\n")
    
    print(f"Created {output_file}")
    print(f"  Pairs matched: {len([p for p in desc1_data if p in desc2_data])}")
    print(f"  {desc1_name} total pairs: {len(desc1_data)}")
    print(f"  {desc2_name} total pairs: {len(desc2_data)}")

if __name__ == "__main__":
    main()
