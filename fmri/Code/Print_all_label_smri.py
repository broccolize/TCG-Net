import os
import csv

# Configuration from sMRI_proc.py
SMRI_PATH = "/home/celery/uk_BIOBANK_DATA/imaging/sMRI/UKB_Freesurfer_APARC.csv"
TIMEPOINT = "_2_0"

def summarize_smri(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        return

    print(f"Analyzing file: {filepath}")
    
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        
        # Count rows (subjects) without loading full data if possible
        # But DictReader is already open, so we can just sum up the rows
        row_count = sum(1 for _ in reader)

    # Identify ROI columns using the logic from sMRI_proc.py
    # 1. Starts with SA_
    # 2. Ends with _2_0
    # 3. Does not contain "Total"
    sa_cols = [col for col in header if col.startswith("SA_") and col.endswith(TIMEPOINT) and "Total" not in col]
    roi_roots = [col.replace("SA_", "").replace(TIMEPOINT, "") for col in sa_cols]
    roi_with_index = [
        (col.replace("SA_", "").replace(TIMEPOINT, ""), header.index(col))
        for col in sa_cols
    ]
    
    num_rois = len(roi_roots)
    
    print("\n" + "="*60)
    print(" sMRI DATASET SUMMARY (Pure Python)")
    print("="*60)
    print(f"Total Subjects (Rows):    {row_count}")
    print(f"Total ROIs Detected:      {num_rois}")
    print(f"Features per ROI:         3 (Surface Area, Thickness, Volume)")
    print(f"Final Dimension:          ({row_count}, {num_rois}, 3)")
    print("-" * 60)
    print("List of detected ROIs with CSV column index (from SA_*_2_0):")
    for roi_name, col_idx in roi_with_index:
        print(f"[{col_idx:>3}] {roi_name}")
    
    print("="*60)

if __name__ == "__main__":
    summarize_smri(SMRI_PATH)
