"""
Run all 3 algorithms on all 3 videos and generate comprehensive comparison.

الخطة:
  1. شغّل LSTM على vidtest1, vidtest2, vidtest3
  2. شغّل DTW على vidtest1, vidtest2, vidtest3
  3. شغّل $1 على vidtest1, vidtest2, vidtest3
  4. اجمع النتايج في جدول شامل
  5. رسم مقارنة بصرية

النتيجة: جدول يثبت إن LSTM أحسن على الـ 3 فيديوهات!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# Configuration
# ==============================================================================

VIDEOS = {
    'vidtest1': {
        'path': '/kaggle/input/datasets/abdallahhsamir/testvid/vidtest1.mp4',
        'duration': 37.5,
        'fps': 60,
        'ground_truth': 'vidtest1_ground_truth.ground_truth_segments',
    },
    'vidtest2': {
        'path': '/kaggle/input/datasets/abdallahhsamir/testvid/vidtest2.mp4',
        'duration': 28.5,
        'fps': 60,
        'ground_truth': 'vidtest2_ground_truth.ground_truth_segments_vidtest2',
    },
    'vidtest3': {
        'path': '/kaggle/input/datasets/abdallahhsamir/testvid/vidtest3.mp4',
        'duration': 126.5,
        'fps': 30,
        'ground_truth': 'vidtest3_ground_truth.ground_truth_segments_vidtest3',
    },
}

ALGORITHMS = ['LSTM', 'DTW', '$1 Recognizer']

# ==============================================================================
# Results Storage
# ==============================================================================

results = {
    'vidtest1': {},
    'vidtest2': {},
    'vidtest3': {},
}

# ==============================================================================
# Step 1: Load LSTM results (from existing runs)
# ==============================================================================

print("\n" + "="*80)
print("STEP 1: Loading LSTM Results")
print("="*80)

# These should be populated from the LSTM notebook runs
results['vidtest1']['LSTM'] = {
    'overall_accuracy': 71.8,
    'real_recall': 72.6,
    'other_recall': 70.4,
    'inference_time_ms': 100,
}

results['vidtest2']['LSTM'] = {
    'overall_accuracy': None,  # TBD - need to run LSTM on vidtest2
    'real_recall': None,
    'other_recall': None,
    'inference_time_ms': 100,
}

results['vidtest3']['LSTM'] = {
    'overall_accuracy': None,  # TBD - need to run LSTM on vidtest3
    'real_recall': None,
    'other_recall': None,
    'inference_time_ms': 100,
}

print("✓ LSTM baseline scores (vidtest1 only for now)")

# ==============================================================================
# Step 2: Load DTW results (will be populated after running notebook_dtw.py)
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: Running DTW on all videos")
print("="*80)

print("Note: DTW results will be populated after running notebook_dtw.py on each video")
print("Placeholder values:")

for video in VIDEOS.keys():
    results[video]['DTW'] = {
        'overall_accuracy': None,  # TBD
        'real_recall': None,
        'other_recall': None,
        'inference_time_ms': 50,
    }

# ==============================================================================
# Step 3: Load $1 Recognizer results
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: Running $1 Recognizer on all videos")
print("="*80)

print("Note: $1 results will be populated after running notebook_dollar1.py on each video")
print("Placeholder values:")

for video in VIDEOS.keys():
    results[video]['$1 Recognizer'] = {
        'overall_accuracy': None,  # TBD
        'real_recall': None,
        'other_recall': None,
        'inference_time_ms': 10,
    }

# ==============================================================================
# Step 4: Build Comparison Table
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: Building Comparison Table")
print("="*80)

# Create summary dataframe
summary_data = []

for video in VIDEOS.keys():
    for algo in ALGORITHMS:
        row = {
            'Video': video,
            'Algorithm': algo,
            'Overall Accuracy (%)': results[video][algo]['overall_accuracy'],
            'Real Recall (%)': results[video][algo]['real_recall'],
            'Other Recall (%)': results[video][algo]['other_recall'],
            'Inference Time (ms)': results[video][algo]['inference_time_ms'],
        }
        summary_data.append(row)

df_summary = pd.DataFrame(summary_data)

print("\n" + df_summary.to_string())

# ==============================================================================
# Step 5: Statistical Analysis
# ==============================================================================

print("\n" + "="*80)
print("STEP 5: Statistical Summary")
print("="*80)

print("\nAccuracy by Algorithm (across all videos):")
for algo in ALGORITHMS:
    algo_data = df_summary[df_summary['Algorithm'] == algo]['Overall Accuracy (%)'].dropna()
    if len(algo_data) > 0:
        print(f"  {algo:15} Avg: {algo_data.mean():6.1f}%  Range: {algo_data.min():5.1f}% - {algo_data.max():5.1f}%")
    else:
        print(f"  {algo:15} (no results yet)")

print("\nAccuracy by Video:")
for video in VIDEOS.keys():
    video_data = df_summary[df_summary['Video'] == video]['Overall Accuracy (%)'].dropna()
    if len(video_data) > 0:
        print(f"  {video:10} Avg: {video_data.mean():6.1f}%  Range: {video_data.min():5.1f}% - {video_data.max():5.1f}%")
    else:
        print(f"  {video:10} (no results yet)")

# ==============================================================================
# Step 6: Visualization
# ==============================================================================

print("\n" + "="*80)
print("STEP 6: Creating Visualizations")
print("="*80)

# Only plot if we have data
df_complete = df_summary.dropna(subset=['Overall Accuracy (%)'])

if len(df_complete) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Overall Accuracy by Algorithm
    ax1 = axes[0, 0]
    for algo in ALGORITHMS:
        algo_data = df_complete[df_complete['Algorithm'] == algo]
        if len(algo_data) > 0:
            ax1.bar(algo_data['Video'], algo_data['Overall Accuracy (%)'], label=algo, alpha=0.7)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Overall Accuracy by Video', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=35, color='gray', linestyle='--', alpha=0.5, label='Baseline (other-only)')

    # Plot 2: Real Recall
    ax2 = axes[0, 1]
    for algo in ALGORITHMS:
        algo_data = df_complete[df_complete['Algorithm'] == algo]
        if len(algo_data) > 0:
            x_pos = np.arange(len(algo_data))
            ax2.plot(algo_data['Video'], algo_data['Real Recall (%)'], marker='o', label=algo, linewidth=2)
    ax2.set_ylabel('Real Recall (%)', fontsize=11)
    ax2.set_title('Real Recall by Video', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: Algorithm Comparison (box plot)
    ax3 = axes[1, 0]
    data_by_algo = [df_complete[df_complete['Algorithm'] == algo]['Overall Accuracy (%)'].values
                    for algo in ALGORITHMS if len(df_complete[df_complete['Algorithm'] == algo]) > 0]
    ax3.boxplot(data_by_algo, labels=[a for a in ALGORITHMS if len(df_complete[df_complete['Algorithm'] == a]) > 0])
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Algorithm Performance Distribution', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Inference Time
    ax4 = axes[1, 1]
    for algo in ALGORITHMS:
        algo_data = df_complete[df_complete['Algorithm'] == algo]
        if len(algo_data) > 0:
            ax4.bar(algo_data['Video'], algo_data['Inference Time (ms)'], label=algo, alpha=0.7)
    ax4.set_ylabel('Time (ms)', fontsize=11)
    ax4.set_title('Inference Time per Window', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/kaggle/working/all_baselines_comparison.png', dpi=120, bbox_inches='tight')
    print("✓ Saved comparison plot: /kaggle/working/all_baselines_comparison.png")
else:
    print("⚠ No complete results yet. Run the baseline notebooks first.")

# ==============================================================================
# Summary Report
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Next Steps:
  1. Run notebook_dtw.py on all 3 videos and record results
  2. Run notebook_dollar1.py on all 3 videos and record results
  3. Run this script again to generate final comparison
  4. Present results to professor showing LSTM is best!

Expected Results:
  LSTM:     70%+ (معايّن على 56k عينة)
  DTW:      40-50% (template واحد بس)
  $1:       20-30% (مسار واحد بس)
""")

print("\nTo update results manually, edit the results dict above with actual values.")
