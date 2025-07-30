"""
Enhanced GRMP Attack Visualization for Top-tier Journal
Three core figures with professional styling and dynamic y-axis limits
Updated to support 1-14 rounds with better x-axis spacing

SIZE ADJUSTMENT GUIDE:
---------------------
To adjust figure and font sizes, modify these parameters:

1. FIGURE SIZE: Change figsize in plot functions
   - Figure 1: fig, ax1 = plt.subplots(figsize=(10, 6))  # (width, height) in inches
   - Figure 2: fig, ax = plt.subplots(figsize=(12, 7))
   - Recommended for larger: (12, 8) or (14, 8)

2. FONT SIZES: Modify plt.rcParams at the beginning
   - Base font: plt.rcParams['font.size'] = 14  # Try 16 or 18
   - Title: plt.rcParams['axes.titlesize'] = 18  # Try 20 or 22
   - Axis labels: plt.rcParams['axes.labelsize'] = 16  # Try 18 or 20
   - Tick labels: plt.rcParams['xtick/ytick.labelsize'] = 14  # Try 16
   - Legend: plt.rcParams['legend.fontsize'] = 13  # Try 15

3. LINE WIDTH & MARKER SIZE:
   - linewidth=3  # Try 4 or 5 for thicker lines
   - markersize=10  # Try 12 or 14 for larger markers

4. DPI for high resolution:
   - fig.savefig(..., dpi=300)  # Try 600 for ultra-high resolution
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Font size configuration variables
FONT_SIZE_BASE = 20
FONT_SIZE_TITLE = 24
FONT_SIZE_LABEL = 20
FONT_SIZE_TICK = 20
FONT_SIZE_LEGEND = 20

# Specific font sizes used in functions
FONT_SIZE_XLABEL = 20  # Used for x-axis labels in functions
FONT_SIZE_YLABEL = 20  # Used for y-axis labels in functions
FONT_SIZE_PLOT_TITLE = 24  # Used for plot titles in functions
FONT_SIZE_TICK_PARAMS = 20  # Used for tick parameters in functions
FONT_SIZE_LEGEND_SMALL = 20  # Used for smaller legends

# Set professional font settings - 简化版
plt.rcParams['font.family'] = 'STIXGeneral'  # STIX字体与Times New Roman非常相似
plt.rcParams['font.size'] = FONT_SIZE_BASE
plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
plt.rcParams['mathtext.fontset'] = 'stix'

# 确保PDF输出中字体可编辑
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Remove top and right spines for cleaner look
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False






def plot_attack_performance_enhanced(json_file_path, output_dir=None):
    """
    Figure 1: Learning Accuracy (left) and ASR (right) with professional styling
    """
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    metrics = data['progressive_metrics']

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure - slightly wider for 14 rounds
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Truncate to first 14 rounds if data has more
    rounds = metrics['rounds'][:15]
    asr = metrics['attack_asr'][:15]
    fl_acc = metrics['clean_acc'][:15]

    # Create second y-axis
    ax2 = ax1.twinx()

    # Remove top spine for ax2 as well
    ax2.spines['top'].set_visible(False)

    # Add background shading with soft colors - adjusted for 14 rounds
    # Trust building phase (light green)
    ax1.axvspan(0.5, 5.5, alpha=0.15, color='#90EE90', zorder=0)
    # Attack escalation phase (light red) - extended to cover more rounds
    ax1.axvspan(5.5, 15.5, alpha=0.15, color='#FFB6C1', zorder=0)

    # Plot Learning Accuracy (left axis)
    line1 = ax1.plot(rounds, fl_acc, 's-', color='#4169E1', linewidth=3,
                    markersize=10, markerfacecolor='white', markeredgewidth=2.5,
                    markeredgecolor='#4169E1', label='Learning Accuracy')

    ax1.set_xlabel('Communication Round', fontsize=FONT_SIZE_XLABEL, fontweight='bold')
    ax1.set_ylabel('Learning Accuracy', fontsize=FONT_SIZE_YLABEL, color='#4169E1', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4169E1', labelsize=FONT_SIZE_TICK_PARAMS)

    # Plot ASR (right axis)
    line2 = ax2.plot(rounds, asr, 'o-', color='#DC143C', linewidth=3,
                    markersize=10, markerfacecolor='white', markeredgewidth=2.5,
                    markeredgecolor='#DC143C', label='Attack Success Rate (ASR)')

    ax2.set_ylabel('Attack Success Rate', fontsize=FONT_SIZE_YLABEL, color='#DC143C', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#DC143C', labelsize=FONT_SIZE_TICK_PARAMS)

    # Highlight peak ASR
    max_asr_idx = asr.index(max(asr))
    ax2.scatter(rounds[max_asr_idx], asr[max_asr_idx], s=200,
               color='#DC143C', zorder=5, edgecolors='black', linewidth=2)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True,
              fancybox=True, shadow=True, framealpha=0.9)

    # Title
    ax1.set_title('Impact of GRMP Attack on Federated Learning Performance',
                 fontsize=FONT_SIZE_PLOT_TITLE, fontweight='bold', pad=20)

    # Grid
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Set x-axis limits with extra space on the left
    ax1.set_xlim(0, max(rounds) + 0.5)  # Changed from 0.5 to 0 for left margin
    
    # Dynamic y-axis limits for Learning Accuracy
    acc_min = min(fl_acc)
    acc_max = max(fl_acc)
    acc_range = acc_max - acc_min
    ax1.set_ylim(acc_min - 0.05 * acc_range, acc_max + 0.4 * acc_range)  # 40% extra space on top
    
    # Dynamic y-axis limits for ASR
    asr_min = min(asr)
    asr_max = max(asr)
    asr_range = asr_max - asr_min
    ax2.set_ylim(asr_min - 0.05 * asr_range, asr_max + 0.4 * asr_range)  # 40% extra space on top

    # Ensure integer x-ticks
    ax1.set_xticks(rounds)

    # Save figure
    plt.tight_layout()

    output_path_png = output_dir / 'figure1_attack_performance.png'
    output_path_pdf = output_dir / 'figure1_attack_performance.pdf'

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

    print(f"Figure 1 saved to: {output_path_png}")
    plt.close()





# def plot_similarity_evolution_bars_style(json_file_path, output_dir=None):
#     """
#     Figure 2: Similarity evolution with bars for benign users and lines for attackers
#     Following the reference style from the advisor
#     """
#     # Load data
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)

#     config = data['config']
#     results = data['results'][:13]  # Truncate to first 14 rounds

#     if output_dir is None:
#         output_dir = Path('./results/figures')
#     output_dir.mkdir(exist_ok=True, parents=True)

#     # Create figure - slightly wider for 14 rounds
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Remove top and right spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # Identify attackers
#     num_clients = config['num_clients']
#     num_attackers = config['num_attackers']
#     attacker_ids = list(range(num_clients - num_attackers, num_clients))

#     # Collect data
#     rounds = []
#     thresholds = []
#     attacker_sims_by_round = []
#     benign_sims_by_round = []

#     for round_data in results:
#         rounds.append(round_data['round'])
#         thresholds.append(round_data['defense']['threshold'])

#         sims = round_data['defense']['similarities']

#         # Separate benign and attacker similarities
#         benign_sims = []
#         attacker_sims = []

#         for i, sim in enumerate(sims):
#             if i in attacker_ids:
#                 attacker_sims.append(sim)
#             else:
#                 benign_sims.append(sim)

#         benign_sims_by_round.append(benign_sims)
#         attacker_sims_by_round.append(attacker_sims)

#     # Plot defense threshold as bars
#     bar_width = 0.8
#     threshold_bars = ax.bar(rounds, thresholds, width=bar_width,
#                            color='#228B22', alpha=0.3, edgecolor='#228B22',
#                            linewidth=1.5, label='Defense threshold', zorder=1)

#     # Plot benign users as average line
#     benign_avg_sims = []
#     for benign_sims in benign_sims_by_round:
#         if benign_sims:
#             benign_avg_sims.append(np.mean(benign_sims))
#         else:
#             benign_avg_sims.append(0)

#     # Plot benign users average line
#     ax.plot(rounds, benign_avg_sims, 'o-', color='#4682B4',
#            linewidth=3, markersize=10, markerfacecolor='white',
#            markeredgewidth=2.5, label='Benign users (avg)',
#            zorder=4)

#     # Plot attackers as lines
#     # Reorganize attacker data by attacker ID
#     attacker_trajectories = {aid: [] for aid in range(num_attackers)}

#     for round_sims in attacker_sims_by_round:
#         for aid in range(num_attackers):
#             if aid < len(round_sims):
#                 attacker_trajectories[aid].append(round_sims[aid])
#             else:
#                 attacker_trajectories[aid].append(None)

#     # Plot each attacker's trajectory
#     attacker_colors = ['#DC143C', '#FF6347']
#     for aid, trajectory in attacker_trajectories.items():
#         valid_rounds = [r for r, s in zip(rounds, trajectory) if s is not None]
#         valid_sims = [s for s in trajectory if s is not None]

#         if valid_sims:
#             ax.plot(valid_rounds, valid_sims, 'o-',
#                    color=attacker_colors[aid % len(attacker_colors)],
#                    linewidth=3, markersize=10, markerfacecolor='white',
#                    markeredgewidth=2.5,
#                    label=f'Attacker {aid + 1}',
#                    zorder=5)

#     # Styling
#     ax.set_xlabel('Communication Round', fontsize=FONT_SIZE_XLABEL, fontweight='bold')
#     ax.set_ylabel('Cosine Similarity', fontsize=FONT_SIZE_YLABEL, fontweight='bold')
#     ax.set_title('Stealthiness of GRMP Attack: Similarity Evolution Analysis',
#                 fontsize=FONT_SIZE_PLOT_TITLE, fontweight='bold', pad=20)

#     # Legend
#     ax.legend(loc='best', frameon=True, fancybox=True,
#              shadow=True, framealpha=0.9)

#     # Grid
#     ax.grid(True, alpha=0.3, linestyle='--', axis='y')
#     ax.set_axisbelow(True)

#     # Set x-axis limits with extra space on the left
#     ax.set_xlim(0, max(rounds) + 0.5)  # Changed from 0.5 to 0 for left margin
    
#     # Dynamic y-axis limits
#     all_values = []
#     all_values.extend(thresholds)
#     all_values.extend(benign_avg_sims)
#     for trajectory in attacker_trajectories.values():
#         all_values.extend([v for v in trajectory if v is not None])
    
#     y_min = min(all_values)
#     y_max = max(all_values)
#     y_range = y_max - y_min
#     ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.3 * y_range)  # 30% extra space on top

#     # Set x-ticks
#     ax.set_xticks(rounds)

#     # Save figure
#     plt.tight_layout()

#     output_path_png = output_dir / 'figure2_similarity_evolution.png'
#     output_path_pdf = output_dir / 'figure2_similarity_evolution.pdf'

#     fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
#     fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

#     print(f"Figure 2 saved to: {output_path_png}")
#     plt.close()






def plot_similarity_individual_benign(json_file_path, output_dir=None):
    """
    Figure 3: Individual benign users' similarity evolution
    Same style as Figure 2 but showing each benign user separately
    """
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    config = data['config']
    results = data['results'][:15]  # Truncate to first 14 rounds

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure - slightly wider for 14 rounds
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Identify attackers
    num_clients = config['num_clients']
    num_attackers = config['num_attackers']
    num_benign = num_clients - num_attackers
    attacker_ids = list(range(num_clients - num_attackers, num_clients))

    # Collect data
    rounds = []
    thresholds = []
    attacker_sims_by_round = []
    benign_sims_by_round = []

    for round_data in results:
        rounds.append(round_data['round'])
        thresholds.append(round_data['defense']['threshold'])

        sims = round_data['defense']['similarities']

        # Separate benign and attacker similarities
        benign_sims = []
        attacker_sims = []

        for i, sim in enumerate(sims):
            if i in attacker_ids:
                attacker_sims.append(sim)
            else:
                benign_sims.append(sim)

        benign_sims_by_round.append(benign_sims)
        attacker_sims_by_round.append(attacker_sims)

    # Plot defense threshold as bars (normal height)
    bar_width = 0.8
    threshold_bars = ax.bar(rounds, thresholds, width=bar_width,
                           color='#228B22', alpha=0.3, edgecolor='#228B22',
                           linewidth=1.5, label='Defense threshold', zorder=1)

    # Plot each benign user individually
    # Support up to 8 benign users
    benign_colors = [
        '#4682B4',  # SteelBlue
        '#5F9EA0',  # CadetBlue  
        '#6495ED',  # CornflowerBlue
        '#4169E1',  # RoyalBlue
        '#1E90FF',  # DodgerBlue
        '#00BFFF',  # DeepSkyBlue
        '#87CEEB',  # SkyBlue
        '#B0C4DE'   # LightSteelBlue
    ]
    
    benign_markers = [
        'o',  # Circle
        's',  # Square
        '^',  # Triangle up
        'D',  # Diamond
        'v',  # Triangle down
        'p',  # Pentagon
        'h',  # Hexagon
        '*'   # Star
    ]
    
    # Organize benign data by user ID
    benign_trajectories = {uid: [] for uid in range(num_benign)}
    
    for benign_sims in benign_sims_by_round:
        for uid in range(num_benign):
            if uid < len(benign_sims):
                benign_trajectories[uid].append(benign_sims[uid])
            else:
                benign_trajectories[uid].append(None)
    
    # Plot each benign user's trajectory
    for uid, trajectory in benign_trajectories.items():
        valid_rounds = [r for r, s in zip(rounds, trajectory) if s is not None]
        valid_sims = [s for s in trajectory if s is not None]
        
        if valid_sims:
            ax.plot(valid_rounds, valid_sims, 
                   f'{benign_markers[uid % len(benign_markers)]}-',
                   color=benign_colors[uid % len(benign_colors)],
                   linewidth=3, markersize=10, markerfacecolor='white',
                   markeredgewidth=2.5,
                   label=f'Benign user {uid + 1}',
                   zorder=4)

    # Plot attackers as lines (same as Figure 2)
    attacker_trajectories = {aid: [] for aid in range(num_attackers)}

    for round_sims in attacker_sims_by_round:
        for aid in range(num_attackers):
            if aid < len(round_sims):
                attacker_trajectories[aid].append(round_sims[aid])
            else:
                attacker_trajectories[aid].append(None)

    # Plot each attacker's trajectory
    attacker_colors = ['#DC143C', '#FF6347']
    for aid, trajectory in attacker_trajectories.items():
        valid_rounds = [r for r, s in zip(rounds, trajectory) if s is not None]
        valid_sims = [s for s in trajectory if s is not None]

        if valid_sims:
            ax.plot(valid_rounds, valid_sims, 'o-',
                   color=attacker_colors[aid % len(attacker_colors)],
                   linewidth=3, markersize=10, markerfacecolor='white',
                   markeredgewidth=2.5,
                   label=f'Attacker {aid + 1}',
                   zorder=5)

    # Styling
    ax.set_xlabel('Communication Round', fontsize=FONT_SIZE_XLABEL, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=FONT_SIZE_YLABEL, fontweight='bold')
    ax.set_title('Individual Client Similarity Evolution',
                fontsize=FONT_SIZE_PLOT_TITLE, fontweight='bold', pad=20)

    # Legend - adjust columns based on number of clients
    if num_benign > 4:
        ax.legend(loc='best', frameon=True, fancybox=True,
                 shadow=True, framealpha=0.9, ncol=2, fontsize=FONT_SIZE_LEGEND_SMALL)
    else:
        ax.legend(loc='best', frameon=True, fancybox=True,
                 shadow=True, framealpha=0.9, ncol=2)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Set x-axis limits with extra space on the left
    ax.set_xlim(0, max(rounds) + 0.5)  # Changed from 0.5 to 0 for left margin
    
    # Dynamic y-axis limits
    all_values = []
    all_values.extend(thresholds)
    
    # Add all benign values
    for trajectory in benign_trajectories.values():
        all_values.extend([v for v in trajectory if v is not None])
    
    # Add all attacker values
    for trajectory in attacker_trajectories.values():
        all_values.extend([v for v in trajectory if v is not None])
    
    y_min = min(all_values)
    y_max = max(all_values)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.4 * y_range)  # 40% extra space on top

    # Set x-ticks
    ax.set_xticks(rounds)

    # Save figure
    plt.tight_layout()

    output_path_png = output_dir / 'figure3_individual_similarity.png'
    output_path_pdf = output_dir / 'figure3_individual_similarity.pdf'

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

    print(f"Figure 3 saved to: {output_path_png}")
    plt.close()


def generate_paper_figures(json_file_path):
    """
    Generate all three figures for the paper
    """
    print("Generating enhanced figures for paper (rounds 1-14)...")
    # print("Note: If your data has 15 rounds, only the first 14 will be plotted.")

    # Generate Figure 1: Attack Performance
    plot_attack_performance_enhanced(json_file_path)

    # Generate Figure 2: Similarity Evolution (with average)
    plot_similarity_evolution_bars_style(json_file_path)
    
    # Generate Figure 3: Individual Similarity Evolution
    plot_similarity_individual_benign(json_file_path)

    print("\nAll three figures generated successfully!")
    print("Plotted rounds: 1-14")
    print("Files saved in: ./results/figures/")


if __name__ == "__main__":
    # Path to your JSON file
    json_file = './results/progressive_grmp_progressive_semantic_poisoning.json'

    # Generate all three figures
    generate_paper_figures(json_file)