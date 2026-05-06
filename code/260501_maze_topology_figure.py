"""
260501_maze_topology_figure.py
Generate 4-panel maze topology figure for paper/main.tex (Fig. 2).
Adds latMaz007 (41-node, larger maze) alongside latMaz100/101/103.
2x2 layout (more compact than 1x4 wide). Fig 2 also slightly smaller in main.tex.

Adapted from 260404_maze_topology_figure.py.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, '<REPO_ROOT>')
from utils_latMaz import load_maze

MAZE_DIR = '<REPO_ROOT>/data_in/mazes/'
OUTPUT_DIR = '<REPO_ROOT>/paper/figures/'
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# Benchmark mazes in the canonical 60-session set + latMaz007 (extra animal-set maze, ~41 nodes)
MAZES = [
    ('250311_latMaz100-adjacency.csv',
     '240507_latent_maze_004-5x5-state_positions.csv',
     'latMaz100', '4 sessions'),
    ('250311_latMaz101-adjacency.csv',
     '240507_latent_maze_004-5x5-state_positions.csv',
     'latMaz101', '37 sessions'),
    ('250311_latMaz103-adjacency.csv',
     '240507_latent_maze_004-5x5-state_positions.csv',
     'latMaz103', '19 sessions'),
    ('240901_latMaz007-mop001half-adjacency.csv',
     '240901_latMaz007-mop001half-state_positions.csv',
     'latMaz007', 'extended set'),
]


def plot_maze_on_ax(adj_mat, st_positions, ax, title='', node_size=10, label_nodes=False):
    """Draw maze graph on a given Axes. Closely follows utils_latMaz.plot_maze."""
    for i in range(len(adj_mat)):
        connected = np.argwhere(adj_mat[i] == 1).flatten().tolist()
        pt1 = st_positions[i]
        for j in connected:
            pt2 = st_positions[j]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '-o',
                    c='#333333', linewidth=2.0, markersize=node_size,
                    markerfacecolor='#2b7bbf', markeredgecolor='#2b7bbf')
    if label_nodes:
        for node_idx, (x, y) in enumerate(st_positions):
            ax.annotate(str(node_idx), (x, y), textcoords='offset points',
                        xytext=(9, 7), ha='center', fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')


fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
axes_flat = axes.flatten()

for idx, (ax, (adj_f, pos_f, maze_label, sess_str)) in enumerate(zip(axes_flat, MAZES)):
    adj_mat, st_pos = load_maze(MAZE_DIR + adj_f, MAZE_DIR + pos_f)
    n_nodes = int((adj_mat.sum(axis=1) > 0).sum())
    n_edges = int(adj_mat.sum() // 2)
    title = f'({chr(97 + idx)}) {maze_label}\n{sess_str}, {n_nodes} nodes, {n_edges} edges'
    plot_maze_on_ax(adj_mat, st_pos, ax, title=title, node_size=10)
    print(f'{maze_label}: {n_nodes} nodes, {n_edges} edges, {sess_str}')

plt.tight_layout()

out_pdf = os.path.join(OUTPUT_DIR, f'{TS}_maze_topologies.pdf')
out_png = os.path.join(OUTPUT_DIR, f'{TS}_maze_topologies.png')
plt.savefig(out_pdf, bbox_inches='tight', dpi=150)
plt.savefig(out_png, bbox_inches='tight', dpi=150)
print(f'Saved: {out_pdf}')
print(f'Saved: {out_png}')
