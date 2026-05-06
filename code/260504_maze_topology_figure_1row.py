"""
260504_maze_topology_figure_1row.py
Compact 1x4 (single-row) version of the maze-topology figure for paper/main_DandB.tex.

Per Author 2026-05-04: half the current vertical footprint, smaller panels, 1 row.

Adapted from 260501_maze_topology_figure.py.
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


def plot_maze_on_ax(adj_mat, st_positions, ax, title='', node_size=6, label_nodes=False):
    for i in range(len(adj_mat)):
        connected = np.argwhere(adj_mat[i] == 1).flatten().tolist()
        pt1 = st_positions[i]
        for j in connected:
            pt2 = st_positions[j]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '-o',
                    c='#333333', linewidth=1.4, markersize=node_size,
                    markerfacecolor='#2b7bbf', markeredgecolor='#2b7bbf')
    if label_nodes:
        for node_idx, (x, y) in enumerate(st_positions):
            ax.annotate(str(node_idx), (x, y), textcoords='offset points',
                        xytext=(7, 5), ha='center', fontsize=6)
    ax.set_title(title, fontsize=8)
    ax.set_aspect('equal')
    ax.axis('off')


fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.2))

for idx, (ax, (adj_f, pos_f, maze_label, sess_str)) in enumerate(zip(axes, MAZES)):
    adj_mat, st_pos = load_maze(MAZE_DIR + adj_f, MAZE_DIR + pos_f)
    n_nodes = int((adj_mat.sum(axis=1) > 0).sum())
    n_edges = int(adj_mat.sum() // 2)
    title = f'({chr(97 + idx)}) {maze_label}\n{sess_str}, {n_nodes} nodes, {n_edges} edges'
    plot_maze_on_ax(adj_mat, st_pos, ax, title=title, node_size=6)
    print(f'{maze_label}: {n_nodes} nodes, {n_edges} edges, {sess_str}')

plt.tight_layout()

out_pdf = os.path.join(OUTPUT_DIR, f'{TS}_maze_topologies_1row.pdf')
out_png = os.path.join(OUTPUT_DIR, f'{TS}_maze_topologies_1row.png')
plt.savefig(out_pdf, bbox_inches='tight', dpi=150)
plt.savefig(out_png, bbox_inches='tight', dpi=150)
print(f'Saved: {out_pdf}')
print(f'Saved: {out_png}')
