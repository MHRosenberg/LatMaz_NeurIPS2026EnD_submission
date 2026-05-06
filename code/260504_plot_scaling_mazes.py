"""
260504_plot_scaling_mazes.py

Visualise the synthetic mazes used in the V=36 → V=4000 scaling sweep
(`260502_run_scaling_sweep.py`/`v2`). Lays out one example maze per
V_target in a 2x4 grid for the appendix figure.

Per Author 2026-05-04: "I also want to view the actual mazes that were used.
Plot those using our standard latent graph depictions."

Output:
  paper/figures/c{ts}_scaling_mazes_grid.pdf  (and .png)
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('<REPO_ROOT>')
sys.path.insert(0, str(Path(__file__).parent))

# Re-use the v2 sweep's lattice-subgraph maze generator so we plot the
# same family the actual benchmark ran on.
from importlib import import_module
sweep_v2 = import_module('260503_run_scaling_sweep_v2')
make_lattice = sweep_v2.make_lattice_subgraph_maze

OUTPUT_DIR = PROJECT_ROOT / 'paper' / 'figures'
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

V_TARGETS = [36, 100, 225, 400, 900, 1500, 2500, 4000]
INSTANCE_SEED = 0   # fixed for reproducibility


def plot_maze_on_ax(adj_mat, st_pos, ax, title='', node_size=None, line_width=None):
    """Same visual style as `paper/figures/c260504-160116_maze_topologies_1row.pdf`."""
    n = len(adj_mat)
    # Node/line size shrinks with n so dense mazes still look clean.
    ns = node_size if node_size is not None else max(0.6, min(8.0, 80.0 / np.sqrt(n)))
    lw = line_width if line_width is not None else max(0.15, min(1.3, 12.0 / np.sqrt(n)))
    for i in range(n):
        for j in np.where(adj_mat[i] == 1)[0]:
            if j > i:
                pt1, pt2 = st_pos[i], st_pos[j]
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                        '-', c='#555555', linewidth=lw, zorder=1)
    ax.scatter([p[0] for p in st_pos], [p[1] for p in st_pos],
               s=ns, c='#2b7bbf', edgecolors='#2b7bbf', zorder=2, linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.5))
    axes_flat = axes.flatten()
    for idx, V in enumerate(V_TARGETS):
        adj_mat, st_pos, meta = make_lattice(V, instance_seed=INSTANCE_SEED)
        n = adj_mat.shape[0]
        n_edges = int(adj_mat.sum() // 2)
        side = meta.get('side', int(np.ceil(np.sqrt(V))))
        title = f'V={V}\\\\n=$n$={n}, edges={n_edges}, lattice={side}$\\times${side}'
        # use raw to avoid double-newline (LaTeX figures render literal \n)
        title = f'V$_\\mathrm{{target}}$={V}\nn={n}, edges={n_edges}, {side}$\\times${side}'
        plot_maze_on_ax(adj_mat, st_pos, axes_flat[idx], title=title)
        print(f'V={V}: n_nodes={n}, n_edges={n_edges}, side={side}')

    plt.tight_layout()
    out_pdf = OUTPUT_DIR / f'{TS}_scaling_mazes_grid.pdf'
    out_png = OUTPUT_DIR / f'{TS}_scaling_mazes_grid.png'
    plt.savefig(out_pdf, bbox_inches='tight', dpi=180)
    plt.savefig(out_png, bbox_inches='tight', dpi=180)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
