"""
260501_fig3_performance_hierarchy.py
Generate Fig 3: agent performance hierarchy organised by 4-rung
information-availability taxonomy (per Author review on 2026-05-01).

Rungs:
  1. Fixed policy, no privileged info
  2. Online learning only, no privileged info
  3. Cross-session pretraining, no privileged info
  4. Privileged information

Within rung 2, markers distinguish update mechanism:
  o gradient SGD          ◇ planner / online MCTS
  △ eligibility traces    ▽ hidden-state-only

Within the 'privileged' category, hatch fills distinguish privilege type:
  solid        graph + reward oracle (POMCP, Greedy)
  cross-hatch  test-set leakage (RecSAC D=32 oracle)
Mouse-trajectory clones are NOT in the 'privileged' category here:
they live in 'fixed' since their performance falls below random / mouse,
which makes the 'privileged' framing misleading.

Within each rung, agents are sorted by RPA descending. Mouse plotted as
a gold-bordered star at rung 3.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Where to save
OUT_DIR = '<REPO_ROOT>/paper/figures/'
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# Categories (renamed from "rung 1/2/3/4" — descriptors throughout, no rung number)
# 'fixed'      — fixed policy, no privileged info (incl. mouse-trajectory clones,
#                placed here because their RPA falls below random / mouse so the
#                "privileged info" framing reads as misleading)
# 'online'     — online learning only, no privileged info
# 'pretrained' — cross-session pretraining, no privileged info
# 'privileged' — privileged information beyond mouse (graph oracle / test-set leakage)
CAT_COLORS = {
    'fixed':      '#888888',   # gray
    'online':     '#3b8db8',   # blue
    'pretrained': '#2ca25f',   # green
    'privileged': '#9333a8',   # purple
}

CAT_LABELS = {
    'fixed':      'Fixed policy',
    'online':     'Online learning only',
    'pretrained': 'Cross-session pretraining',
    'privileged': 'Privileged information',
}

# Plot order — top to bottom on the y-axis
CAT_PLOT_ORDER = ['privileged', 'pretrained', 'online', 'fixed']

# Marker shape per online-learning update mechanism
MARKER_SHAPE = {
    'sgd':         'o',   # gradient SGD
    'planner':     'D',   # MCTS / online graph build
    'eligibility': '^',   # eligibility traces
    'hidden':      'v',   # hidden state only (no weight updates)
}

# Hatch per privileged-info privilege type
PRIVILEGE_HATCH = {
    'graph':      '',     # solid
    'leakage':    'xx',   # cross-hatched
}


# ---- Agents: (label, category, rpa, ci, sub-tag) ----
# sub-tag: for 'online' = update mechanism; for 'privileged' = privilege type;
#          else None. Mouse-trajectory clones live in 'fixed' per Author 260503.
AGENTS = [
    # Privileged information
    ('POMCP (c=20)',         'privileged', 0.820, 0.004, 'graph'),
    ('Greedy oracle',        'privileged', 0.878, 0.005, 'graph'),
    ('RecSAC D=32 (oracle)', 'privileged', 0.796, 0.011, 'leakage'),

    # Cross-session pretraining
    ('RecSAC (causal)',      'pretrained', 0.555, 0.034, None),
    ('DQN pretrained',       'pretrained', 0.527, 0.032, None),
    ('DRQN_seq pretrained',  'pretrained', 0.426, 0.020, None),
    ('DRQN_rand pretrained', 'pretrained', 0.426, 0.020, None),

    # Online learning only
    ('FSC_bio',              'online',     0.503, 0.019, 'eligibility'),
    ('DRQN_seq',             'online',     0.368, 0.017, 'sgd'),
    ('DQN',                  'online',     0.362, 0.018, 'sgd'),
    ('DRQN_rand',            'online',     0.313, 0.020, 'sgd'),
    ('OnlineBeliefBFS',      'online',     0.294, 0.020, 'planner'),
    ('Frontier-Plan',        'online',     0.225, 0.017, 'planner'),
    ('QRDQN',                'online',     0.248, 0.011, 'sgd'),
    ('TRPO',                 'online',     0.243, 0.013, 'sgd'),
    ('PPO',                  'online',     0.199, 0.008, 'sgd'),
    ('A2C',                  'online',     0.197, 0.011, 'sgd'),
    ('RecurrentPPO',         'online',     0.171, 0.009, 'hidden'),

    # Fixed policy — no online learning, no per-episode adaptation
    ('NoBk+FullFwd',         'fixed',      0.576, 0.023, None),
    ('Random_forward',       'fixed',      0.385, 0.012, None),
    ('Clone (ego)',          'fixed',      0.259, 0.014, None),
    ('FullFwd',              'fixed',      0.198, 0.009, None),
    ('Random',               'fixed',      0.178, 0.005, None),
    ('Clone (allo-latent)',  'fixed',      0.162, 0.007, None),
    ('Clone (allo-real)',    'fixed',      0.154, 0.010, None),

    # Mouse (special — plotted at the pretrained band)
    ('Mouse',                'pretrained', 0.274, 0.032, 'mouse'),
]


def plot_fig3():
    # Sort within each category by RPA descending; Mouse separately
    agents_by_cat = {c: [] for c in CAT_PLOT_ORDER}
    mouse = None
    for a in AGENTS:
        if a[0] == 'Mouse':
            mouse = a
            continue
        agents_by_cat[a[1]].append(a)
    for c in agents_by_cat:
        agents_by_cat[c].sort(key=lambda x: x[2], reverse=True)

    # Build the plot order: privileged top → fixed bottom
    plot_order = []
    for c in CAT_PLOT_ORDER:
        for a in agents_by_cat[c]:
            plot_order.append(a)
    # Insert Mouse at the boundary between pretrained and online
    boundary = sum(len(agents_by_cat[c]) for c in ['privileged', 'pretrained'])
    plot_order.insert(boundary, mouse)

    # y-positions (top → bottom)
    n = len(plot_order)
    y = np.arange(n)[::-1]  # top of plot has highest y, agent index 0

    fig, ax = plt.subplots(figsize=(8.5, 9))

    for i, agent in enumerate(plot_order):
        label, cat, rpa, ci, sub = agent

        if label == 'Mouse':
            # Gold star marker
            ax.scatter([rpa], [y[i]], marker='*', s=320,
                       edgecolors='#b8860b', facecolors='gold',
                       linewidths=1.6, zorder=5, label=None)
            ax.errorbar([rpa], [y[i]], xerr=[ci], fmt='none',
                        ecolor='#b8860b', capsize=3, zorder=4)
            continue

        color = CAT_COLORS[cat]

        # Bar
        bar_kwargs = {'color': color, 'edgecolor': 'black', 'linewidth': 0.6,
                      'height': 0.65, 'zorder': 2}
        if cat == 'privileged' and sub in PRIVILEGE_HATCH and PRIVILEGE_HATCH[sub]:
            bar_kwargs['hatch'] = PRIVILEGE_HATCH[sub]
        ax.barh(y[i], rpa, **bar_kwargs)

        # Marker for online-learning mechanism
        if cat == 'online' and sub in MARKER_SHAPE:
            ax.scatter([rpa + 0.02], [y[i]], marker=MARKER_SHAPE[sub],
                       s=80, color=color, edgecolors='black', linewidths=0.6,
                       zorder=3)

        # CI errorbar
        ax.errorbar([rpa], [y[i]], xerr=[ci], fmt='none', ecolor='black',
                    capsize=2, linewidth=0.8, zorder=4)

    # Reference lines
    random_rpa = 0.178
    pomcp_rpa = 0.820
    mouse_rpa = 0.274
    ax.axvline(random_rpa, linestyle=':', color='gray', linewidth=1, label='Random floor')
    ax.axvline(mouse_rpa, linestyle='--', color='#b8860b', linewidth=1, label='Mouse')
    ax.axvline(pomcp_rpa, linestyle=':', color='black', linewidth=1, label='POMCP oracle ceiling')

    # y-axis labels
    labels = [a[0] for a in plot_order]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Reward Per Action (RPA)', fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, n - 0.3)

    # Subtle alternating background for category bands
    cat_y_ranges = {c: [] for c in CAT_PLOT_ORDER}
    cur = 0
    for c in CAT_PLOT_ORDER:
        for _ in agents_by_cat[c]:
            cat_y_ranges[c].append(n - 1 - cur)
            cur += 1
        if c == 'pretrained':  # Mouse star sits in this band
            cat_y_ranges[c].append(n - 1 - cur)
            cur += 1
    for c in ['fixed', 'pretrained']:  # alternate-band shading
        ys = cat_y_ranges[c]
        if ys:
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=CAT_COLORS[c],
                       alpha=0.08, zorder=1)

    # Custom legend: categories + mechanism + privilege subtype
    cat_handles = [
        mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c])
        for c in CAT_PLOT_ORDER
    ]
    legend1 = ax.legend(handles=cat_handles, loc='lower right', fontsize=8,
                        title='Information availability', title_fontsize=9,
                        frameon=True)
    ax.add_artist(legend1)

    mech_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=7, label='gradient SGD'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=7, label='planner/MCTS'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=8, label='eligibility traces'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=7, label='hidden state only'),
    ]
    legend2 = ax.legend(handles=mech_handles, loc='center right', fontsize=8,
                        title='Online-learning update rule', title_fontsize=9,
                        frameon=True, bbox_to_anchor=(1.0, 0.45))
    ax.add_artist(legend2)

    priv_handles = [
        mpatches.Patch(facecolor=CAT_COLORS['privileged'], edgecolor='black', label='graph oracle'),
        mpatches.Patch(facecolor=CAT_COLORS['privileged'], edgecolor='black', hatch='xx', label='test-set leakage'),
    ]
    legend3 = ax.legend(handles=priv_handles, loc='upper right', fontsize=8,
                        title='Privilege type', title_fontsize=9,
                        frameon=True)
    ax.add_artist(legend3)

    ax.grid(axis='x', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    out_pdf = os.path.join(OUT_DIR, f'{TS}_performance_hierarchy.pdf')
    out_png = os.path.join(OUT_DIR, f'{TS}_performance_hierarchy.png')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=150)
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    plot_fig3()
