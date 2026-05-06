"""
260504_fig3_performance_hierarchy_NNS.py
Fig 3, NNS version: identical structure to 260501_fig3_performance_hierarchy.py
but x-axis = Normalized Navigation Score (NNS, Eq. 2 in main.tex), not RPA.

Per Author's 2026-05-04 directive:
  1. NNS instead of RPA on x-axis (NNS = (RPA - Random) / (POMCP - Random) * 100%).
  2. "Privilege type" legend moved out of upper-right (was obscuring top bars).
  3. Purple replaced with deep red (#a50f15, ColorBrewer Reds-9 darkest) for the
     "privileged" category — more standard / professional / attractive.
  4. Mouse now gets BOTH a gold bar AND the existing gold star marker
     (per Author: "I like Mouse as gold but why isn't there a bar?").

The original RPA-axis version (260501_fig3_performance_hierarchy.py) is
preserved and re-rendered to a supplement figure path so the supplement can
keep showing performance in raw RPA units.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

OUT_DIR = '<REPO_ROOT>/paper/figures/'
TS = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# ---- NNS reference points (from canonical-60 paper_values.tex) ----
# 2026-05-04 per Author: NNS denominator switched from POMCP to Greedy_oracle so
# all bars stay in [0, 100%]. POMCP retained as a reference line.
RANDOM_RPA = 0.178
POMCP_RPA = 0.820
GREEDY_RPA = 0.878
MOUSE_RPA = 0.274
NNS_DENOM = GREEDY_RPA - RANDOM_RPA  # 0.700


def rpa_to_nns(rpa):
    return 100.0 * (rpa - RANDOM_RPA) / NNS_DENOM


def rpa_ci_to_nns_ci(ci):
    """Convert RPA CI half-width to NNS-percent CI half-width (linear scaling)."""
    return 100.0 * ci / NNS_DENOM


CAT_COLORS = {
    'fixed':      '#888888',   # gray (unchanged)
    'online':     '#3b8db8',   # blue (unchanged)
    'pretrained': '#2ca25f',   # green (unchanged)
    'privileged': '#a50f15',   # ColorBrewer Reds-9 darkest — replaces purple
}

CAT_LABELS = {
    'fixed':      'Fixed policy',
    'online':     'Online learning only',
    'pretrained': 'Cross-session pretraining',
    'privileged': 'Privileged information',
}

CAT_PLOT_ORDER = ['privileged', 'pretrained', 'online', 'fixed']

MARKER_SHAPE = {
    'sgd':         'o',
    'planner':     'D',
    'eligibility': '^',
    'hidden':      'v',
}

PRIVILEGE_HATCH = {
    'graph':      '',
    'leakage':    '',  # was 'xx'; removed per Author follow-up to drop all hashing
}

# Per Author follow-up to #30: drop the per-bar hatching (poorly described in
# legend); instead extend the background-band shading from {'fixed',
# 'pretrained'} to all four categories so each gets a faint colored
# background like green/gray already had.
CAT_HATCH = {c: '' for c in ['privileged', 'pretrained', 'online', 'fixed']}

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

    # Fixed policy
    ('NoBk+FullFwd',         'fixed',      0.576, 0.023, None),
    ('Random_forward',       'fixed',      0.385, 0.012, None),
    ('Mouse-bias (ego)',     'fixed',      0.259, 0.014, None),
    ('FullFwd',              'fixed',      0.198, 0.009, None),
    ('Random',               'fixed',      0.178, 0.005, None),
    ('Mouse-bias (allo-latent)', 'fixed',  0.162, 0.007, None),
    ('Mouse-bias (allo-real)',   'fixed',  0.154, 0.010, None),

    # Mouse (special — pretrained band)
    ('Mouse',                'pretrained', 0.274, 0.032, 'mouse'),
]


def plot_fig3_nns(metric='nns'):
    """Render Fig 3 either in NNS (default, new version) or RPA (supplement)."""

    # Sort within each category by RPA descending; Mouse separately.
    agents_by_cat = {c: [] for c in CAT_PLOT_ORDER}
    mouse = None
    for a in AGENTS:
        if a[0] == 'Mouse':
            mouse = a
            continue
        agents_by_cat[a[1]].append(a)
    for c in agents_by_cat:
        agents_by_cat[c].sort(key=lambda x: x[2], reverse=True)

    plot_order = []
    for c in CAT_PLOT_ORDER:
        for a in agents_by_cat[c]:
            plot_order.append(a)
    boundary = sum(len(agents_by_cat[c]) for c in ['privileged', 'pretrained'])
    plot_order.insert(boundary, mouse)

    n = len(plot_order)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 9))

    for i, agent in enumerate(plot_order):
        label, cat, rpa, ci, sub = agent
        if metric == 'nns':
            x_val = rpa_to_nns(rpa)
            x_ci = rpa_ci_to_nns_ci(ci)
        else:
            x_val = rpa
            x_ci = ci

        if label == 'Mouse':
            # Author 2026-05-04: black bar + star (was gold). Edge / star outline still
            # darker so the star is visible against the bar.
            bar_kwargs = {'color': 'black', 'edgecolor': 'black',
                          'linewidth': 1.0, 'height': 0.65, 'zorder': 2}
            ax.barh(y[i], x_val, **bar_kwargs)
            ax.scatter([x_val], [y[i]], marker='*', s=320,
                       edgecolors='black', facecolors='white',
                       linewidths=1.6, zorder=5, label=None)
            ax.errorbar([x_val], [y[i]], xerr=[x_ci], fmt='none',
                        ecolor='black', capsize=3, zorder=4)
            continue

        color = CAT_COLORS[cat]
        # Apply category-default hatch unless the privileged subtype overrides
        cat_default_hatch = CAT_HATCH.get(cat) or ''
        bar_kwargs = {'color': color, 'edgecolor': 'black', 'linewidth': 0.6,
                      'hatch': cat_default_hatch,
                      'height': 0.65, 'zorder': 2}
        if cat == 'privileged' and sub in PRIVILEGE_HATCH and PRIVILEGE_HATCH[sub]:
            bar_kwargs['hatch'] = PRIVILEGE_HATCH[sub]
        ax.barh(y[i], x_val, **bar_kwargs)

        if cat == 'online' and sub in MARKER_SHAPE:
            marker_offset = 2.5 if metric == 'nns' else 0.02
            ax.scatter([x_val + marker_offset], [y[i]], marker=MARKER_SHAPE[sub],
                       s=80, color=color, edgecolors='black', linewidths=0.6,
                       zorder=3)
        ax.errorbar([x_val], [y[i]], xerr=[x_ci], fmt='none', ecolor='black',
                    capsize=2, linewidth=0.8, zorder=4)

    # Reference lines: Random floor, Mouse, and Greedy_oracle ceiling (since
    # the NNS denominator is now Greedy_oracle per Author 2026-05-04).
    if metric == 'nns':
        random_x = 0.0
        ceiling_x = 100.0
        mouse_x = rpa_to_nns(MOUSE_RPA)
    else:
        random_x = RANDOM_RPA
        ceiling_x = GREEDY_RPA
        mouse_x = MOUSE_RPA
    ax.axvline(random_x, linestyle=':', color='gray', linewidth=1, label='Random floor')
    ax.axvline(mouse_x, linestyle='--', color='black', linewidth=1, label='Mouse')
    ax.axvline(ceiling_x, linestyle=':', color='black', linewidth=1, label='Greedy oracle ceiling')

    labels = [a[0] for a in plot_order]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    if metric == 'nns':
        ax.set_xlabel('Normalized Navigation Score (NNS, %)', fontsize=11)
        ax.set_xlim(-10, 105)
    else:
        ax.set_xlabel('Reward Per Action (RPA)', fontsize=11)
        ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, n - 0.3)

    # Background bands
    cat_y_ranges = {c: [] for c in CAT_PLOT_ORDER}
    cur = 0
    for c in CAT_PLOT_ORDER:
        for _ in agents_by_cat[c]:
            cat_y_ranges[c].append(n - 1 - cur)
            cur += 1
        if c == 'pretrained':
            cat_y_ranges[c].append(n - 1 - cur)
            cur += 1
    for c in ['fixed', 'pretrained', 'online', 'privileged']:
        ys = cat_y_ranges[c]
        if ys:
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=CAT_COLORS[c],
                       alpha=0.08, zorder=1)

    # Author 2026-05-04 (later): all three legends left-aligned to the same x.
    # Privilege type was at (0.78, 0.72); moved slightly left to (0.68, ...).
    # Info-availability and Online-learning-update-rule legends now share the
    # same left edge so the legend stack reads as a single vertical column.
    LEGEND_LEFT_X = 0.68
    # Privilege-type legend dropped (post-Author-2026-05-05): the hatch encoding
    # was removed; graph-oracle vs test-set-leakage privileged subtypes are
    # now described in Table 1 / caption text only.

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
    legend2 = ax.legend(handles=mech_handles, loc='upper left', fontsize=8,
                        title='Online-learning update rule', title_fontsize=9,
                        frameon=True, bbox_to_anchor=(LEGEND_LEFT_X, 0.55))
    ax.add_artist(legend2)

    cat_handles = [
        mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c])
        for c in CAT_PLOT_ORDER
    ]
    legend1 = ax.legend(handles=cat_handles, loc='upper left', fontsize=8,
                        title='Information availability', title_fontsize=9,
                        frameon=True, bbox_to_anchor=(LEGEND_LEFT_X, 0.25))
    ax.add_artist(legend1)

    ax.grid(axis='x', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    suffix = '_NNS' if metric == 'nns' else '_RPA'
    out_pdf = os.path.join(OUT_DIR, f'{TS}_performance_hierarchy{suffix}.pdf')
    out_png = os.path.join(OUT_DIR, f'{TS}_performance_hierarchy{suffix}.png')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=150)
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close()


if __name__ == '__main__':
    plot_fig3_nns(metric='nns')
    plot_fig3_nns(metric='rpa')
