#!/usr/bin/env python3
"""
Plot consolidated session metadata for inspection.

Usage:
  conda run -n latMaz_RL python3 code/260414_plot_session_metadata.py
"""

import os
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_OUT     = os.path.join(PROJECT_ROOT, 'data_out')
REPORTS_DIR  = os.path.join(PROJECT_ROOT, 'reports', 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)

TIMESTAMP = 'c' + datetime.now().strftime('%y%m%d-%H%M%S')

# ---------------------------------------------------------------------------
# Load most recent session_metadata CSV
# ---------------------------------------------------------------------------
candidates = sorted([
    f for f in os.listdir(DATA_OUT)
    if f.endswith('_session_metadata.csv')
])
if not candidates:
    raise FileNotFoundError('No session_metadata CSV found in data_out/')
csv_path = os.path.join(DATA_OUT, candidates[-1])
print(f'Loading: {csv_path}')
df = pd.read_csv(csv_path)

# Parse exp_session_date as a proper date for time-axis plots
def _parse_date(s):
    s = str(s).strip()
    if re.match(r'^\d{6}$', s):
        try:
            return pd.Timestamp(f'20{s[:2]}-{s[2:4]}-{s[4:6]}')
        except Exception:
            return pd.NaT
    return pd.NaT

df['date'] = df['exp_session_date'].apply(_parse_date)

# ---------------------------------------------------------------------------
# Figure 1: Sessions per animal over time (timeline)
# ---------------------------------------------------------------------------
animals = sorted(df['animal_ID'].unique())
cmap = plt.cm.get_cmap('tab20', len(animals))
animal_color = {a: cmap(i) for i, a in enumerate(animals)}

fig, ax = plt.subplots(figsize=(14, 5))
y_ticks = []
y_labels = []
for i, animal in enumerate(animals):
    sub = df[df['animal_ID'] == animal].dropna(subset=['date'])
    dates = sub['date']
    ax.scatter(dates, [i] * len(dates), color=animal_color[animal], s=18, alpha=0.8, zorder=3)
    y_ticks.append(i)
    y_labels.append(animal)

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Session date')
ax.set_title('Sessions per animal over time')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
out1 = os.path.join(REPORTS_DIR, f'{TIMESTAMP}_sessions_timeline.png')
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f'Saved: {out1}')

# ---------------------------------------------------------------------------
# Figure 2: Reward function distribution (pie + bar)
# ---------------------------------------------------------------------------
rwd_counts = df['reward_function'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
axes[0].barh(rwd_counts.index, rwd_counts.values, color='steelblue')
axes[0].set_xlabel('Count')
axes[0].set_title('Reward function distribution')
for bar, val in zip(axes[0].patches, rwd_counts.values):
    axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 str(val), va='center', fontsize=9)
axes[0].tick_params(axis='y', labelsize=8)

# Pie chart
wedge_labels = [f'{k}\n(n={v})' for k, v in rwd_counts.items()]
axes[1].pie(rwd_counts.values, labels=wedge_labels, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 8})
axes[1].set_title('Reward function share')

plt.tight_layout()
out2 = os.path.join(REPORTS_DIR, f'{TIMESTAMP}_reward_function.png')
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f'Saved: {out2}')

# ---------------------------------------------------------------------------
# Figure 3: Reverse-actions-allowed distribution per animal
# ---------------------------------------------------------------------------
rev_map = {
    True:  'False (reverse blocked)',
    False: 'True (reverse allowed: numeric)',
    'True (inferred: backwards actions present in data)': 'True (inferred: no usr_params)',
}
# Recode for display
def _rev_label(v):
    if v is True or v == True:
        return 'reverse blocked (inf)'
    if v is False or v == False:
        return 'reverse allowed (numeric threshold)'
    return 'reverse allowed (inferred, no usr_params)'

df['rev_label'] = df['reverse_actions_allowed'].apply(_rev_label)
rev_order = ['reverse blocked (inf)', 'reverse allowed (numeric threshold)',
             'reverse allowed (inferred, no usr_params)']
rev_colors = ['#d62728', '#2ca02c', '#aec7e8']

animal_order = sorted(df['animal_ID'].unique())
bottom_vals = np.zeros(len(animal_order))

fig, ax = plt.subplots(figsize=(12, 5))
for cat, color in zip(rev_order, rev_colors):
    counts = [len(df[(df['animal_ID'] == a) & (df['rev_label'] == cat)]) for a in animal_order]
    ax.bar(animal_order, counts, bottom=bottom_vals, label=cat, color=color)
    bottom_vals += np.array(counts)

ax.set_xlabel('Animal ID')
ax.set_ylabel('Session count')
ax.set_title('Reverse actions allowed — per animal')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
out3 = os.path.join(REPORTS_DIR, f'{TIMESTAMP}_reverse_actions.png')
fig.savefig(out3, dpi=150)
plt.close(fig)
print(f'Saved: {out3}')

# ---------------------------------------------------------------------------
# Figure 4: Maze ID usage per animal (stacked bar)
# ---------------------------------------------------------------------------
mazes = sorted(df['maze_ID'].unique())
maze_cmap = plt.cm.get_cmap('tab20', len(mazes))
maze_color = {m: maze_cmap(i) for i, m in enumerate(mazes)}

bottom_vals = np.zeros(len(animal_order))
fig, ax = plt.subplots(figsize=(13, 5))
for maze in mazes:
    counts = [len(df[(df['animal_ID'] == a) & (df['maze_ID'] == maze)]) for a in animal_order]
    ax.bar(animal_order, counts, bottom=bottom_vals, label=maze, color=maze_color[maze])
    bottom_vals += np.array(counts, dtype=float)

ax.set_xlabel('Animal ID')
ax.set_ylabel('Session count')
ax.set_title('Maze used — per animal')
ax.legend(loc='upper right', fontsize=6, ncol=2)
plt.tight_layout()
out4 = os.path.join(REPORTS_DIR, f'{TIMESTAMP}_maze_per_animal.png')
fig.savefig(out4, dpi=150)
plt.close(fig)
print(f'Saved: {out4}')

# ---------------------------------------------------------------------------
# Figure 5: PNG validation and rwd_initial_source provenance
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# PNG validated per animal
png_val_counts = df.groupby('animal_ID')['png_validated'].sum().reindex(animal_order, fill_value=0)
png_tot_counts = df.groupby('animal_ID').size().reindex(animal_order, fill_value=0)
axes[0].bar(animal_order, png_tot_counts, label='total', color='lightgrey')
axes[0].bar(animal_order, png_val_counts, label='PNG validated', color='royalblue')
axes[0].set_xlabel('Animal ID')
axes[0].set_ylabel('Sessions')
axes[0].set_title('PNG-validated reward states per animal')
axes[0].legend(fontsize=9)

# rwd_initial_source breakdown
src_counts = df['rwd_initial_source'].value_counts()
src_colors = {'png_validated': 'royalblue', 'usr_params': 'green',
              'rwd_configs_df_first_onset': 'orange', 'corrected_df': 'purple',
              'unknown': 'red'}
bar_colors = [src_colors.get(k, 'grey') for k in src_counts.index]
axes[1].barh(src_counts.index, src_counts.values, color=bar_colors)
axes[1].set_xlabel('Count')
axes[1].set_title('rewarded_states_initial source provenance')
for bar, val in zip(axes[1].patches, src_counts.values):
    axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va='center', fontsize=9)
axes[1].tick_params(axis='y', labelsize=8)

plt.tight_layout()
out5 = os.path.join(REPORTS_DIR, f'{TIMESTAMP}_png_and_sources.png')
fig.savefig(out5, dpi=150)
plt.close(fig)
print(f'Saved: {out5}')

# ---------------------------------------------------------------------------
# Figure 6: Sessions over time coloured by reward_function
# ---------------------------------------------------------------------------
rwd_funcs = df['reward_function'].unique()
rwd_cmap = plt.cm.get_cmap('Set1', len(rwd_funcs))
rwd_color = {r: rwd_cmap(i) for i, r in enumerate(rwd_funcs)}

fig, ax = plt.subplots(figsize=(14, 4))
for rwd in rwd_funcs:
    sub = df[df['reward_function'] == rwd].dropna(subset=['date'])
    ax.scatter(sub['date'], [rwd] * len(sub), color=rwd_color[rwd],
               s=20, alpha=0.7, label=rwd)

ax.set_xlabel('Session date')
ax.set_title('Reward function over time')
ax.tick_params(axis='y', labelsize=7)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
out6 = os.path.join(REPORTS_DIR, f'{TIMESTAMP}_rwd_function_timeline.png')
fig.savefig(out6, dpi=150)
plt.close(fig)
print(f'Saved: {out6}')

print(f'\nAll plots written to: {REPORTS_DIR}')
