#!/usr/bin/env python3
"""
Generate a detailed POMCP algorithm report with LaTeX math,
variant tree, code mapping, and assumptions audit.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate latMaz_RL
    python code/generate_pomcp_algorithm_report.py
"""
import os
import sys
import tempfile
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, Preformatted, KeepTogether, HRFlowable,
)
from PIL import Image as PILImage

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
TMPDIR = tempfile.mkdtemp(prefix='pomcp_report_')

# ======================================================================
# Math rendering via matplotlib
# ======================================================================
_eq_counter = [0]

def render_latex(latex_str, fontsize=14, width=6.5, dpi=150):
    """Render LaTeX string to a PNG image, return path."""
    _eq_counter[0] += 1
    path = os.path.join(TMPDIR, f'eq_{_eq_counter[0]:03d}.png')

    fig, ax = plt.subplots(figsize=(width, 0.1))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    text = ax.text(0.0, 0.5, latex_str,
                   fontsize=fontsize, ha='left', va='center',
                   transform=ax.transAxes,
                   usetex=False,  # use mathtext, not full LaTeX
                   math_fontfamily='cm')

    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    plt.close(fig)
    return path


def render_equation_block(lines, fontsize=12, width=6.5, dpi=150):
    """Render multiple lines of math/text as a block image."""
    _eq_counter[0] += 1
    path = os.path.join(TMPDIR, f'eq_{_eq_counter[0]:03d}.png')

    n = len(lines)
    fig_h = max(0.4, 0.35 * n)
    fig, ax = plt.subplots(figsize=(width, fig_h))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    for i, line in enumerate(lines):
        y = 1.0 - (i + 0.5) / n
        ax.text(0.02, y, line, fontsize=fontsize, ha='left', va='center',
                transform=ax.transAxes, family='monospace',
                usetex=False)

    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    plt.close(fig)
    return path


def render_diagram(draw_func, figsize=(6.5, 4), dpi=150):
    """Render a matplotlib diagram to PNG, return path."""
    _eq_counter[0] += 1
    path = os.path.join(TMPDIR, f'eq_{_eq_counter[0]:03d}.png')

    fig, ax = plt.subplots(figsize=figsize)
    draw_func(fig, ax)
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                pad_inches=0.1, facecolor='white')
    plt.close(fig)
    return path


def img_flowable(path, max_width=6.2*inch, max_height=7.5*inch):
    """Create a ReportLab Image flowable from a PNG, auto-scaled."""
    with PILImage.open(path) as img:
        w, h = img.size
    scale = min(max_width / w, max_height / h, 1.0)
    return Image(path, width=w * scale, height=h * scale)


# ======================================================================
# Variant tree diagram
# ======================================================================
def draw_variant_tree(fig, ax):
    """Draw the POMCP variant tree showing what each variant assumes."""
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 6.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Boxes
    boxes = {
        'POMDP':      (5, 6,   '#e8e8e8', 'POMDP Framework\n(general formulation)'),
        'POMCP':      (2.5, 4, '#a8d5a8', 'POMCP (omniscient)\nKnows: T, O, R, rebait\nParticle belief tracking'),
        'POMCP_bio':  (7.5, 4, '#c5a8d5', 'POMCP_bio (no prior knowledge)\nLearns: T, O, R, rebait\nThree-phase decision cycle'),
        'Explore':    (5, 1.8, '#f5c6c6', 'EXPLORE phase\nRandom unexplored dir\nat frontier node'),
        'Navigate':   (7.5, 1.8, '#f5e6c6', 'NAVIGATE phase\nBFS to nearest frontier\nover known edges'),
        'Exploit':    (10, 1.8, '#c6d5f5', 'EXPLOIT phase\nPOMCP planning on\nlearned subgraph'),
    }

    for key, (x, y, color, text) in boxes.items():
        bbox = dict(boxstyle='round,pad=0.4', facecolor=color,
                    edgecolor='#333', linewidth=1.5)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=8, bbox=bbox, linespacing=1.4)

    # Arrows
    arrows = [
        (5, 5.55, 2.5, 4.55),   # POMDP -> POMCP
        (5, 5.55, 7.5, 4.55),   # POMDP -> POMCP_bio
        (7.5, 3.45, 5, 2.35),   # POMCP_bio -> Explore
        (7.5, 3.45, 7.5, 2.35), # POMCP_bio -> Navigate
        (7.5, 3.45, 10, 2.35),  # POMCP_bio -> Exploit
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333',
                                    lw=1.5, connectionstyle='arc3,rad=0'))

    # Edge labels
    ax.text(3.2, 5.1, 'perfect model\ngiven a priori',
            fontsize=7, ha='center', color='#555', style='italic')
    ax.text(6.8, 5.1, 'no model\nlearns online',
            fontsize=7, ha='center', color='#555', style='italic')

    # Assumption annotations
    ax.text(0.2, 2.5,
            'POMCP assumptions:\n'
            '1. Knows adjacency matrix\n'
            '2. Knows reward locations\n'
            '3. Knows rebait rule\n'
            '4. Knows observation model\n'
            '5. Perfect state tracking',
            fontsize=7, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='#e8f5e8',
                      edgecolor='#aaa'),
            family='monospace')

    ax.text(0.2, 0.4,
            'POMCP_bio removes 1-4:\n'
            '1. Discovers edges by moving\n'
            '2. Learns rewards by visiting\n'
            '3. Infers rebait empirically\n'
            '4. Obs model from memory only\n'
            '5. Perfect state tracking (kept)',
            fontsize=7, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='#f0e8f5',
                      edgecolor='#aaa'),
            family='monospace')


# ======================================================================
# MCTS tree diagram
# ======================================================================
def draw_mcts_tree(fig, ax):
    """Draw MCTS tree structure showing V-nodes, Q-nodes, rollout."""
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 8)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Root V-node (belief node)
    ax.text(5, 7.2, r'$V_0$: Root belief $b_0$', fontsize=10, ha='center',
            va='center', bbox=dict(boxstyle='circle,pad=0.3',
                                   facecolor='#a8d5f5', edgecolor='#333', lw=1.5))

    # Actions from root
    actions = [(2, 5.5, 'N'), (4, 5.5, 'E'), (6, 5.5, 'S'), (8, 5.5, 'W')]
    for x, y, label in actions:
        ax.plot([5, x], [6.7, y + 0.4], 'k-', lw=1)
        ax.text(x, y, f'$Q({label})$\n$n_a, \\bar{{Q}}_a$', fontsize=8,
                ha='center', va='center',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='#f5d5a8',
                          edgecolor='#333', lw=1))

    # UCB annotation
    ax.annotate('SELECT via UCB1',
                xy=(3, 6.4), xytext=(0.5, 7.5),
                fontsize=8, color='#c00', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c00'))

    # Observation nodes from Q(E)
    obs_nodes = [(3, 3.5, 'obs₁'), (5, 3.5, 'obs₂')]
    for x, y, label in obs_nodes:
        ax.plot([4, x], [5.0, y + 0.4], 'k-', lw=1)
        ax.text(x, y, f'$V$: {label}\nparticles',
                fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='#a8d5f5',
                          edgecolor='#333', lw=1))

    # Rollout from leaf
    ax.plot([5, 6.5], [3.0, 1.5], 'k--', lw=1)
    ax.text(6.5, 1.2, 'ROLLOUT\n(random policy\nto max_depth)',
            fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#f5f5a8',
                      edgecolor='#333', lw=1))

    # Backup arrow
    ax.annotate('BACKUP:\n$Q_a \\leftarrow Q_a + \\frac{G - Q_a}{n_a}$',
                xy=(7, 3.5), xytext=(8.5, 2),
                fontsize=8, color='#00a', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#00a'))

    ax.set_title('MCTS Tree Structure (one simulation)', fontsize=11,
                 fontweight='bold', pad=10)


# ======================================================================
# Build the report
# ======================================================================
def main():
    output_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%y%m%d-%H%M%S')
    output_path = os.path.join(output_dir, f'{ts}_pomcp_algorithm_report.pdf')

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontSize=18, spaceAfter=12, alignment=TA_CENTER)
    h1 = ParagraphStyle('H1', parent=styles['Heading1'],
                         fontSize=15, spaceBefore=20, spaceAfter=8)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                         fontSize=12, spaceBefore=14, spaceAfter=6)
    h3 = ParagraphStyle('H3', parent=styles['Heading3'],
                         fontSize=10, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle('Body', parent=styles['Normal'],
                           fontSize=10, spaceAfter=8, leading=14)
    body_sm = ParagraphStyle('BodySm', parent=styles['Normal'],
                              fontSize=9, spaceAfter=6, leading=12)
    code = ParagraphStyle('Code', parent=styles['Code'], fontSize=7.5,
                           fontName='Courier',
                           backColor=colors.Color(0.95, 0.95, 0.95),
                           leftIndent=8, rightIndent=8,
                           spaceBefore=4, spaceAfter=8)
    note_style = ParagraphStyle('Note', parent=body, fontSize=9,
                                 leftIndent=20, textColor=colors.Color(0.3, 0.3, 0.3))

    content = []

    # ── Title ──
    content.append(Paragraph(
        'POMCP Algorithm Report: Mathematical Foundations,<br/>'
        'Variant Tree, and Assumptions Audit', title_style))
    content.append(Paragraph(f'{ts} — latMaz RL Project', body))
    content.append(HRFlowable(width='100%', thickness=1, color=colors.grey))
    content.append(Spacer(1, 12))

    # ================================================================
    # SECTION 1: POMDP FORMULATION
    # ================================================================
    content.append(Paragraph('1. POMDP Formulation', h1))

    content.append(Paragraph(
        'A Partially Observable Markov Decision Process (POMDP) is defined by '
        'the 7-tuple:', body))

    eq1 = render_latex(
        r'$\mathrm{POMDP} = \langle S,\; A,\; T,\; R,\; \Omega,\; O,\; \gamma \rangle$',
        fontsize=15)
    content.append(img_flowable(eq1))
    content.append(Spacer(1, 6))

    defs = [
        ('<b>S</b> — State space: <i>s = (node, rewards_tuple)</i>. '
         'Node is the agent\'s position (0..N-1). Rewards_tuple tracks '
         'which nodes still have uncollected pellets.'),
        ('<b>A</b> — Action space: {N, E, S, W} = {0, 1, 2, 3}. '
         'Allocentric compass directions.'),
        ('<b>T(s\'|s,a)</b> — Transition model: deterministic. '
         'If edge (node, direction) exists, move to neighbor; else stay. '
         'Reward at destination consumed. Rebait when sum &lt; threshold.'),
        ('<b>R(s,a,s\')</b> — Reward: R = rewards[s\'.node]. '
         'Reward equals the pellet value at the destination node (0 or 1).'),
        (u'<b>\u03a9</b> — Observation space: (bool, bool, bool, bool) for '
         'available corridors at the current node in N, E, S, W directions.'),
        ('<b>O(o|s\',a)</b> — Observation model: deterministic. '
         'Returns which compass directions have neighbors from the current node.'),
        (u'<b>\u03b3</b> — Discount factor: 0.99.'),
    ]
    for d in defs:
        content.append(Paragraph(d, body_sm))

    content.append(Paragraph(
        '<b>Key property</b>: Both T and O are <i>deterministic</i>. '
        'The environment has no stochastic transitions or noisy observations. '
        'The only source of partial observability is that the agent cannot see '
        'beyond its current node (no global map in its observation).', body))

    # ================================================================
    # SECTION 2: POMCP ALGORITHM
    # ================================================================
    content.append(PageBreak())
    content.append(Paragraph('2. POMCP Algorithm', h1))

    content.append(Paragraph(
        'Partially Observable Monte Carlo Planning (Silver &amp; Veness, 2010) '
        'combines Monte Carlo Tree Search (MCTS) with particle-based belief tracking. '
        'It plans online by simulating possible futures from the current belief state.', body))

    content.append(Paragraph('2.1 Belief Representation', h2))
    content.append(Paragraph(
        'The belief b(s) is represented as a set of <i>weighted particles</i>. '
        'In our implementation, the initial belief is:', body))

    eq2 = render_latex(
        r'$b_0 = \{s_0^{(1)}, s_0^{(2)}, \ldots, s_0^{(K)}\}$ '
        r'where $K = 100$ particles',
        fontsize=13)
    content.append(img_flowable(eq2))
    content.append(Spacer(1, 6))

    content.append(Paragraph(
        '<b>Assumption A1 (POMCP)</b>: All 100 particles are initialized to the same '
        'state <i>s₀ = (start_node, initial_rewards)</i>. This is a <b>point belief</b> — '
        'the agent starts with perfect knowledge of its position and reward map. '
        'There is no initial uncertainty.', body))

    content.append(Paragraph(
        '<b>Assumption A1 (POMCP_bio)</b>: Same point belief, but initialized with '
        '<i>believed_rewards</i> (all zeros initially, updated through experience). '
        'The agent knows its position but not the reward map.', body))

    content.append(Paragraph('2.2 MCTS Planning (per step)', h2))
    content.append(Paragraph(
        'At each timestep, the agent runs <i>num_sims</i> = 500 Monte Carlo simulations '
        'through a search tree. Each simulation has four phases:', body))

    content.append(Paragraph('<b>Phase 1: SELECTION (UCB1)</b>', h3))
    content.append(Paragraph(
        'Starting from the root, traverse the tree by selecting actions according to '
        'the Upper Confidence Bound:', body))

    eq_ucb = render_latex(
        r'$a^* = \arg\max_a \left[ \bar{Q}(a) + c \cdot \sqrt{\frac{\ln N}{n_a}} \right]$'
        r'$\quad$ where $c = 20.0$, $N = $ parent visits, $n_a = $ action visits',
        fontsize=13)
    content.append(img_flowable(eq_ucb))
    content.append(Spacer(1, 4))

    content.append(Paragraph(
        '<b>Assumption A2</b>: exploration_const = 20.0 is very high relative to reward '
        'magnitudes (0 or 1 per step). This strongly favors exploration in the search tree. '
        'With max reward ~1.0/step and c=20.0, the UCB exploration term dominates until '
        'each action has been tried many times.', body))

    content.append(Paragraph('<b>Phase 2: EXPANSION</b>', h3))
    content.append(Paragraph(
        'When a leaf node is reached, expand it by creating child Q-nodes for all '
        '4 actions {N, E, S, W}. Initial Q-values and visit counts are both 0.', body))

    content.append(Paragraph(
        '<b>Assumption A3</b>: All 4 actions are always considered valid in the search '
        'tree, even actions that would cause the agent to stay in place (no corridor). '
        'The planner discovers dead-end actions through simulation, not by pruning.', body))

    content.append(Paragraph('<b>Phase 3: ROLLOUT (random policy)</b>', h3))
    content.append(Paragraph(
        'From the newly expanded leaf, execute a random policy to max_depth:', body))

    eq_rollout = render_latex(
        r'$G_{\mathrm{rollout}} = \sum_{t=0}^{D-d} \gamma^t \cdot r_t$'
        r'$\quad$ where $D = 50$ (max_depth), $d = $ current depth',
        fontsize=13)
    content.append(img_flowable(eq_rollout))
    content.append(Spacer(1, 4))

    content.append(Paragraph(
        '<b>Assumption A4</b>: The rollout policy is uniform random over all 4 actions. '
        'This is a weak baseline — in a maze, random walks explore inefficiently. '
        'A smarter rollout (e.g., prefer unvisited nodes) would improve planning quality '
        'but is not implemented.', body))

    content.append(Paragraph('<b>Phase 4: BACKUP</b>', h3))
    content.append(Paragraph(
        'Propagate the total discounted return back up the tree, updating Q-values '
        'via incremental mean:', body))

    eq_backup = render_latex(
        r'$n_a \leftarrow n_a + 1$'
        r'$, \quad \bar{Q}(a) \leftarrow \bar{Q}(a) + \frac{G - \bar{Q}(a)}{n_a}$',
        fontsize=13)
    content.append(img_flowable(eq_backup))
    content.append(Spacer(1, 4))

    content.append(Paragraph(
        'After all 500 simulations, select the action with the highest Q-value '
        'at the root.', body))

    # MCTS tree diagram
    content.append(Spacer(1, 8))
    mcts_img = render_diagram(draw_mcts_tree, figsize=(7, 5))
    content.append(img_flowable(mcts_img))

    # ── Detailed Search Tree Section ──
    content.append(PageBreak())
    content.append(Paragraph('2.3 Search Tree: Detailed Walkthrough', h2))

    content.append(Paragraph(
        'The MCTS search tree is the central data structure of POMCP. It alternates '
        'between two node types — <b>V-nodes</b> (belief/observation nodes) and '
        '<b>Q-nodes</b> (action nodes) — forming a tree that represents the space of '
        'possible future action-observation histories.', body))

    content.append(Paragraph('<b>Node Types and Their Roles</b>', h3))

    content.append(Paragraph(
        '<b>V-nodes</b> (<i>VNode</i> / <i>VNodeParticles</i> in pomdp_py) represent '
        'belief states — points in the planning process where the agent has received '
        'an observation and must choose an action. Each V-node stores: '
        '(1) a visit count <i>N</i>, (2) child Q-nodes for each available action, and '
        '(3) in POMCP, a set of <i>particles</i> (state samples) representing the '
        'agent\'s belief at that history point. The root V-node holds the current '
        'belief <i>b₀</i>. Implementation: '
        '<code>pomdp_py.algorithms.po_uct.VNode</code> base class, extended by '
        '<code>pomcp.VNodeParticles</code> which adds a particle belief list.', body))

    content.append(Paragraph(
        '<b>Q-nodes</b> (<i>QNode</i> in pomdp_py) represent the state after taking '
        'an action but before receiving an observation. Each Q-node stores: '
        '(1) a visit count <i>n_a</i>, (2) the running mean Q-value <i>Q̄(a)</i>, and '
        '(3) child V-nodes keyed by observation. When a Q-node is expanded, it '
        'generates a V-node child for the observation received during that particular '
        'simulation. Implementation: <code>pomdp_py.algorithms.po_uct.QNode</code>.', body))

    content.append(Paragraph(
        'The tree thus has the structure: '
        '<b>V-node → Q-node → V-node → Q-node → ...</b>, alternating actions and '
        'observations at each level. A path from root to leaf encodes a complete '
        'action-observation history <i>h = (a₁, o₁, a₂, o₂, ...)</i>.', body))

    content.append(Paragraph('<b>Simulation Walkthrough (one of 500)</b>', h3))

    content.append(Paragraph(
        'Each MCTS simulation proceeds through the tree in four phases. Here we '
        'trace one simulation step by step, with pomdp_py code references:', body))

    sim_detail_code = """\
# pomdp_py/algorithms/po_uct.pyx — _simulate() [recursive]
# Called once per simulation from _search()

def _simulate(state, history, root_vnode, parent_qnode, obs, depth):

    # ── BASE CASE: max depth reached ──
    if depth > max_depth:          # max_depth=50
        return 0

    # ── EXPANSION: leaf node (root_vnode is None) ──
    if root_vnode is None:
        root_vnode = VNodeParticles()        # create new V-node
        parent_qnode[obs] = root_vnode       # attach to parent Q-node
        _expand_vnode(root_vnode, history)   # create child Q-nodes for all 4 actions
        rollout_reward = _rollout(state, history, root_vnode, depth)
        return rollout_reward                # return total discounted reward from rollout

    # ── SELECTION: pick action via UCB1 ──
    action = _ucb(root_vnode)
    # UCB1: argmax_a [ Q̄(a) + c * sqrt(ln(N) / n_a) ]
    # where c=20.0, N=root_vnode.num_visits, n_a=root_vnode[action].num_visits

    # ── SIMULATE: step the generative model ──
    next_state, obs, reward, nsteps = sample_generative_model(agent, state, action)
    # This calls: blackbox.sample(state, action) → (next_state, obs, reward)
    # which internally calls transition.sample(), observation.sample(), reward.sample()

    # ── RECURSE: continue simulation deeper ──
    total_reward = reward + (discount**nsteps) * _simulate(
        next_state, history + ((action, obs),),
        root_vnode[action][obs],              # child V-node (may be None → expansion)
        root_vnode[action],                   # parent Q-node
        obs, depth + nsteps)

    # ── BACKUP: update statistics ──
    root_vnode.num_visits += 1                # V-node visit count
    root_vnode[action].num_visits += 1        # Q-node visit count (n_a)
    root_vnode[action].value += \\
        (total_reward - root_vnode[action].value) / root_vnode[action].num_visits
    # This is the incremental mean: Q̄(a) ← Q̄(a) + (G - Q̄(a)) / n_a
    # where G = total_reward = r + γ * G_deeper

    return total_reward"""
    content.append(Preformatted(sim_detail_code, code))

    content.append(Paragraph(
        '<b>Key insight — what G contains</b>: The variable <code>total_reward</code> '
        'backed up into Q-node values is <i>not</i> just the immediate reward <i>r</i>. '
        'It is the full discounted return from that point onward: '
        '<i>G = r + γ·r\' + γ²·r\'\' + ...</i>, computed recursively. '
        'At the deepest tree node, G includes the rollout return. At shallower nodes, '
        'G includes both the tree-phase rewards and the rollout return. '
        'This is why the Phase 4 backup formula uses <i>G</i> (not <i>R</i>).', body))

    content.append(Paragraph('<b>The Rollout Phase in Detail</b>', h3))

    rollout_detail_code = """\
# pomdp_py/algorithms/po_uct.pyx — _rollout()
def _rollout(state, history, root, depth):
    discount = 1.0
    total_discounted_reward = 0

    while depth < max_depth:                   # max_depth=50
        action = rollout_policy.rollout(state, history)
        # MazeRolloutPolicy: random.choice([N, E, S, W])
        # NoveltySeekingRolloutPolicy: prefer rewarded neighbors → valid moves → random

        next_state, obs, reward, nsteps = sample_generative_model(agent, state, action)
        # Calls blackbox.sample(state, action):
        #   transition.sample(state, action) → next_state (move + consume pellet + rebait)
        #   reward.sample(state, action, next_state) → pellet value (0 or 1)

        total_discounted_reward += reward * discount
        discount *= discount_factor**nsteps    # discount_factor=0.99
        depth += nsteps
        state = next_state

    return total_discounted_reward             # G_rollout: discounted sum from leaf to max_depth"""
    content.append(Preformatted(rollout_detail_code, code))

    content.append(Paragraph(
        'The rollout runs from the current tree depth to <code>max_depth=50</code>. '
        'With a random policy, a typical rollout takes 50 steps from root (or fewer '
        'from deeper leaves). In a 16-node maze, random walks oscillate — frequently '
        'revisiting already-depleted nodes — so the rollout return underestimates '
        'the value of good plans. This is the weakness identified in Issue I4.', body))

    content.append(Paragraph('<b>Tree Growth Across 500 Simulations</b>', h3))

    content.append(Paragraph(
        'The tree starts as a single root V-node with 4 child Q-nodes (one per direction). '
        'Each simulation extends the tree by one V-node (at the expansion step). After 500 '
        'simulations, the tree has at most 500 V-nodes, though in practice it is smaller '
        'because many simulations follow existing tree paths to different depths. The tree '
        'is asymmetric — promising branches (high UCB score) are explored more deeply than '
        'poor branches.', body))

    content.append(Paragraph(
        'After all 500 simulations, the agent selects the action at the root with the '
        'highest Q̄(a) value (breaking ties by visit count). In pomdp_py:', body))

    action_sel_code = """\
# pomdp_py/algorithms/po_uct.pyx — plan()
best_action = agent.tree.argmax()   # action with highest Q̄(a) at root
# Equivalent to: argmax_a root[action].value"""
    content.append(Preformatted(action_sel_code, code))

    content.append(Paragraph('<b>Key Implementation Lines in advanced_agents.py</b>', h3))

    key_lines_code = """\
# POMCPAgent.run_episode() — main planning loop
# Line ~409: action = planner.plan(agent)          # runs 500 MCTS simulations
# Line ~410: next_state = transition.sample(...)   # execute in REAL environment
# Line ~429: planner.update(agent, action, obs)    # prune tree, retain subtree under (a, o)
#
# POMCPbioAlloAgent.run_episode() — exploit phase
# Line ~630: planner = POMCP(...)                  # FRESH planner each exploit step (no tree reuse)
# Line ~638: action = planner.plan(agent_obj)      # 500 MCTS sims on learned model
#
# MazeBlackboxModel.sample() — generative model used during MCTS simulations
# Line ~300: next_state = self.transition.sample(state, action)  # deterministic transition
# Line ~302: observation = self.observation.sample(next_state, action)
# Line ~303: reward = self.reward.sample(state, action, next_state)
#
# MazeTransitionModel._transitions — precomputed {node: {dir_idx: next_node}}
# Line ~122-129: builds the edge lookup table from adjacency matrix
# This dict is also passed to NoveltySeekingRolloutPolicy for reward-aware rollouts"""
    content.append(Preformatted(key_lines_code, code))

    content.append(Paragraph('2.4 Belief Update (after real action)', h2))
    content.append(Paragraph(
        'After executing action <i>a</i> and receiving observation <i>o</i> in '
        'the real environment:', body))

    belief_steps = [
        '1. Prune the search tree: keep only the subtree under (a, o).',
        '2. The surviving particles at that node become the new belief.',
        '3. <b>Particle reinvigoration</b>: if too few particles survived, '
        'resample from the existing ones to restore count to K=100.',
    ]
    for s in belief_steps:
        content.append(Paragraph(s, body_sm))

    content.append(Paragraph(
        '<b>Assumption A5</b>: In our implementation, the planner is rebuilt from scratch '
        'every step (POMCP_bio) or the tree is updated incrementally (POMCP). For POMCP_bio, '
        'no tree is carried forward — belief update is implicit through the agent\'s '
        'known_edges and believed_rewards state.', body))

    # ================================================================
    # SECTION 3: VARIANT TREE
    # ================================================================
    content.append(PageBreak())
    content.append(Paragraph('3. Variant Tree', h1))

    variant_img = render_diagram(draw_variant_tree, figsize=(8, 5))
    content.append(img_flowable(variant_img))
    content.append(Spacer(1, 10))

    # Comparison table
    comp_data = [
        ['Property', 'POMCP', 'POMCP_bio'],
        ['Transition model', 'Full adjacency matrix\n(given a priori)', 'Learned edges only\n(unknown → stay in place)'],
        ['Observation model', 'Full adjacency matrix\n(all node corridors)', 'Remembered corridors only\n(visited nodes)'],
        ['Reward knowledge', 'Full reward vector\n(given a priori)', 'Optimistic prior (1.0)\n→ 0.0 when visited'],
        ['Rebait knowledge', 'Known from start', 'Inferred when observed'],
        ['Planning phase', 'Every step\n(500 MCTS sims)', 'Only after maze\nfully explored'],
        ['Exploration', 'Implicit (UCB1\nin search tree)', 'Explicit three-phase\n(explore/navigate/exploit)'],
        ['Belief particles', '100, updated\nvia tree pruning', '100, point belief\nrebuilt each step'],
        ['State tracking', 'Perfect', 'Perfect'],
    ]

    t = Table(comp_data, colWidths=[1.5*inch, 2.3*inch, 2.3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.2)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (0, -1), colors.Color(0.9, 0.9, 0.9)),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))
    content.append(t)

    # ================================================================
    # SECTION 4: CODE CALL MAPPING
    # ================================================================
    content.append(PageBreak())
    content.append(Paragraph('4. Code Call Mapping', h1))

    content.append(Paragraph(
        'This section maps each algorithmic component to its implementation '
        'in the codebase.', body))

    # 4.1 POMCP (omniscient)
    content.append(Paragraph('4.1 POMCP (omniscient) — POMCPAgent.run_episode()', h2))

    pomcp_code = """\
# advanced_agents.py: POMCPAgent.run_episode() — simplified
transition = MazeTransitionModel(adj_mat, st_positions, rewards, reset_val)
observation = MazeObservationModel(adj_mat, st_positions)  # FULL adj_mat
reward_model = MazeRewardModel()
rollout_policy = MazeRolloutPolicy()    # random
blackbox = MazeBlackboxModel(transition, observation, reward_model)

init_state = MazeState(start_node, rewards_tuple)
init_belief = Particles([init_state] * 100)   # A1: point belief

agent = pomdp_py.Agent(init_belief, policy, transition,
                       observation, reward_model, blackbox)

planner = POMCP(max_depth=50, num_sims=500,
                exploration_const=20.0, discount=0.99,
                rollout_policy=rollout_policy)

for step in range(n_actions):
    action = planner.plan(agent)          # 500 MCTS simulations
    next_state = transition.sample(current_state, action)
    obs = observation.sample(next_state, action)
    reward = reward_model.sample(current_state, action, next_state)
    planner.update(agent, action, obs)    # prune tree + reinvigorate
    current_state = next_state"""
    content.append(Preformatted(pomcp_code, code))

    content.append(Paragraph(
        '<b>Key</b>: planner.update() carries the search tree forward between steps. '
        'Particles propagate through tree pruning. The agent replans at every step '
        'using 500 new simulations on top of the retained tree.', body))

    # 4.2 POMCP_bio decision cycle
    content.append(Paragraph('4.2 POMCP_bio — Three-Phase Decision Cycle', h2))

    bio_code = """\
# advanced_agents.py: POMCPbioAlloAgent.run_episode() — decision logic
avail_dirs = set(dir_to_neighbor[current_node].keys())    # REAL corridors
known_from_here = set(known_edges.get(current_node, {}).keys())
unexplored = list(avail_dirs - known_from_here)

if unexplored:                            # PHASE 1: EXPLORE
    action = MazeAction(random.choice(unexplored))

elif _has_frontier(known_edges, remembered_corridors):
    # PHASE 2: NAVIGATE                  # BFS to nearest frontier
    next_dir = _bfs_to_frontier(current_node, known_edges,
                                 remembered_corridors, blocked_node)
    action = MazeAction(next_dir)

else:                                     # PHASE 3: EXPLOIT
    # Build POMCP from learned model (FRESH planner each step)
    bio_transition = _BioTransitionModel(known_edges, ...)
    bio_state = MazeState(current_node, believed_rewards_tuple)
    bio_belief = Particles([bio_state] * 100)  # A1b: point belief
    ...
    planner = POMCP(max_depth=50, num_sims=500, ...)
    action = planner.plan(agent_obj)      # NO tree carried forward"""
    content.append(Preformatted(bio_code, code))

    # 4.3 Transition model comparison
    content.append(Paragraph('4.3 Transition Models', h2))

    trans_code = """\
# MazeTransitionModel.sample() — POMCP (omniscient)
def sample(self, state, action):
    node = state.node
    rewards = list(state.rewards)
    if action.direction in self._transitions[node]:  # FULL adjacency
        next_node = self._transitions[node][action.direction]
    else:
        next_node = node
    if next_node != node:
        rewards[next_node] = 0.0
        if sum(rewards) < self.min_rewarded:
            rewards = list(self.initial_rewards)  # KNOWS initial rewards
    return MazeState(next_node, tuple(rewards))

# _BioTransitionModel.sample() — POMCP_bio
def sample(self, state, action):
    node = state.node
    rewards = list(state.rewards)
    edges = self.known_edges.get(node, {})           # LEARNED edges only
    if action.direction in edges:
        next_node = edges[action.direction]
    else:
        next_node = node  # Unknown edge: stay in place
    if next_node != node:
        rewards[next_node] = 0.0
        if self.knows_rebait and sum(rewards) < self.min_rewarded:
            for n in self.known_edges:  # Reset only known nodes
                if n < len(rewards): rewards[n] = 1.0
    return MazeState(next_node, tuple(rewards))"""
    content.append(Preformatted(trans_code, code))

    content.append(Paragraph(
        '<b>Issue I1 (rebait — FIXED 2026-02-13)</b>: Rebait in _BioTransitionModel '
        'now resets only discovered nodes to 1.0, matching the agent\'s real '
        'believed rebait logic. Previously reset all n_nodes positions.', body))

    # 4.4 Observation model comparison
    content.append(Paragraph('4.4 Observation Models', h2))

    obs_code = """\
# MazeObservationModel — POMCP (omniscient)
def __init__(self, adj_mat, st_positions):
    # Precompute observations for ALL nodes from FULL adj_mat
    self._obs_cache = {}
    for node in range(len(adj_mat)):
        avail = [False, False, False, False]
        for j, connected in enumerate(adj_mat[node]):
            if connected == 1:
                direction = compute_compass(st_positions, node, j)
                avail[direction] = True
        self._obs_cache[node] = tuple(avail)

# _BioObservationModel — POMCP_bio
def __init__(self, remembered_corridors):
    self.remembered = remembered_corridors  # {node: (bool,bool,bool,bool)}

def _get_obs(self, node):
    if node in self.remembered:
        return self.remembered[node]      # Only visited nodes
    return (True, True, True, True)       # Unknown: optimistic fallback"""
    content.append(Preformatted(obs_code, code))

    # ================================================================
    # SECTION 5: ASSUMPTIONS AUDIT
    # ================================================================
    content.append(PageBreak())
    content.append(Paragraph('5. Assumptions Audit', h1))

    content.append(Paragraph(
        'Every assumption in the POMCP and POMCP_bio implementations, '
        'with assessment of correctness and impact.', body))

    assumptions = [
        ['ID', 'Assumption', 'Applies To', 'Correct?', 'Impact'],
        ['A1', 'Point belief: all 100 particles\nidentical at initialization',
         'Both', 'Justified*',
         'No initial uncertainty.\nAgent knows its start position.'],
        ['A2', 'exploration_const = 20.0\n(very high)',
         'Both', 'Tuned via HPO',
         'Strongly explores in MCTS tree.\nReward ~1/step, so 20x overweight.'],
        ['A3', 'All 4 actions always valid\nin search tree',
         'Both', 'Yes',
         'Dead-end actions discovered by\nsimulation, not pruned a priori.'],
        ['A4', 'Random rollout policy',
         'Both', 'Conservative',
         'Underestimates value of good plans.\nSmarter rollout would help.'],
        ['A5a', 'Tree carried forward between\nsteps (planner.update)',
         'POMCP only', 'Yes',
         'Efficient: reuses computation.\nQ-values not stale — see §7 I2.'],
        ['A5b', 'Fresh planner every exploit\nstep (no tree reuse)',
         'POMCP_bio', 'Wasteful',
         'Discards 500 sims of context\nper exploit step. Fixable — see §7 I2.'],
        ['A6', 'Deterministic transitions',
         'Environment', 'Yes',
         'No stochastic movement.\nAgent always reaches intended node.'],
        ['A7', 'Deterministic observations',
         'Environment', 'Yes',
         'No observation noise.\nCorridor info is always correct.'],
        ['A8', 'Perfect state tracking\n(agent always knows its node)',
         'Both', 'Given',
         'Not biologically realistic.\nMouse may have spatial uncertainty.'],
        ['A9', 'avail_dirs from ground truth\nat current node',
         'POMCP_bio', 'Justified',
         'Agent is physically present.\nCan "look around" at current node.'],
        ['A10', 'Optimistic reward prior\n(1.0 for new nodes)',
         'POMCP_bio', 'Heuristic',
         'Encourages visiting new nodes.\nOverestimates true reward density.'],
        ['A11', 'Rebait inferred from first\nobservation of reward reset',
         'POMCP_bio', 'Reasonable',
         'Agent notices rewards reappear.\nSingle observation sufficient.'],
        ['A12', 'Rebait resets ALL nodes\nto 1.0 (incl. wall nodes)',
         'POMCP_bio', 'ISSUE',
         'Should reset only initially-\nrewarded nodes. Mild distortion.'],
        ['A13', 'No planning during\nexplore/navigate phases',
         'POMCP_bio', 'Design choice',
         'Reactive exploration.\nPOMCP planning deferred to exploit.'],
        ['A14', 'BFS uses known_edges +\nremembered_corridors',
         'POMCP_bio', 'Correct',
         'No information leak.\nOnly uses agent\'s own observations.'],
        ['A15', 'Replay buffer for POMCP\nbelief = 100 identical particles',
         'POMCP_bio', 'Degenerate',
         'No belief diversity.\nEffectively a deterministic planner.'],
    ]

    t2 = Table(assumptions,
               colWidths=[0.35*inch, 1.6*inch, 0.85*inch, 0.75*inch, 1.8*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.2)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ]))

    # Color-code correctness column
    for i in range(1, len(assumptions)):
        val = assumptions[i][3]
        if 'ISSUE' in val:
            t2.setStyle(TableStyle([
                ('BACKGROUND', (3, i), (3, i), colors.Color(1, 0.8, 0.8))]))
        elif 'Wasteful' in val or 'Degenerate' in val:
            t2.setStyle(TableStyle([
                ('BACKGROUND', (3, i), (3, i), colors.Color(1, 1, 0.8))]))

    content.append(t2)

    content.append(Spacer(1, 12))
    content.append(Paragraph(
        '<b>*A1 "Justified"</b>: In our maze, the agent truly does know its start '
        'position (it\'s placed there). The question is whether it should know the '
        'full reward map (POMCP: yes, POMCP_bio: no).', note_style))

    # ================================================================
    # SECTION 6: PARAMETERS
    # ================================================================
    content.append(PageBreak())
    content.append(Paragraph('6. Parameter Values', h1))

    param_data = [
        ['Parameter', 'Symbol', 'Value', 'Set By', 'Sensitivity'],
        ['Max MCTS depth', 'D', '50', 'HPO', 'Medium: deeper = better but slower'],
        ['Num simulations', 'N_sim', '500', 'HPO', 'High: more sims = better plans'],
        ['Exploration const', 'c', '20.0', 'HPO', 'High: controls explore vs exploit'],
        ['Discount factor', r'γ', '0.99', 'HPO', 'Low: 0.95 vs 0.99 similar'],
        ['Initial particles', 'K', '100', 'Default', 'Low: all identical (point belief)'],
        ['Rollout policy', '—', 'Uniform random', 'Fixed', 'Medium: smarter = better'],
        ['Rebait threshold', 'τ', '2', 'Domain', 'Fixed by maze design'],
        ['Observation type', '—', 'Egocentric 4-bool', 'Fixed', 'N/A'],
    ]

    t3 = Table(param_data, colWidths=[1.3*inch, 0.5*inch, 1.1*inch, 0.6*inch, 2.0*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.2)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    content.append(t3)

    # ================================================================
    # SECTION 7: IDENTIFIED ISSUES
    # ================================================================
    content.append(Spacer(1, 20))
    content.append(Paragraph('7. Identified Issues and Potential Improvements', h1))

    issues = [
        ('<b>I1 — Rebait resets wall nodes (A12) [FIXED 2026-02-13]</b>: '
         '_BioTransitionModel.sample() previously used <code>rewards = [1.0] * n_nodes</code> '
         'on rebait, which set wall nodes (never rewarded) to 1.0. '
         'Fixed: rebait now resets only known (discovered) nodes to 1.0, matching '
         'the agent\'s real believed rebait logic. This was critical for enabling '
         'tree reuse (I2), as the all-nodes rebait caused reward state mismatch '
         'in reused MCTS particles.'),

        ('<b>I2 — No tree reuse in POMCP_bio (A5b) [FIXED 2026-02-13]</b>: '
         'Previously a fresh POMCP planner was constructed at every exploit step. '
         'Now the MCTS tree persists across consecutive exploit steps via '
         'planner.update(agent, action, obs). The tree is invalidated when the '
         'agent leaves exploit mode (explore/navigate). A try/except fallback '
         'handles rare particle deprivation errors by rebuilding from scratch. '
         'Result: +1.7pp RPA improvement (0.733 → 0.747, 91% → 93% of omniscient).'),

        ('<b>I3 — Point belief defeats particle filtering (A15)</b>: '
         'All 100 particles are identical. This makes POMCP a deterministic '
         'planner (no belief diversity). In a truly partially observable setting, '
         'particles should represent uncertainty about the state. '
         'In our maze, since transitions and observations are deterministic and '
         'the agent knows its position, a point belief is actually correct — '
         'there IS no state uncertainty to represent.'),

        ('<b>I4 — Random rollout is weak (A4)</b>: '
         'The rollout policy chooses actions uniformly at random. In a graph maze, '
         'random walks tend to revisit nodes and oscillate. A heuristic rollout '
         '(e.g., prefer unvisited adjacent nodes) could substantially improve '
         'MCTS value estimates. This would benefit both POMCP and POMCP_bio.'),

        ('<b>I5 — No planning during exploration (A13)</b>: '
         'POMCP_bio uses reactive strategies (random explore, BFS navigate) during '
         'the exploration phase. An integrated approach would plan exploration '
         'trajectories that maximize expected information gain. However, since '
         'exploration takes only 20-29 steps in these small mazes, the simple '
         'strategy is near-optimal.'),

        ('<b>I6 — Exploration constant very high (A2)</b>: '
         'c=20.0 with rewards in [0,1] means the exploration bonus dominates '
         'Q-values until each action is tried ~400 times (exp(20²/1²) ≈ e^400). '
         'Effectively, this ensures very thorough exploration of the search tree. '
         'May cause the planner to under-exploit early in the simulation budget.'),
    ]
    for issue in issues:
        content.append(Paragraph(issue, body_sm))
        content.append(Spacer(1, 4))

    # ================================================================
    # SECTION 8: PERFORMANCE CONTEXT
    # ================================================================
    content.append(PageBreak())
    content.append(Paragraph('8. Performance Context', h1))

    content.append(Paragraph(
        'Why POMCP_bio achieves 93% of POMCP despite removing knowledge assumptions:', body))

    perf_table = [
        ['Phase', 'Steps', '% Budget', 'Reward Collected', 'Information Used'],
        ['Explore', '20-29', '4-7%', '~12-16\n(one full sweep)', 'None (random)'],
        ['Navigate', '4-7', '1-2%', '~0\n(revisiting)', 'Known edges only'],
        ['Exploit', '380-660', '91-95%', 'Remainder\n(POMCP planning)', 'Full learned model'],
        ['Total', '420-695', '100%', 'See results', '—'],
    ]

    t4 = Table(perf_table, colWidths=[0.9*inch, 0.7*inch, 0.7*inch, 1.3*inch, 1.6*inch])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.2, 0.2)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, -1), (-1, -1), colors.Color(0.9, 0.9, 0.9)),
    ]))
    content.append(t4)

    content.append(Spacer(1, 10))
    content.append(Paragraph(
        '<b>Core insight</b>: With 12-16 accessible nodes and action budgets of 420-695, '
        'exploration consumes &lt;10% of the budget. Once exploration completes, '
        'POMCP_bio has a complete model and behaves identically to POMCP (minus '
        'tree reuse). The 9% performance gap is attributable to: (1) no tree reuse, '
        '(2) slightly incorrect reward beliefs after rebait, and '
        '(3) suboptimal exploration path.', body))

    # Build
    doc.build(content)
    print(f'Report: {output_path}')
    return output_path


if __name__ == '__main__':
    main()
