#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
TAB = ROOT / 'tables'
DAT = ROOT / 'data'
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 200,
    'savefig.dpi': 300,
})

# Figure 1: timing alphabet
alpha = pd.read_csv(DAT/'dswp_audio_timing_alphabet.csv')
labels = [f"{r.symbol}\n{r.center_ms:.0f} ms" for _, r in alpha.iterrows()]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(labels, alpha['count'])
ax.set_ylabel('Detected inter-click intervals')
ax.set_xlabel('Unsupervised gap symbol and center')
ax.set_title('Timing alphabet from DSWP audio recordings')
for i, c in enumerate(alpha['count']):
    ax.text(i, c + max(alpha['count'])*0.02, f'{int(c)}', ha='center', va='bottom', fontsize=8)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(FIG/'fig1_timing_alphabet.pdf')
fig.savefig(FIG/'fig1_timing_alphabet.png')
plt.close(fig)

# Figure 2: audio density families, click count and mean ICI
fam = pd.read_csv(TAB/'dswp_audio_density_family_stats.csv')
order = ['sparse_0_4','canonical_5_6','expanded_7_10','long_11_16','dense_17plus']
fam = fam.set_index('density_class').loc[order].reset_index()
fig, ax1 = plt.subplots(figsize=(6.4, 3.2))
x = np.arange(len(fam))
ax1.bar(x - 0.18, fam['nclick_med'], width=0.36, label='Median clicks')
ax1.set_ylabel('Median click count')
ax1.set_xticks(x)
ax1.set_xticklabels(['sparse\n0-4','canonical\n5-6','expanded\n7-10','long\n11-16','dense\n17+'])
ax2 = ax1.twinx()
ax2.plot(x + 0.18, fam['mean_ici_med']*1000, marker='o', label='Median mean ICI')
ax2.set_ylabel('Median mean ICI (ms)')
ax1.set_title('Density families in the DSWP audio recordings')
# combined legend
l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lab1+lab2, loc='upper center', ncols=2, frameon=False)
ax1.spines[['top']].set_visible(False); ax2.spines[['top']].set_visible(False)
fig.tight_layout()
fig.savefig(FIG/'fig2_audio_density_families.pdf')
fig.savefig(FIG/'fig2_audio_density_families.png')
plt.close(fig)

# Figure 3: operator rates for selected R classes
op = pd.read_csv(TAB/'dialogue_operator_stats.csv')
sel = [2,4,0,17,13,7,11,15,14]
op_sel = op[op['R'].isin(sel)].set_index('R').loc[sel].reset_index()
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x=np.arange(len(op_sel)); w=0.28
ax.bar(x-w, op_sel['share'], width=w, label='Corpus share')
ax.bar(x, op_sel['switch_next'], width=w, label='Next speaker switch')
ax.bar(x+w, op_sel['overlap_next'], width=w, label='Next overlap')
ax.set_xticks(x); ax.set_xticklabels([f'R{int(r)}' for r in op_sel['R']])
ax.set_ylim(0, 1.0)
ax.set_ylabel('Proportion')
ax.set_title('Interactional profiles of selected rhythm classes')
ax.legend(frameon=False, ncols=3, loc='upper center')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(FIG/'fig3_operator_rates.pdf')
fig.savefig(FIG/'fig3_operator_rates.png')
plt.close(fig)

# Figure 4: R2 microtiming minimal pair
mp = pd.read_csv(TAB/'r2_slowest_frontloading_minimal_pair.csv')
mp['class'] = mp['front_high'].map({False:'low front mass', True:'high front mass'})
fig, ax = plt.subplots(figsize=(4.8, 3.0))
x=np.arange(len(mp)); w=0.32
ax.bar(x-w/2, mp['switch'], width=w, label='Speaker switch')
ax.bar(x+w/2, mp['overlap'], width=w, label='Overlap')
ax.set_xticks(x); ax.set_xticklabels(mp['class'])
ax.set_ylim(0, 0.7)
ax.set_ylabel('Proportion')
ax.set_title('R2 slow-quartile minimal pair: internal timing matters')
for i,r in mp.iterrows():
    ax.text(i-w/2, r['switch']+0.015, f"{r['switch']:.2f}", ha='center', va='bottom', fontsize=8)
    ax.text(i+w/2, r['overlap']+0.015, f"{r['overlap']:.2f}", ha='center', va='bottom', fontsize=8)
ax.legend(frameon=False, loc='upper left')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(FIG/'fig4_r2_microtiming.pdf')
fig.savefig(FIG/'fig4_r2_microtiming.png')
plt.close(fig)

# Figure 5: pair matrix for selected classes; value = cross-whale rate for n>=10
pairs=pd.read_csv(TAB/'dialogue_pair_stats.csv')
classes=[2,4,0,17,16,13,7,11,9,15]
mat=np.full((len(classes),len(classes)), np.nan)
nmat=np.zeros_like(mat)
for _,r in pairs.iterrows():
    a=int(r['a']); b=int(r['b'])
    if a in classes and b in classes:
        i=classes.index(a); j=classes.index(b); mat[i,j]=r['cross_rate']; nmat[i,j]=r['n']
fig, ax = plt.subplots(figsize=(5.8, 4.8))
im=ax.imshow(mat, vmin=0, vmax=1, aspect='auto')
ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels([f'R{c}' for c in classes], rotation=45, ha='right')
ax.set_yticklabels([f'R{c}' for c in classes])
ax.set_xlabel('Next rhythm class')
ax.set_ylabel('Current rhythm class')
ax.set_title('Cross-whale rate for within-recording adjacent pairs')
for i in range(len(classes)):
    for j in range(len(classes)):
        if not np.isnan(mat[i,j]):
            ax.text(j,i,f"{mat[i,j]:.2f}\n({int(nmat[i,j])})",ha='center',va='center',fontsize=6)
cbar=fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Probability next row is another whale')
fig.tight_layout()
fig.savefig(FIG/'fig5_pair_heatmap.pdf')
fig.savefig(FIG/'fig5_pair_heatmap.png')
plt.close(fig)

# Figure 6: AUC comparison
auc=pd.read_csv(TAB/'switch_prediction_auc_grouped.csv')
pretty = {
    'R_only':'R only',
    'R_duration_density':'R + duration/density',
    'timing_no_R':'timing, no R',
    'full_timing_plus_R':'timing + R'
}
auc['label'] = auc['feature_set'].map(pretty)
fig, ax = plt.subplots(figsize=(5.6, 3.0))
x=np.arange(len(auc))
ax.bar(x, auc['mean_auc'])
ax.errorbar(x, auc['mean_auc'], yerr=auc['sd_auc'], fmt='none', capsize=3)
ax.axhline(0.5, linestyle='--', linewidth=1)
ax.set_ylim(0.45, 0.70)
ax.set_ylabel('Grouped 5-fold AUC')
ax.set_xticks(x); ax.set_xticklabels(auc['label'], rotation=20, ha='right')
ax.set_title('Fine timing predicts speaker switching better than R alone')
for i,r in auc.iterrows():
    ax.text(i, r['mean_auc']+r['sd_auc']+0.008, f"{r['mean_auc']:.3f}", ha='center', va='bottom', fontsize=8)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(FIG/'fig6_auc.pdf')
fig.savefig(FIG/'fig6_auc.png')
plt.close(fig)
