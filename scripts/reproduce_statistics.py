#!/usr/bin/env python3
"""
Reproduce numerical tables for the temporal-operator analysis.

Expected directory structure (relative to this script's parent):
  data/sperm-whale-dialogues.csv                 [from pratyushasharma/sw-combinatoriality]
  data/rhythms.p                                 [same release]
  data/ornaments.p                               [same release]
  data/dswp_audio_recording_features.csv         [derived from DSWP audio recordings]
  data/dswp_audio_timing_alphabet.csv            [derived from DSWP audio recordings]

This script recomputes:
  - within-recording operator statistics
  - within-recording adjacent-pair statistics
  - R2 slow-quartile front-loading minimal pair
  - DSWP audio density-family statistics
  - motif-compression statistics
  - grouped 5-fold AUC values for speaker-switch prediction
"""
from pathlib import Path
import collections, pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'tables_recomputed'
OUT.mkdir(exist_ok=True)

D = pd.read_csv(DATA / 'sperm-whale-dialogues.csv')
D['R'] = pickle.load(open(DATA / 'rhythms.p', 'rb'))
D['O'] = pickle.load(open(DATA / 'ornaments.p', 'rb'))
D['_orig_idx'] = np.arange(len(D))
D = D.sort_values(['REC', 'TsTo', '_orig_idx']).reset_index(drop=True)
D['end'] = D['TsTo'] + D['Duration']

# Bout boundaries: same recording timeline, split when silence after previous coda exceeds 8 s.
D['prev_REC'] = D['REC'].shift()
D['prev_end'] = D['end'].shift()
D['gap_from_prev'] = D['TsTo'] - D['prev_end']
D['new_bout'] = (D['REC'] != D['prev_REC']) | D['gap_from_prev'].isna() | (D['gap_from_prev'] > 8.0)
D['bout_id'] = D['new_bout'].cumsum()
D['pos_in_bout'] = D.groupby('bout_id').cumcount()
D['bout_len'] = D.groupby('bout_id')['R'].transform('size')
D['is_start'] = D['pos_in_bout'] == 0
D['is_end'] = D['pos_in_bout'] == D['bout_len'] - 1
D['is_medial'] = ~(D['is_start'] | D['is_end'])

# Adjacent rows within the same recording only. No transition is permitted across REC boundaries.
D['next_REC'] = D['REC'].shift(-1)
D['next_R'] = D['R'].shift(-1)
D['next_Whale'] = D['Whale'].shift(-1)
D['next_TsTo'] = D['TsTo'].shift(-1)
D['has_next_same_REC'] = D['next_REC'] == D['REC']
D['gap_next'] = D['next_TsTo'] - D['end']
D['switch_next'] = (D['Whale'] != D['next_Whale']) & D['has_next_same_REC']
D['overlap_next'] = (D['gap_next'] < 0.0) & D['has_next_same_REC']
adj = D[D['has_next_same_REC']].copy()

rows = []
for r, sub in D.groupby('R'):
    sub_adj = sub[sub['has_next_same_REC']]
    rows.append(dict(
        R=int(r), n=len(sub), share=len(sub)/len(D),
        start_rate=sub['is_start'].mean(), end_rate=sub['is_end'].mean(), medial_rate=sub['is_medial'].mean(),
        start_lift=sub['is_start'].mean()/D['is_start'].mean(), end_lift=sub['is_end'].mean()/D['is_end'].mean(),
        switch_next=sub_adj['switch_next'].mean() if len(sub_adj) else np.nan,
        overlap_next=sub_adj['overlap_next'].mean() if len(sub_adj) else np.nan,
        gap_median=sub_adj['gap_next'].median() if len(sub_adj) else np.nan,
        duration_med=sub['Duration'].median(),
        mean_ici_med=(sub['Duration']/(sub['nClicks']-1)).median(),
        nclick_med=sub['nClicks'].median()
    ))
operator = pd.DataFrame(rows).sort_values('n', ascending=False)
operator.to_csv(OUT / 'dialogue_operator_stats.csv', index=False)

pairs = []
for (a, b), sub in adj.groupby(['R', 'next_R']):
    if len(sub) >= 10:
        pairs.append(dict(a=int(a), b=int(b), n=len(sub),
                          cross_rate=sub['switch_next'].mean(),
                          overlap_rate=sub['overlap_next'].mean(),
                          gap_median=sub['gap_next'].median(),
                          gap_p25=sub['gap_next'].quantile(.25),
                          gap_p75=sub['gap_next'].quantile(.75)))
pairs = pd.DataFrame(pairs).sort_values('n', ascending=False)
pairs.to_csv(OUT / 'dialogue_pair_stats.csv', index=False)

def normalized_ici_columns(df):
    out = df.copy()
    for j in range(1, 29):
        out[f'N_ici{j}'] = 0.0
    for idx, row in out.iterrows():
        k = int(row['nClicks']) - 1
        if k > 0:
            arr = np.array([row[f'ICI{j}'] for j in range(1, k+1)], dtype=float)
            total = arr.sum()
            if total > 0:
                arr = arr / total
                for j, val in enumerate(arr, start=1):
                    out.at[idx, f'N_ici{j}'] = val
    return out

D_norm = normalized_ici_columns(D)
r2 = D_norm[(D_norm['R'] == 2) & D_norm['has_next_same_REC']].copy()
r2['front2_mass'] = r2['N_ici1'] + r2['N_ici2']
slow_thr = r2['Duration'].quantile(.75)
r2s = r2[r2['Duration'] >= slow_thr].copy()
q = r2s['front2_mass'].median()
r2s['front_high'] = r2s['front2_mass'] >= q
minimal = r2s.groupby('front_high').agg(
    n=('R', 'size'), switch=('switch_next', 'mean'), overlap=('overlap_next', 'mean'),
    gap_median=('gap_next', 'median'), duration_median=('Duration', 'median'),
    front2_mass_median=('front2_mass', 'median')).reset_index()
minimal.to_csv(OUT / 'r2_slowest_frontloading_minimal_pair.csv', index=False)

A = pd.read_csv(DATA / 'dswp_audio_recording_features.csv')
family = A.groupby('density_class').agg(n=('name', 'count'), nclick_med=('nclick', 'median'),
    mean_ici_med=('mean_ici_s', 'median'), front2_mass_med=('front2_mass', 'median'),
    fast_share_med=('fast_share', 'median'), long_share_med=('long_share', 'median')).reset_index().sort_values('n', ascending=False)
family.to_csv(OUT / 'dswp_audio_density_family_stats.csv', index=False)

bigram_counts = collections.Counter()
for s in A['gap_string'].dropna().astype(str):
    for j in range(len(s)-1):
        bigram_counts[s[j:j+2]] += 1
top12 = set(k for k, _ in bigram_counts.most_common(12))
rows = []
for fam, sub in A.groupby('density_class'):
    cov = []
    for s in sub['gap_string'].dropna().astype(str):
        if len(s) < 2:
            continue
        total = len(s)-1
        cov.append(sum(1 for j in range(total) if s[j:j+2] in top12)/total)
    rows.append(dict(density_class=fam, n=len(cov),
                     median_top12_bigram_coverage=float(np.median(cov)) if cov else np.nan,
                     mean_top12_bigram_coverage=float(np.mean(cov)) if cov else np.nan))
pd.DataFrame(rows).to_csv(OUT / 'dswp_audio_motif_compression.csv', index=False)

# AUC: grouped by REC, with no leakage across recordings.
adj_auc = normalized_ici_columns(adj)
adj_auc['mean_ici'] = adj_auc['Duration']/(adj_auc['nClicks']-1)
adj_auc['density'] = adj_auc['nClicks']/adj_auc['Duration']
adj_auc['front2_mass'] = adj_auc['N_ici1'] + adj_auc['N_ici2']
Z = adj_auc[[f'N_ici{j}' for j in range(1, 29)]].replace(0, np.nan)
adj_auc['cv_norm_ici'] = Z.std(axis=1) / Z.mean(axis=1)
adj_auc['first_last_ratio'] = np.nan
for idx, row in adj_auc.iterrows():
    k = int(row['nClicks']) - 1
    if k > 1:
        last = row[f'N_ici{k}']
        if last > 0:
            adj_auc.at[idx, 'first_last_ratio'] = row['N_ici1'] / last
adj_auc = adj_auc.replace([np.inf, -np.inf], np.nan)

feature_sets = {
    'R_only': (['R'], []),
    'R_duration_density': (['R'], ['Duration', 'nClicks', 'mean_ici', 'density', 'O']),
    'timing_no_R': ([], ['Duration', 'nClicks', 'mean_ici', 'density', 'front2_mass', 'cv_norm_ici', 'first_last_ratio', 'O'] + [f'N_ici{j}' for j in range(1, 29)]),
    'full_timing_plus_R': (['R'], ['Duration', 'nClicks', 'mean_ici', 'density', 'front2_mass', 'cv_norm_ici', 'first_last_ratio', 'O'] + [f'N_ici{j}' for j in range(1, 29)]),
}
y = adj_auc['switch_next'].astype(int).values
groups = adj_auc['REC'].values
gkf = GroupKFold(n_splits=5)
auc_rows = []
fold_rows = []
for name, (cat, num) in feature_sets.items():
    scores = []
    for fold, (train, test) in enumerate(gkf.split(adj_auc, y, groups), start=1):
        transformers = []
        if cat:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat))
        if num:
            transformers.append(('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num))
        model = make_pipeline(ColumnTransformer(transformers, remainder='drop'),
                              LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'))
        model.fit(adj_auc.iloc[train], y[train])
        p = model.predict_proba(adj_auc.iloc[test])[:, 1]
        auc = roc_auc_score(y[test], p)
        scores.append(auc)
        fold_rows.append({'feature_set': name, 'fold': fold, 'auc': auc, 'n_test': len(test)})
    auc_rows.append({'feature_set': name, 'mean_auc': float(np.mean(scores)), 'sd_auc': float(np.std(scores))})
pd.DataFrame(auc_rows).to_csv(OUT / 'switch_prediction_auc_grouped.csv', index=False)
pd.DataFrame(fold_rows).to_csv(OUT / 'switch_prediction_auc_folds.csv', index=False)

summary = pd.DataFrame([
    {'metric': 'annotated_codas', 'value': len(D)},
    {'metric': 'recordings_REC', 'value': D['REC'].nunique()},
    {'metric': 'whale_ids', 'value': D['Whale'].nunique()},
    {'metric': 'bouts_gap8s', 'value': D['bout_id'].nunique()},
    {'metric': 'adjacent_pairs_same_REC', 'value': len(adj)},
    {'metric': 'base_switch_rate', 'value': adj['switch_next'].mean()},
    {'metric': 'base_overlap_rate', 'value': adj['overlap_next'].mean()},
    {'metric': 'audio_recordings_with_detected_codas', 'value': len(A)},
])
summary.to_csv(OUT / 'corpus_summary.csv', index=False)
print(summary.to_string(index=False))
print('\nWrote recomputed tables to', OUT)
