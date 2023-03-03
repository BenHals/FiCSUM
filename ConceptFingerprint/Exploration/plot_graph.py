import argparse
from collections import deque
import tqdm
import json
import pathlib
import pickle
import copy
import logging, time
from logging.handlers import RotatingFileHandler
import subprocess, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# file_loc = pathlib.Path('S:\\output\\expDefault\\ConceptFingerprint-optimalClassifierClean-9605696\Arabic\\8907\\run_1.csv')
file_loc = pathlib.Path('S:\\output\\testMultiScript\\ConceptFingerprint-optimalClassifierClean-6b2d60a\\STAGGER\\1\\run_-7240173567209892185_0.csv')
save_loc = file_loc.parent / f"{file_loc.stem}_plot.pdf"

df = pd.read_csv(file_loc)


plt.figure(figsize=(20,20))
similarity_plots = {}
model_ids = df['active_model'].unique()
print(model_ids)
print(df['all_state_buffered_similarity'].replace('-1', np.nan).replace(-1, np.nan).dropna())
# active_similarity = df['all_state_buffered_similarity'].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
active_similarity = df['active_state_active_similarity'].replace('-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
print(active_similarity)

active_concepts = df['ground_truth_concept'].fillna(method='ffill').values
active_models = df['active_model'].values

for m_id in model_ids:
    model_observations = active_similarity[m_id].dropna()
    x = list(model_observations.index.values)
    y = list(model_observations.values)
    plt.plot(x, y)

current_concept = int(active_concepts[0])
concept_start = 0
index = 0
for c in active_concepts[1:]:
    if int(c) != current_concept:
        plt.hlines(y = -0.01, xmin = concept_start, xmax = index, colors = "C{}".format(current_concept))
        current_concept = int(c)
        concept_start = index
    index += 1
plt.hlines(y = -0.01, xmin = concept_start, xmax = index, colors = "C{}".format(current_concept))

current_model = int(active_models[0])
model_start = 0
index = 0
for c in active_models[1:]:
    if int(c) != current_model:
        plt.hlines(y = -0.005, xmin = model_start, xmax = index, colors = "C{}".format(current_model))
        current_model = int(c)
        model_start = index
    index += 1
plt.hlines(y = -0.005, xmin = model_start, xmax = index, colors = "C{}".format(current_model))

current_model = active_models[0]
index = 0
for m in active_models[1:]:
    if m != current_model:
        plt.vlines(x = index, ymin = -0.01, ymax = -0.1, colors = 'red')
        current_model = m
    index += 1

# plt.show()
plt.savefig(save_loc, figsize = (40, 40))