#%%
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt

#%%
base_path = pathlib.Path('S:/PhD/Packages/ConceptFingerprint/output/probabilityTest2')
# base_path = pathlib.Path('G:/My Drive/UniMine/Uni/PhD/ConceptConfidence/probabilityTest2')
data_name = ''
# data_name = 'STAGGERS'
# data_name = 'Arabic'
# data_name = 'AQSex'
# data_name = 'RTREE'
# data_name = 'RTREESAMPLE'
# data_name = 'AQTemp'
print(base_path.resolve())
csv_paths = list(base_path.rglob(f"run_*.csv"))
csv_paths = [x for x in csv_paths if data_name in str(x)]
print(csv_paths[0])
df = pd.read_csv(csv_paths[0])
df.head()

# %%
probs = df['concept_probs'].str.split(';', expand=True)
del_cols = []
for c in probs.columns:

    probs[c] = probs[c].str.rsplit(':').str[-1].astype('float').fillna(-0.05)
    print(f"{c}: {probs[c].sum()}")
    if probs[c].sum() < 10:
        del_cols.append(c)
probs = probs.drop(del_cols, axis=1)
probs['ts'] = probs.index
m_df = pd.melt(probs.iloc[::10, :], id_vars='ts')
m_df['variable'] = m_df['variable'].astype('category')
m_df['value'] = m_df['value'].rolling(window=100).mean()

# %%
# sns.lineplot(data=m_df.rolling(window=100).sum(), x='ts', y='value', hue='variable')
sns.lineplot(data=m_df, x='ts', y='value', hue='variable')
plt.savefig(f"explore_probabilities/{data_name}.pdf")
plt.show()
for c in m_df['variable'].unique():
    sns.lineplot(data=m_df[m_df['variable'] == c], x='ts', y='value', hue='variable')
    plt.savefig(f"explore_probabilities/{data_name}-{c}.pdf")
    plt.show()
# %%
