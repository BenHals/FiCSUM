#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, lognorm

from ConceptFingerprint.Classifier.feature_selection.fisher_score import fisher_score
from ConceptFingerprint.Classifier.feature_selection.mutual_information import bin_X, MI_histogram_estimation_cache_mat, MI_estimation_cache_mat
sns.set()
# %%
def make_f1_generator():
    def generate_f1(concept):
        return np.random.normal(50, 1)
    return generate_f1

def make_f2_generator():
    concept_means = [*np.random.normal(0, 1, size=5), *np.random.normal(100, 1, size=5)]
    def generate_f2(concept):
        mu = concept_means[concept]
        return np.random.normal(mu, 1)
    return generate_f2

def make_f3_generator():
    concept_means = [*np.random.normal(50, 10, size=10)]
    def generate_f3(concept):
        mu = concept_means[concept]
        return np.random.normal(mu, 1)
    return generate_f3


def make_f4_generator():
    concept_means = [*np.random.normal(52, 0.01, 5), *np.random.normal(48, 0.01, 5)]
    def generate_f4(concept):
        mu = concept_means[concept]
        skew = 500 if concept < 5 else -500
        v = skewnorm.rvs(loc=mu, scale=25, a=skew)
        while v < 0 or v > 100:
            v = skewnorm.rvs(loc=mu, scale=25, a=skew)
        return v
    return generate_f4
# def make_f4_generator():
#     # concept_means = [*np.random.normal(50, 1, size=10)]
#     concept_means = [*np.random.rand(10)]
#     def generate_f4(concept):
#         mu = concept_means[concept]
#         skew = 2
#         if concept < 5:
#             return max(min(lognorm.rvs(loc=mu, scale=50, s=skew), 100), 0)
#         return max(min(100 - lognorm.rvs(loc=mu, scale=50, s=skew), 100), 0)

#     return generate_f4


# %%
X = []
generators = [make_f1_generator(), make_f2_generator(), make_f3_generator(), make_f4_generator()]
for c in range(10):
    for i in range(1000):
        x = np.array([*[g(c) for g in generators], c])
        X.append(x)
print(X)
X_mat = np.vstack(X)
print(X_mat.shape)
df = pd.DataFrame(X)
# %%
df.columns = ['f1', 'f2', 'f3', 'f4', 'c']
df =df.reset_index()
df
#%%
sns.scatterplot(data=df, x='index', y='f4')
# %%

for f in ['f1', 'f2', 'f3', 'f4']:
    overall = df[f].values
    o_count = len(overall)
    o_mu = np.mean(overall)
    o_sigma = np.std(overall)
    overall_count = o_count
    o_min = overall.min()
    o_max = overall.max()
    overall_mean = o_mu
    overall_stdev = o_sigma
    overall_histogram = np.histogram(overall, bins=100)
    overall_gaussian_histogram = bin_X(o_mu, o_sigma, o_min, o_max, 100, o_count)
    concept_counts = []
    concept_means = []
    concept_stdevs = []
    concept_histograms = []
    concept_gaussian_histograms = []
    for c in df['c'].unique():
        concept_matrix = df[df['c'] == c]
        values = concept_matrix[f].values
        count = len(values)
        mu = np.mean(values)
        sigma = np.std(values)
        c_min = values.min()
        c_max = values.max()
        concept_counts.append(count)
        concept_means.append(mu)
        concept_stdevs.append(sigma)
        concept_histograms.append(np.histogram(values, bins=100, range=[o_min, o_max]))
        concept_gaussian_histograms.append(bin_X(mu, sigma, o_min, o_max, 100, count))
    binned_Y_counts = np.hstack([h.reshape(-1, 1) for h,b in concept_gaussian_histograms])
    bin_weights = np.sum(binned_Y_counts, axis=1)
    total_weight = np.sum(binned_Y_counts)
    concept_histograms_mat = np.vstack([c[0] for c in concept_histograms])
    concept_bins_mat = np.vstack([c[1] for c in concept_histograms])
    fisher_Overall = fisher_score(np.array(concept_means), np.array(concept_counts), overall_stdev)
    fisher_group = fisher_score(np.array(concept_means), np.array(concept_counts), overall_stdev, concept_stdevs)
    gaussian_MI = MI_estimation_cache_mat(np.array(concept_counts), binned_Y_counts, bin_weights, total_weight)
    histogram_MI = MI_histogram_estimation_cache_mat(overall_histogram[0], overall_histogram[1], len(overall), concept_histograms_mat, concept_bins_mat, np.array(concept_counts), total_weight)
    print(f)
    print(f"fisher overall: {fisher_Overall}")
    print(f"fisher group: {fisher_group}")
    # Gaussian_MI estimates inter-concept distribution, so fixes the overestimation of f2
    # but not the under estimation of f4
    print(f"fisher MI: {gaussian_MI}")
    print(f"histogram: {histogram_MI}")


    
# %%
fig, ax = plt.subplots(1, 1)
s = 2
x = np.linspace(0,
                100, 100)
ax.plot(x, lognorm.pdf(x, s, loc=0, scale=50),
       'r-', lw=5, alpha=0.6, label='lognorm pdf')
ax.plot(x, lognorm.pdf(100-x, s, loc=0, scale=50),
       'r-', lw=5, alpha=0.6, label='lognorm pdf2')
plt.show()
# %%
fig, ax = plt.subplots(1, 1)
s = 50
x = np.linspace(0,
                500, 100)
ax.plot(x, skewnorm.pdf(x, loc = 50, scale=50, a = 3),
       'r-', lw=5, alpha=0.6, label='lognorm pdf')
ax.plot(x, skewnorm.pdf(x, loc = 50, scale=50, a = -3),
       'r-', lw=5, alpha=0.6, label='lognorm pdf2')
# ax.plot(x, skewnorm.pdf(500-x, scale=50, a = 0),
#        'r-', lw=5, alpha=0.6, label='lognorm pdf2')
plt.show()
# %%
