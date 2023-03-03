#%%
import pathlib
import pickle

#%%
p = pathlib.Path(__file__).parents[2] / 'RawData/Synthetic/STAGGERS/seeds/1' / 'concept_0.pickle'
p2 = pathlib.Path('S:/') / 'PhD/results' / 'concept_1.pickle'
#%%
with open(str(p), 'rb') as f:
    stream = pickle.load(f)
with open(str(p2), 'rb') as f:
    stream2 = pickle.load(f)

#%%
print(stream.next_sample())
print(stream2.next_sample())