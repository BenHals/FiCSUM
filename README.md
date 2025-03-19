[![Build Status](https://dev.azure.com/benhalstd/ConceptFingerprint/_apis/build/status/BenHals.ConceptFingerprint?branchName=master)](https://dev.azure.com/benhalstd/ConceptFingerprint/_build/latest?definitionId=1&branchName=master)
# FiCSUM
A data stream framework to make predictions in non-stationary conditions.
FiCSUM uses a combination of many "meta-information" features in order to detect change in many aspects of a data stream, both supervised (change in relationship to labels) and unsupervised (change in feature space).
Individually, each meta-information feature has been shown to be able to discriminate between useful "concepts" in a data stream, or periods displaying a similar relationship to be learned. 
In real-world streams change is possible in many different aspects, and single meta-information features are not able to detect all aspects at once.
For example, looking only at feature space may miss changes in class label distribution.
FiCSUM shows that a combined approach can increase performance.

For more details, please see the full paper published in ICDE 2021 availiable [here](https://ieeexplore.ieee.org/abstract/document/9458895).

![Concept Similarity](https://github.com/BenHals/ConceptFingerprint/raw/master/ConceptSimilarity.png)
# Implementation

The basic idea of FiCSUM is to capture many aspects of data stream behaviour in a vector, which we call a 'concept fingerprint'.
This fingerprint represents an overall set of behaviours seen over a window of observations. 
Vector similarity measures can be used to compare the fingerprints at different points in the stream.
This allows change to be detected, as a significant difference in similarity, and reccurences to be identified, as a resurgence in similarity to a previously seen fingerprint.
This allows change to be adapted to by saving an individual model to be used along side each fingerprint, and reused when it best matches the stream.
This also allows a concept history of the stream to be built, representing similar periods in the stream. A concept history can be mapped to past behaviour and even environmental conditions to contextualize future recurrence. For example, if a previous fingerprint was associated with medium accuracy and always occured alongside stormy conditions, a future recurrence to this fingerprint can be expected to also bring medium accuracy and stormy conditions.

## Instructions to Run
0. Install
- numpy
- scikit-multiflow
- tqdm
- psutil
- statsmodels
- pyinstrument
1. meta-information dependencies
- [shap](https://github.com/slundberg/shap) version 0.35 (version is required as we use a patched file based on this version. Will require the patched code to be updated for a newer version.)
- Enable shap to work with Scikit-Multiflow by replacing the shap tree.py file with the patched `tree.py` found in `ConceptFIngerprint\Exploration`. (Or just use the section defining the translation from )
- [PyEMD](https://github.com/laszukdawid/PyEMD)
- [EntroPy](https://raphaelvallat.com/entropy/build/html/index.html)
- Can run pip -install -r requirements.txt
2. Create a data directory. This is expected to contain `Real` and `Synthetic` subdirectories. These should each contain a directory for each data set to be tested on. The expected format for these is a `.csv` file for each ground truth context. The system will work with only a single `.csv` if context is unknown, but some evaluation measures will not be able to be calculated. For synthetic datasets created with a known generator, an empty directory in the `Synthetic` directory is needed to store files. Each dataset folder should be named the name you will use to call it. The base data directory should be passed as the `--datalocation` commandline argument. The dataset name is passed in the `--datasets` argument. New datasets will need to be added to the relevent list of allowed datasets in `run_experiment.py`, `synthetic_MI_datasets` for `Synthetic` datasets, or `real_drift_datasets` for `Real` datasets.
3. Set commandline arguments. Most are set to reasonable defaults. 
- Set seed using the `--seeds` command. This should be set to the number of runs if `--seedaction` is set to `new` (will create new random seeds) or `reuse` (will reuse previously used seeds). Or should be a list if `--seedaction` is `list`, e.g. `--seedaction list --seeds 1 2 3` will run 3 runs using seeds 1, 2 and 3.
- Set multiprocessing options. `--single` can be set to turn off multiprocessing for simpler output and ease of cancelation. Or `--cpus` can set the desired number of cpu cores to run on.
- Set meta-information functions and behaviour sources to disable using `--ifeatures` and `--isources`. For Quick runs, disabling `MI` and `IMF` can improve runtime significantly.
4. Results are placed in `~\output\expDefault\[dataset_name]` by default. This can be set with `--outputlocation` and `--experimentname`.
- The `results_run_....txt` file contains the calculated performance measures of the run.
- The `run_...csv` file contains measures describing how each observation was handled.
- The `run_...._options.txt` file contains the options used for a specific run.


### Running
The main evaluation entry point is `run_experiment.py`. This script runs FiCSUM on a specified data set. 
`run_moa.py` runs the same experiment calling a moa classifier on the commandline. Used to test against baselines.
`run_other.py` runs the same experiment using alternative python-based frameworks. Used to test against baselines.

# Citation
Please cite this work as:
`Halstead, Ben, Yun Sing Koh, Patricia Riddle, Russel Pears, Mykola Pechenizkiy, and Albert Bifet. "Fingerprinting Concepts in Data Streams with
Supervised and Unsupervised Meta-Information
" International Conference on Data Engineering (ICDE) (2021)`
