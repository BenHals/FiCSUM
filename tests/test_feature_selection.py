from ConceptFingerprint.Classifier.feature_selection.fisher_score import fisher_score
import numpy as np

# from skfeature.function.similarity_based.fisher_score import fisher_score as fs2

def test_fisher_score():
    X = [
        [1, 1.1, 1],
        [2, 2.1, 2],
        [3, 3.1, 3],
        [4, 4.1, 4]
    ]
    X_obs = np.array([
        [1, 1],
        [1.1, 1.1],
        [1, 1],
        [2, 1.1],
        [2.1, 1],
        [2, 1],
        [3, 1],
        [3.1, 1.1],
        [3, 1],
        [4, 1.1],
        [4.1, 1],
        [4, 1],
    ])

    y = [1, 2, 3, 4]
    y_obs = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    group_means = np.array([np.mean(x) for x in X])
    group_stdev = np.array([np.std(x) for x in X])
    group_counts = np.array([len(x) for x in X])
    overall_standard_deviation = np.std([x for group in X for x in group])
    # Fisher score should be high if there is a high difference between groups compared to
    # overall standard deviation

    high_correlation_A = fisher_score(group_means, group_counts, overall_standard_deviation, group_stdev)
    assert high_correlation_A > 0.9
    X = [
        [1, 2, 3],
        [11, 12, 13],
        [21, 22, 23],
        [31, 32, 33]
    ]

    y = [1, 2, 3, 4]
    group_means = np.array([np.mean(x) for x in X])
    group_counts = np.array([len(x) for x in X])
    overall_standard_deviation = np.std([x for group in X for x in group])
    # Fisher score should be high if there is a high difference between groups compared to
    # overall standard deviation
    high_correlation_B = fisher_score(group_means, group_counts, overall_standard_deviation)
    assert high_correlation_B > 0.9


    X = [
        [1, 1.1, 1],
        [1.1, 1, 1],
        [1, 1.1, 1],
        [1.1, 1, 1]
    ]
    group_means = np.array([np.mean(x) for x in X])
    group_counts = np.array([len(x) for x in X])
    overall_standard_deviation = np.std([x for group in X for x in group])
    # Fisher score should be low if there is a low difference between groups compared to
    # overall standard deviation
    no_correlation_A = fisher_score(group_means, group_counts, overall_standard_deviation)
    assert no_correlation_A < 0.1

    X = [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
    ]
    group_means = np.array([np.mean(x) for x in X])
    group_counts = np.array([len(x) for x in X])
    overall_standard_deviation = np.std([x for group in X for x in group])
    # Fisher score should be low if there is a low difference between groups compared to
    # overall standard deviation
    no_correlation_B = fisher_score(group_means, group_counts, overall_standard_deviation)
    assert no_correlation_B < 0.1
    X = [
        [1, 2, 3],
        [1.1, 2.1, 3.1],
        [1.2, 2.2, 3.2],
        [1.3, 2.3, 3.3]
    ]
    group_means = np.array([np.mean(x) for x in X])
    group_counts = np.array([len(x) for x in X])
    overall_standard_deviation = np.std([x for group in X for x in group])
    # Fisher score should be low if there is a low difference between groups compared to
    # overall standard deviation
    no_correlation_C = fisher_score(group_means, group_counts, overall_standard_deviation)
    assert no_correlation_C < 0.1
    X = [
        [3], 
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1],
    ]
    group_means = np.array([np.mean(x) for x in X])
    group_counts = np.array([len(x) for x in X])
    overall_standard_deviation = np.std([x for group in X for x in group])
    # Fisher score should take into acount group size, deviation in a small group counts less.
    no_correlation_D_small = fisher_score(group_means, group_counts, overall_standard_deviation)
    X = [
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1],
    ]
    group_means = np.array([np.mean(x) for x in X])
    group_counts = np.array([len(x) for x in X])
    overall_standard_deviation = np.std([x for group in X for x in group])
    # Fisher score should take into acount group size, deviation in a small group counts less.
    no_correlation_D_large = fisher_score(group_means, group_counts, overall_standard_deviation)
    assert no_correlation_D_small < no_correlation_D_large

    print(X_obs.shape)
    print(y_obs.shape)
    # print(fs2(X_obs, y_obs))
    print([high_correlation_A, no_correlation_A])

# test_fisher_score()