import pickle
import json
import pathlib
from math import log, floor, ceil

import matplotlib.pyplot as plt
from scipy.special import softmax
import numpy as np
import numpy.ma as ma


def get_data(path, name):
    with open(str(path / f"{name}_stats.pickle"), 'rb') as f:
        stats = pickle.load(f)
    with open(str(path / f"{name}_sims.pickle"), 'rb') as f:
        sims = pickle.load(f)
    with open(str(path / f"{name}_concepts.pickle"), 'rb') as f:
        real_concepts = pickle.load(f)
    with open(str(path / f"{name}_length.pickle"), 'rb') as f:
        length = pickle.load(f)

    return stats, sims, real_concepts, length

def turningpoints(lst):
    dx = np.diff(lst)
    return np.sum(dx[1:] * dx[:-1] < 0)

def turning_points(array):
    ''' turning_points(array) -> min_indices, max_indices
    Finds the turning points within an 1D array and returns the indices of the minimum and 
    maximum turning points in two separate lists.
    '''
    idx_max, idx_min = [], []
    if (len(array) < 3): 
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)
    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING: 
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
    return idx_min, idx_max

def run_eval(stats, sims, real_concepts, length, name, data_type = "real", data_source = "Real", path = None, show = True, graph=True, window_size = 100, only_sim = False):
    output_path = path if path is not None else pathlib.Path.cwd() / "output" / name

    concepts = list(sims.keys())
    # sims[concept_name][timestep_index] = ({dict of similarity types: similarity}, timestep)
    # So this is iterating through the similarity types
    losses = {}
    for v in sims[concepts[0]][0][0]:
        
        # print(v)
        concept_similarities = []
        for concept in concepts:
            # Mapping to a list of (timestep, similarity values) for the given concept and similarity type
            sim_to_concept = [(x[1], x[0][v]) for x in sims[concept] if not np.isnan(x[0][v])]
            concept_similarities.append((sim_to_concept, concept))
        
        # Getting a list of softmax values, where the softmax is done at each timestep
        softmax_values = []
        softmax_values_by_concept = {}
        print(concept_similarities)
        
        timesteps = set()
        for concept in concept_similarities:
            for ts, val in concept[0]:
                timesteps.add(ts)
        timesteps = sorted(list(timesteps))
        print(timesteps)
        for timestep in timesteps:
            concept_sims_at_timestep = []
            for concept in concept_similarities:
                for ts, val in concept[0]:
                    if ts == timestep:
                        concept_sims_at_timestep.append((val, concept[1], timestep))
            # concept_sims_at_timestep = [(concept[0][timestep_index][1], concept[1], concept[0][timestep_index][0]) for concept in concept_similarities]
            # softmax_for_ts = softmax([x[0] for x in concept_sims_at_timestep])
            softmax_for_ts = [x[0] for x in concept_sims_at_timestep]
            softmax_values += [*softmax_for_ts]
            
            for i, concept_sim in enumerate(concept_sims_at_timestep):
                sim_val, concept, timestep = concept_sim
                if concept not in softmax_values_by_concept:
                    softmax_values_by_concept[concept] = []
                softmax_values_by_concept[concept].append((timestep, softmax_for_ts[i]))
                if v == 'weighted_u_cosine' and timestep == 26010:
                    print(softmax_for_ts[i])


        
        
        max_val = max(softmax_values)
        min_val = min(softmax_values)
        val_range = max_val - min_val

        loss = []
        for timestep in timesteps:
            ts_loss = 0
            # timestep = concept_similarities[0][0][timestep_index][0]
            for concept in softmax_values_by_concept:
                value = None
                for ts, val in softmax_values_by_concept[concept]:
                    if ts == timestep:
                        value = val
                if value is None:
                     continue
                for real_concept in real_concepts:
                    if real_concept[0] <= timestep <= real_concept[1]:
                        true_concept = real_concept[3]
                        break
                
                real_val = 0 if concept == true_concept else 1
                similarity_val = (value - min_val) / val_range
                # difference = real_val - similarity_val
                # difference_square = difference * difference
                # l = difference_square
                # print(similarity_val)
                if similarity_val <= 0 or np.isnan(similarity_val):
                    l = real_val
                elif similarity_val == 1:
                    l = 1 - real_val
                else:
                    try:
                        l = -log(similarity_val) if real_val == 1 else -log(1 - similarity_val)
                    except Exception as e:
                        print(similarity_val)
                        raise e

                # print(f"{true_concept} : {concept}, {similarity_val} -> {l}")
                # difference_square = abs(difference)
                multiplier = len(concepts) if real_val == 0 else 1
                # multiplier = 1 if real_val == 0 else 0
                ts_loss += l * multiplier
            # exit()
                
            loss.append((ts_loss / (len(concepts) * 2), timestep))
            # loss.append((ts_loss, timestep))
        avg_loss = np.mean([x[0] for x in loss])
        losses[v] = avg_loss
        # print(f"Avg Loss: {avg_loss}")

        # for sim, c_name in concept_similarities:
        #     softmax_values += [*softmax([x[1] for x in sim])]

        # for sim, c_name in concept_similarities:
        #     shift = length - len(sim)
        #     plt.plot([x[0] for x in sim], softmax([x[1] for x in sim]), label = c_name, linewidth = 1)

        if graph:
            plt.figure(figsize=(20,20))
            for concept in softmax_values_by_concept:
                similarities = softmax_values_by_concept[concept]
                shift = length - len(similarities)
                # plt.plot([x[0] for x in similarities], [(max_val - x[1])/val_range for x in similarities], label = concept, linewidth = 1)
                plt.plot([x[0] for x in similarities], [x[1] for x in similarities], label = concept, linewidth = 1)

            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()

            for concept in real_concepts:
                # print(labels)
                # print(concepts)
                if concept[3] in labels:
                    color = handles[labels.index(concept[3])].get_color()
                    plt.hlines(y = -0.01, xmin = concept[0], xmax = concept[1], colors = [color])
                else:
                    color_index = list(set([c[3] for c in real_concepts])).index(concept[3])
                    plt.hlines(y = -0.01, xmin = concept[0], xmax = concept[1], colors = "C{}".format(color_index))

            # plt.ylabel("Distance to concept (Lower is closer)")
            plt.ylabel("Similarity to concept (Higher is closer)")
            plt.xlabel("Observation")
            plt.legend()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(v)
            plt.savefig(str(output_path / f"{name}_{v}.pdf"), figsize = (40, 40), bbox_inches = 'tight')
            plt.close('all')

            

            plt.figure(figsize=(20,20))

            plt.plot([x[1] for x in loss], [x[0] for x in loss])
            plt.savefig(str(output_path / f"{name}_{v}_loss.pdf"), figsize = (40, 40), bbox_inches = 'tight')
            plt.close('all')

            if show:
                plt.show()
            else:
                plt.close('all')
    if only_sim:
        return None, None
    data_source = list(stats[0][0].keys())
    features = list(stats[0][0][data_source[0]].keys())
    print(data_source)
    print(features)
    by_feature = {}
    for source in data_source:
        by_feature[source] = {}
        for f in features:
            by_feature[source][f] = []
    for ts in stats:
        for source in data_source:
            for f in features:
                by_feature[source][f].append(ts[0][source][f])
    #Main Image
    masked_standard_deviations = {}
    masked_standard_deviations_by_source = {}
    for source in data_source:
        plt.figure(figsize=(20,20))
        fig, ax = plt.subplots(len(features), 1, figsize=(20,20))
        masked_standard_deviations_by_source[source] = {}
        for i,f in enumerate(features):
            points = [x[0] for x in concept_similarities[0][0]]
            x_vals = np.array(points)

            drift_points = []
            s = 0
            for concept in real_concepts:
                ax[i].axvline(x = concept[1], color = 'red')
                e = concept[1]
                
                greater_than_start = x_vals >= (s + window_size)
                less_than_end = x_vals < e
                points_in_concept = np.logical_and(greater_than_start, less_than_end)
                masked_y_in_concept = np.array(by_feature[source][f])[points_in_concept]
                avg_point = np.mean(masked_y_in_concept)
                # in_concept  = [(x, y) for x,y in zip(points, by_feature[source][f]) if s <= x < e]
                # avg_point = np.mean([y for x,y in in_concept])
                ax[i].hlines(avg_point, s, e, alpha = 0.2)
                ax[i].hlines(avg_point, 0, s, linestyles = 'dotted', alpha = 0.2)
                ax[i].hlines(avg_point, e, points[-1], linestyles = 'dotted', alpha = 0.2)
                drift_points.append((e, avg_point))
                s = e
            def close_to_drift(x, y):
                for dp,avg in drift_points:
                    if 0 < x-dp < window_size:
                        # return avg
                        return y
                return y

            def close_to_drift(x, y):
                for dp,avg in drift_points:
                    if 0 < x-dp < window_size:
                        # return avg
                        return True
                return False
            
            def mask_close_to_drift(X, Y):
                mask = []
                for x,y in zip(X, Y):
                    mask.append(1 if close_to_drift(x, y) else 0)
                return ma.masked_array(X, mask=mask), ma.masked_array(Y, mask=mask)
            masked_X, masked_Y = mask_close_to_drift(points, by_feature[source][f])
            masked_standard_deviations[f"{source}-{f}"] = np.std(masked_Y)
            # masked_standard_deviations[f"{source}-{f}"] = 0
            masked_standard_deviations_by_source[source][f] = np.std(masked_Y)
            # masked_standard_deviations_by_source[source][f] = 0

            sorted_y_values = sorted(by_feature[source][f])
            min_y = sorted_y_values[round(len(sorted_y_values) * 0.05)]
            max_y = sorted_y_values[round(len(sorted_y_values) * 0.95)]
            y_width = max_y - min_y
            min_y -= y_width / 2
            max_y += y_width / 2

            # ax[i].plot(points, [close_to_drift(x, y) for x,y in zip(points, by_feature[source][f])])
            ax[i].plot(masked_X, masked_Y)
            
            # if f == 'kurtosis':
            ax[i].set_ylim(min_y, max_y)
            ax[i].set_title(f"{f}")
        fig.suptitle(f"{source}")
        fig.tight_layout()
        fig.subplots_adjust(hspace = 2)
        plt.savefig(str(output_path / f"{name}_{source}.pdf"), figsize = (80, 80), bbox_inches = 'tight')
        plt.close('all')
    def moving_average(a, n=3) :
        print(a)
        a = np.array(a)
        mask = np.isnan(a)
        print(mask)
        print(np.sum(mask))
        m = np.flatnonzero(mask)
        nm = np.flatnonzero(~mask)
        print(m)
        print(n)
        a[mask] = np.interp(m, nm, a[~mask])
        ret = np.cumsum(a, dtype=float)
        print(ret)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
        # return ret / n
    #Smoothed Image
    standard_deviations = {}
    standard_deviations_by_source = {}
    for source in data_source:
        plt.figure(figsize=(20,20))
        fig, ax = plt.subplots(len(features), 1, figsize=(20,20))
        standard_deviations_by_source[source] = {}
        for i,f in enumerate(features):
            points = [x[0] for x in concept_similarities[0][0]]
            drift_points = []
            s = 0
            # tp = max(int(turningpoints(by_feature[source][f])), 1)
            tp = max(int(sum(len(tp_type) for tp_type in turning_points(by_feature[source][f]))), 1)
            len_tp = len(by_feature[source][f])
            tp_rate = tp / len_tp
            smoothed_points = moving_average(by_feature[source][f], 200)
            # smoothed_tp = max(int(turningpoints(smoothed_points)), 1)
            smoothed_tp = max(int(sum(len(tp_type) for tp_type in turning_points(smoothed_points))), 1)
            len_smoothed_tp = len(smoothed_points)
            smoothed_tp_rate = smoothed_tp / len_smoothed_tp

            # normalized_points = (smoothed_points - smoothed_points.min()) / (smoothed_points.max() - smoothed_points.min())
            standard_deviations[f"{source}-{f}"] = np.std(smoothed_points)
            standard_deviations[f"{source}-{f}-smoothed_deviation"] = np.sum(np.power((smoothed_points - np.mean(smoothed_points)), 2))
            standard_deviations[f"{source}-{f}-smoothed_deviation_per_tp"] = standard_deviations[f"{source}-{f}-smoothed_deviation"] / smoothed_tp
            standard_deviations[f"{source}-{f}-tp"] = tp
            standard_deviations[f"{source}-{f}-tp_rate"] = tp_rate
            standard_deviations[f"{source}-{f}-len_tp"] = len_tp
            standard_deviations[f"{source}-{f}-smoothed_tp"] = smoothed_tp
            standard_deviations[f"{source}-{f}-smoothed_tp_rate"] = smoothed_tp_rate
            standard_deviations[f"{source}-{f}-len_smoothed_tp"] = len_smoothed_tp
            standard_deviations_by_source[source][f] = np.std(smoothed_points)
            points = points[-len(smoothed_points):]
            # for concept in real_concepts:
            #     ax[i].axvline(x = concept[1], color = 'red')
            #     e = concept[1]
            #     in_concept  = [(x, y) for x,y in zip(points, smoothed_points) if s <= x < e]
            #     avg_point = np.mean([y for x,y in in_concept])
            #     ax[i].hlines(avg_point, s, e)
            #     ax[i].hlines(avg_point, 0, s, linestyles = 'dotted')
            #     ax[i].hlines(avg_point, e, points[-1], linestyles = 'dotted')
            #     drift_points.append((e, avg_point))
            #     s = e

            sorted_y_values = sorted(smoothed_points)
            print(sorted_y_values)
            min_y = sorted_y_values[floor(len(sorted_y_values) * 0.05)]
            print(min_y)
            max_y = sorted_y_values[ceil((len(sorted_y_values) - 1) * 0.95)]
            print(max_y)
            y_width = max_y - min_y
            print(y_width)
            min_y -= y_width / 2
            print(min_y)
            max_y += y_width / 2
            print(max_y)
            min_y = floor(min_y*4)/4
            max_y = ceil(max_y*4)/4

            ax[i].plot(points, [y for x,y in zip(points, smoothed_points)])
            # if f == 'kurtosis':
            ax[i].set_ylim(min_y, max_y)
            ax[i].set_title(f"{f}")
        fig.suptitle(f"{source}")
        fig.tight_layout()
        fig.subplots_adjust(hspace = 2)
        plt.savefig(str(output_path / f"{name}_{source}_smooth.pdf"), figsize = (80, 80), bbox_inches = 'tight')
        plt.close('all')
        # plt.show()

    overall_avg_loss = np.mean([v for v in losses.values()])

    standard_deviations['overall_average'] = sum(standard_deviations.values()) / len(standard_deviations.values())
    sd_by_feature = {}
    masked_sd_by_feature = {}
    sd_by_feature_tp_ratio = {}
    sd_by_feature_dev_tp_ratio = {}
    for source in data_source:
        avg_sum = []
        avg_sum_tp_rate = []
        avg_dev_tp_rate = []
        masked_avg_sum = []
        for i,f in enumerate(features):
            avg_sum.append(standard_deviations_by_source[source][f])
            masked_avg_sum.append(masked_standard_deviations_by_source[source][f])
            avg_sum_tp_rate.append(standard_deviations[f"{source}-{f}"] / standard_deviations[f"{source}-{f}-smoothed_tp_rate"])
            avg_dev_tp_rate.append(standard_deviations[f"{source}-{f}-smoothed_deviation_per_tp"])
            if f not in sd_by_feature:
                sd_by_feature[f] = []
                masked_sd_by_feature[f] = []
                sd_by_feature_tp_ratio[f] = []
                sd_by_feature_dev_tp_ratio[f] = []
            sd_by_feature[f].append(standard_deviations_by_source[source][f])
            masked_sd_by_feature[f].append(masked_standard_deviations_by_source[source][f])
            sd_by_feature_tp_ratio[f].append(standard_deviations[f"{source}-{f}"] / standard_deviations[f"{source}-{f}-smoothed_tp_rate"])
            sd_by_feature_dev_tp_ratio[f].append(standard_deviations[f"{source}-{f}-smoothed_deviation_per_tp"])
            
        standard_deviations[f"{source}_average"] = sum(avg_sum) / len(avg_sum)
        standard_deviations[f"masked_{source}_average"] = sum(masked_avg_sum) / len(masked_avg_sum)
        standard_deviations[f"{source}_tp_ratio_average"] = sum(avg_sum_tp_rate) / len(avg_sum_tp_rate)
        standard_deviations[f"{source}_smoothed_deviation_per_tp"] = sum(avg_dev_tp_rate) / len(avg_dev_tp_rate)
    for f in sd_by_feature:
        standard_deviations[f"{f}_average"] = sum(sd_by_feature[f]) / len(sd_by_feature[f])
        standard_deviations[f"masked_{f}_average"] = sum(masked_sd_by_feature[f]) / len(masked_sd_by_feature[f])
        standard_deviations[f"{f}_tp_ratio_average"] = sum(sd_by_feature_tp_ratio[f]) / len(sd_by_feature_tp_ratio[f])
        standard_deviations[f"{f}_smoothed_deviation_per_tp"] = sum(sd_by_feature_dev_tp_ratio[f]) / len(sd_by_feature_dev_tp_ratio[f])
        print(f)
        print(sd_by_feature_dev_tp_ratio[f])
        print(standard_deviations[f"{f}_smoothed_deviation_per_tp"])

    with (output_path / f"{name}_std.txt").open('w') as f:
        json.dump(standard_deviations, f)
    return losses, overall_avg_loss


if __name__ == "__main__":
    name = "cmc"
    data_type = "real"
    data_source = "Real"
    # name = "stagger"
    # data_type = "synthetic"
    # data_source = "STAGGER"


    output_path = pathlib.Path.cwd() / "output" / name
    stats, sims, real_concepts, length = get_data(output_path, name)
    run_eval(stats, sims, real_concepts, length, name)

    # concepts = list(sims.keys())
    # plt.figure(figsize=(20,20))
    # # sims[concept_name][timestep_index] = ({dict of similarity types: similarity}, timestep)
    # # So this is iterating through the similarity types
    # for v in sims[concepts[0]][0][0]:
    #     print(v)
    #     concept_similarities = []
    #     for concept in concepts:
    #         # Mapping to a list of (timestep, similarity values) for the given concept and similarity type
    #         sim_to_concept = [(x[1], x[0][v]) for x in sims[concept] if not np.isnan(x[0][v])]
    #         concept_similarities.append((sim_to_concept, concept))
        
    #     # Getting a list of softmax values, where the softmax is done at each timestep
    #     softmax_values = []
    #     softmax_values_by_concept = {}
        
    #     for timestep_index in range(len(concept_similarities[0][0])):
    #         concept_sims_at_timestep = [(concept[0][timestep_index][1], concept[1], concept[0][timestep_index][0]) for concept in concept_similarities]
    #         softmax_for_ts = softmax([x[0] for x in concept_sims_at_timestep])
    #         softmax_values += [*softmax_for_ts]
            
    #         for i, concept_sim in enumerate(concept_sims_at_timestep):
    #             sim_val, concept, timestep = concept_sim
    #             if concept not in softmax_values_by_concept:
    #                 softmax_values_by_concept[concept] = []
    #             softmax_values_by_concept[concept].append((timestep, softmax_for_ts[i]))

        
        
    #     max_val = max(softmax_values)
    #     min_val = min(softmax_values)
    #     val_range = max_val - min_val

    #     loss = []
    #     for timestep_index in range(len(concept_similarities[0][0])):
    #         ts_loss = 0
    #         timestep = concept_similarities[0][0][timestep_index][0]
    #         for concept in softmax_values_by_concept:
    #             for real_concept in real_concepts:
    #                 if real_concept[0] <= timestep <= real_concept[1]:
    #                     true_concept = real_concept[3]
    #                     break
                
    #             real_val = 0 if concept == true_concept else 1
    #             similarity_val = (softmax_values_by_concept[concept][timestep_index][1] - min_val) / val_range
    #             difference = real_val - similarity_val
    #             difference_square = difference * difference
    #             # difference_square = abs(difference)
    #             multiplier = len(concepts) if real_val == 0 else 1
    #             # multiplier = 1 if real_val == 0 else 0
    #             ts_loss += difference_square * multiplier
                
    #         loss.append((ts_loss / (len(concepts) * 2), timestep))
    #         # loss.append((ts_loss, timestep))

    #     print(f"Avg Loss: {np.mean([x[0] for x in loss])}")

    #     # for sim, c_name in concept_similarities:
    #     #     softmax_values += [*softmax([x[1] for x in sim])]

    #     # for sim, c_name in concept_similarities:
    #     #     shift = length - len(sim)
    #     #     plt.plot([x[0] for x in sim], softmax([x[1] for x in sim]), label = c_name, linewidth = 1)
    #     for concept in softmax_values_by_concept:
    #         similarities = softmax_values_by_concept[concept]
    #         shift = length - len(similarities)
    #         plt.plot([x[0] for x in similarities], [(max_val - x[1])/val_range for x in similarities], label = concept, linewidth = 1)

    #     ax = plt.gca()
    #     handles, labels = ax.get_legend_handles_labels()

    #     for concept in real_concepts:
    #         color = handles[labels.index(concept[3])].get_color()
    #         plt.hlines(y = -0.01, xmin = concept[0], xmax = concept[1], colors = [color])

    #     plt.ylabel("Distance to concept (Lower is closer)")
    #     plt.xlabel("Observation")
    #     plt.legend()
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    #     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #     plt.title(v)
    #     plt.savefig(str(output_path / f"{name}_{v}.pdf"), figsize = (40, 40), bbox_inches = 'tight')

    #     # data_source = list(stats[0][0].keys())
    #     # features = list(stats[0][0][data_source[0]].keys())
    #     # print(data_source)
    #     # print(features)
    #     # by_feature = {}
    #     # for source in data_source:
    #     #     by_feature[source] = {}
    #     #     for f in features:
    #     #         by_feature[source][f] = []
    #     # for ts in stats:
    #     #     for source in data_source:
    #     #         for f in features:
    #     #             by_feature[source][f].append(ts[0][source][f])
    #     # for source in data_source:
    #     #     fig, ax = plt.subplots(len(features), 1)
    #     #     for i,f in enumerate(features):
    #     #         ax[i].plot([x[0] for x in concept_similarities[0][0]], by_feature[source][f])
    #     #         for concept in real_concepts:
    #     #             ax[i].axvline(x = concept[1], color = 'red')
    #     #         ax[i].set_title(f"{f}")
    #     #     fig.suptitle(f"{source}")
    #     #     fig.tight_layout()
    #     #     fig.subplots_adjust(hspace = 2)
    #     #     plt.savefig(str(output_path / f"{name}_{source}.pdf"), figsize = (80, 80), bbox_inches = 'tight')
    #     #     plt.show()

    #     plt.figure()

    #     plt.plot([x[1] for x in loss], [x[0] for x in loss])
    #     plt.savefig(str(output_path / f"{name}_{v}_loss.pdf"), figsize = (40, 40), bbox_inches = 'tight')
    #     plt.show()



# xmin = 0
# true_concepts = []
# for i,seg in enumerate(real_concepts):
#     if i == 0:
#         continue
#     xmax = seg[0]
#     true_concepts.append([real_concepts[i-1][1], xmin, xmax])
#     xmin = xmax
# true_concepts.append([real_concepts[-1][1], xmin, length])