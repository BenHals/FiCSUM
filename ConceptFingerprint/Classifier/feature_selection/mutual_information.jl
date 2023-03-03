function histogram_entropy(histogram, bins, hist_total::Float64)
    entropy = 0.0
    for (bin_count, (bin_start, bin_end)) in zip(histogram, bins)
        if bin_start == bin_end
            continue
        end
        bin_probability = bin_count / hist_total
        if bin_probability <= 0.0
            bin_probability = 1 / hist_total
        end
        entropy -= bin_probability * log(bin_probability / (bin_end - bin_start))
    end
    return entropy
end

# def histogram_entropy(histogram, bins, hist_total):
#     entropy = 0.0
#     for bin_count, (bin_start, bin_end) in zip(histogram, bins):
#         if bin_start == bin_end:
#             continue
#         bin_probability = bin_count / hist_total
#         if bin_probability <= 0.0:
#             bin_probability = 1 / hist_total
#         entropy -= bin_probability * np.log2(bin_probability / (bin_end - bin_start))
#     return entropy

function get_bin(value::Float64, bins::Array{Tuple{Float64, Float64}})
    # Julia starts indexing at 1
    bin_i = 1
    if value < bins[1][1]
        return bin_i
    end
    for (bin_start, bin_end) in bins
        if bin_start <= value < bin_end
            return bin_i
        end
        bin_i += 1
    end
    return length(bins)
end

# def get_bin(value, bins):
#     for bin_i, (bin_start, bin_end) in enumerate(bins):
#         if bin_start <= value < bin_end:
#             return bin_i
#     return len(bins) - 1

function merge_histograms(histograms::Array{Float64, 2}, bins::Array{Tuple{Float64, Float64}, 2})
    values = Dict()
    min_val = typemax(Float64)
    max_val = typemin(Float64)
    num_bins = size(bins)[2]
    total_count = 0
    for (histogram, binning) in zip(eachrow(histograms), eachrow(bins))
        for (count, (bin_start, bin_end)) in zip(histogram, binning)
            val = (bin_start + bin_end) / 2
            values[val] = get(values, val, 0) + count
            total_count += count
            min_val = min(val, min_val)
            max_val = max(val, max_val)
        end
    end
    new_bins = Tuple{Float64, Float64}[]
    total_width = max_val - min_val
    bin_width = total_width / num_bins
    for b_i in 1:num_bins
        i = min_val + (b_i - 1) * bin_width
        push!(new_bins, (i, i+bin_width))
    end
    new_histogram = [0.0 for i in 1:num_bins]
    for (val, c) in values
        new_histogram[get_bin(val, new_bins)] += c
    end
    return new_histogram, new_bins, total_count
end
# function merge_histograms(histograms::Array{Float64, 2}, bins::Array{Tuple{Float64, Float64}, 2})
#     return histograms[1, :], bins[1, :], 100.0
# end
# def merge_histograms(histograms, bins):
#     values = {}
#     min_val = None
#     max_val = None
#     num_bins = len(bins[0])
#     total_count = 0
#     for histogram, binning in zip(histograms, bins):
#         for count, (bin_start, bin_end) in zip(histogram, binning):
#             val = (bin_start + bin_end) / 2
#             if val not in values:
#                 values[val] = 0
#             values[val] += count
#             total_count += count
#             if min_val is None or val < min_val:
#                 min_val = val
#             if max_val is None or val > max_val:
#                 max_val = val
#     new_bins = []
#     total_width = max_val - min_val
#     bin_width = total_width / num_bins
#     for b_i in range(num_bins):
#         i = min_val + b_i * bin_width
#         new_bins.append((i, i+bin_width))
    
#     new_histogram = [0 for i in range(num_bins)]
#     for val in values:
#         new_histogram[get_bin(val, new_bins)] += values[val]
#     return new_histogram, new_bins, total_count

    
function MI_histogram_estimation(overall_histogram::Array{Float64, 1}, overall_bins::Array{Tuple{Float64, Float64}, 1}, overall_count::Float64, 
    concept_histograms::Array{Float64, 2}, concept_bins::Array{Tuple{Float64, Float64}, 2}, concept_counts::Array{Float64, 1})
    merged_histogram, merged_bins, merged_count = merge_histograms(concept_histograms, concept_bins)
    h_x = histogram_entropy(merged_histogram, merged_bins, merged_count)
    h_x_given_concept = 0.0
    total_weight = 0.0
    for (c_histogram, c_bin, c_weight) in zip(eachrow(concept_histograms), eachrow(concept_bins), concept_counts)
        total_weight += c_weight
        concept_entropy = histogram_entropy(c_histogram, c_bin, c_weight) * c_weight
        h_x_given_concept += concept_entropy
    end
    h_x_given_concept /= total_weight

    return max(h_x - h_x_given_concept, 0)
end

function MI_histogram_estimation(overall_histogram::Array{Int64, 1}, overall_bins::Array{Tuple{Float64, Float64}, 1}, overall_count::Int64, 
    concept_histograms::Array{Int64, 2}, concept_bins::Array{Tuple{Float64, Float64}, 2}, concept_counts::Array{Int64, 1})
    return MI_histogram_estimation(convert(Array{Float64}, overall_histogram), overall_bins, convert(Float64, overall_count),
        convert(Array{Float64, 2}, concept_histograms), concept_bins, convert(Array{Float64}, concept_counts))
end

# def MI_histogram_estimation(overall_histogram, overall_bins, overall_count, concept_histograms, concept_bins, concept_counts):
#     """ fingerprints in normalizer (overall) and each concept
#     store a histogram of MI feature values. 
#     We calculate entropy from each histogram.
#     """
#     np.seterr('raise')
#     # overall_h_x = histogram_entropy(overall_histogram, overall_bins, overall_count)
#     merged_histogram, merged_bins, merged_count = merge_histograms(concept_histograms, concept_bins)
#     h_x = histogram_entropy(merged_histogram, merged_bins, merged_count)
#     h_x_given_concept = 0.0
#     total_weight = 0.0
#     for c_histogram, c_bin, c_weight in zip(concept_histograms, concept_bins, concept_counts):
#         total_weight += c_weight
#         concept_entropy = histogram_entropy(c_histogram, c_bin, c_weight) * c_weight
#         h_x_given_concept += concept_entropy
#     h_x_given_concept /= total_weight

#     return max(h_x - h_x_given_concept, 0)