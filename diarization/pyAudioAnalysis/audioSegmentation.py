from __future__ import print_function
import os
import csv
import glob
import scipy
import sklearn
import numpy as np
import hmmlearn.hmm
import sklearn.cluster
import pickle as cpickle
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sklearn.discriminant_analysis
import audioBasicIO
import audioTrainTest as at
import MidTermFeatures as mtf
import ShortTermFeatures as stf

""" General utility functions """


def train_hmm_compute_statistics(features, labels):
    """
    This function computes the statistics used to train
    an HMM joint segmentation-classification model
    using a sequence of sequential features and respective labels

    ARGUMENTS:
     - features:  a np matrix of feature vectors (numOfDimensions x n_wins)
     - labels:    a np array of class indices (n_wins x 1)
    RETURNS:
     - class_priors:            matrix of prior class probabilities
                                (n_classes x 1)
     - transmutation_matrix:    transition matrix (n_classes x n_classes)
     - means:                   means matrix (numOfDimensions x 1)
     - cov:                     deviation matrix (numOfDimensions x 1)
    """
    unique_labels = np.unique(labels)
    n_comps = len(unique_labels)

    n_feats = features.shape[0]

    if features.shape[1] < labels.shape[0]:
        print("trainHMM warning: number of short-term feature vectors "
              "must be greater or equal to the labels length!")
        labels = labels[0:features.shape[1]]

    # compute prior probabilities:
    class_priors = np.zeros((n_comps,))
    for i, u_label in enumerate(unique_labels):
        class_priors[i] = np.count_nonzero(labels == u_label)
    # normalize prior probabilities
    class_priors = class_priors / class_priors.sum()

    # compute transition matrix:
    transmutation_matrix = np.zeros((n_comps, n_comps))
    for i in range(labels.shape[0]-1):
        transmutation_matrix[int(labels[i]), int(labels[i + 1])] += 1
    # normalize rows of transition matrix:
    for i in range(n_comps):
        transmutation_matrix[i, :] /= transmutation_matrix[i, :].sum()

    means = np.zeros((n_comps, n_feats))
    for i in range(n_comps):
        means[i, :] = \
            np.array(features[:,
                     np.nonzero(labels == unique_labels[i])[0]].mean(axis=1))

    cov = np.zeros((n_comps, n_feats))
    for i in range(n_comps):
        """
        cov[i, :, :] = np.cov(features[:, np.nonzero(labels == u_labels[i])[0]])
        """
        # use line above if HMM using full gaussian distributions are to be used
        cov[i, :] = np.std(features[:,
                           np.nonzero(labels == unique_labels[i])[0]],
                           axis=1)

    return class_priors, transmutation_matrix, means, cov



def speaker_diarization(filename, n_speakers, mid_window=2.0, mid_step=0.2,
                                                                    short_window=0.05, lda_dim=35, plot_res=False):
    """
    ARGUMENTS:
        - filename:        the name of the WAV file to be analyzed
        - n_speakers       the number of speakers (clusters) in
                           the recording (<=0 for unknown)
        - mid_window (opt)    mid-term window size
        - mid_step (opt)    mid-term window step
        - short_window  (opt)    short-term window size
        - lda_dim (opt     LDA dimension (0 for no LDA)
        - plot_res         (opt)   0 for not plotting the results 1 for plotting
    """
    sampling_rate, signal = audioBasicIO.read_audio_file(filename)
    signal = audioBasicIO.stereo_to_mono(signal)
    duration = len(signal) / sampling_rate
    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "data/models")

    classifier_all, mean_all, std_all, class_names_all, _, _, _, _, _ = \
        at.load_model_knn(os.path.join(base_dir, "knn_speaker_10"))
    classifier_fm, mean_fm, std_fm, class_names_fm, _, _, _, _,  _ = \
        at.load_model_knn(os.path.join(base_dir, "knn_speaker_male_female"))

    mid_feats, st_feats, _ = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                   mid_window * sampling_rate,
                                   mid_step * sampling_rate,
                                   round(sampling_rate * short_window),
                                   round(sampling_rate * short_window * 0.5))

    mid_term_features = np.zeros((mid_feats.shape[0] + len(class_names_all) +
                                  len(class_names_fm), mid_feats.shape[1]))

    for index in range(mid_feats.shape[1]):
        feature_norm_all = (mid_feats[:, index] - mean_all) / std_all
        feature_norm_fm = (mid_feats[:, index] - mean_fm) / std_fm
        _, p1 = at.classifier_wrapper(classifier_all, "knn", feature_norm_all)
        _, p2 = at.classifier_wrapper(classifier_fm, "knn", feature_norm_fm)
        start = mid_feats.shape[0]
        end = mid_feats.shape[0] + len(class_names_all)
        mid_term_features[0:mid_feats.shape[0], index] = mid_feats[:, index]
        mid_term_features[start:end, index] = p1 + 1e-4
        mid_term_features[end::, index] = p2 + 1e-4

    mid_feats = mid_term_features    # TODO
    feature_selected = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41,
                        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

    mid_feats = mid_feats[feature_selected, :]

    mid_feats_norm, mean, std = at.normalize_features([mid_feats.T])
    mid_feats_norm = mid_feats_norm[0].T
    n_wins = mid_feats.shape[1]

    dist_all = np.sum(distance.squareform(distance.pdist(mid_feats_norm.T)),
                      axis=0)
    m_dist_all = np.mean(dist_all)
    i_non_outliers = np.nonzero(dist_all < 1.2 * m_dist_all)[0]

    mt_feats_norm_or = mid_feats_norm
    mid_feats_norm = mid_feats_norm[:, i_non_outliers]
    if n_speakers <= 0:
        s_range = range(2, 10)
    else:
        s_range = [n_speakers]
    cluster_labels = []
    sil_all = []
    cluster_centers = []

    for speakers in s_range:
        k_means = sklearn.cluster.KMeans(n_clusters=speakers)
        k_means.fit(mid_feats_norm.T)
        cls = k_means.labels_        
        means = k_means.cluster_centers_
        cluster_labels.append(cls)
        cluster_centers.append(means)
        sil_1 = []; sil_2 = []
        for c in range(speakers):
            # for each speaker (i.e. for each extracted cluster)
            clust_per_cent = np.nonzero(cls == c)[0].shape[0] / float(len(cls))
            if clust_per_cent < 0.020:
                sil_1.append(0.0)
                sil_2.append(0.0)
            else:
                mt_feats_norm_temp = mid_feats_norm[:, cls == c]
                dist = distance.pdist(mt_feats_norm_temp.T)
                sil_1.append(np.mean(dist)*clust_per_cent)
                sil_temp = []
                for c2 in range(speakers):
                    if c2 != c:
                        clust_per_cent_2 = np.nonzero(cls == c2)[0].shape[0] /\
                                           float(len(cls))
                        mid_features_temp = mid_feats_norm[:, cls == c2]
                        dist = distance.cdist(mt_feats_norm_temp.T,
                                              mid_features_temp.T)
                        sil_temp.append(np.mean(dist)*(clust_per_cent
                                                       + clust_per_cent_2)/2.0)
                sil_temp = np.array(sil_temp)
                sil_2.append(min(sil_temp))
        sil_1 = np.array(sil_1)
        sil_2 = np.array(sil_2)
        sil = []
        for c in range(speakers):
            sil.append((sil_2[c] - sil_1[c]) / (max(sil_2[c], sil_1[c]) + 1e-5))
        sil_all.append(np.mean(sil))

    imax = int(np.argmax(sil_all))
    num_speakers = s_range[imax]
    cls = np.zeros((n_wins,))
    for index in range(n_wins):
        j = np.argmin(np.abs(index-i_non_outliers))
        cls[index] = cluster_labels[imax][j]
    for index in range(1):
        start_prob, transmat, means, cov = \
            train_hmm_compute_statistics(mt_feats_norm_or, cls)
        hmm = hmmlearn.hmm.GaussianHMM(start_prob.shape[0], "diag")
        hmm.startprob_ = start_prob
        hmm.transmat_ = transmat            
        hmm.means_ = means; hmm.covars_ = cov
        cls = hmm.predict(mt_feats_norm_or.T)                    
    
    # Post-process method 2: median filtering:
    cls = scipy.signal.medfilt(cls, 13)
    cls = scipy.signal.medfilt(cls, 11)

    class_names = ["speaker{0:d}".format(c) for c in range(num_speakers)]

    if plot_res:
        fig = plt.figure(figsize=(10, 4))    
        if n_speakers > 0:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        list_labels = np.array(range(len(cls))) * mid_step + mid_step
        ax1.set_xticks(list_labels[::25])
        ax1.plot(np.array(range(len(cls))) * mid_step + mid_step, cls)

    if plot_res:
        plt.xlabel("time (seconds)")
        if n_speakers <= 0:
            plt.subplot(212)
            plt.plot(s_range, sil_all)
            plt.xlabel("number of clusters")
            plt.ylabel("average clustering's sillouette")
        plt.savefig('foo.png')
    return cls, sampling_rate, len(signal)