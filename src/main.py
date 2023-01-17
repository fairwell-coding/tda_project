import matplotlib.pyplot as plt
import pervect
from music21 import corpus
import numpy as np
import persim
from sklearn.cluster import FeatureAgglomeration, KMeans, MeanShift, Birch, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE

from ripser import Rips

RANDOM_STATE = 42


def __get_bach_choral(index: int):
    chorales = corpus.search('bach')
    return chorales[index].parse()


def __plot_pianoroll_for_range(choral, start_idx: int, end_idx: int):
    choral.measures(start_idx, end_idx).plot('pianoroll', figureSize=(10, 3))


def __prepare_data(choral):
    """ Retrieve note-based representation for Bach chorales separately for each musical measure.
    :param choral: a single Bach choral
    :return: preprocessed data
    """

    note_based_representation = []
    measure_idx = 0

    while len(choral.measure(measure_idx).flat) != 0:
        measure_representation = []
        for n in choral.measure(measure_idx).flat.notes:
            note_on = float(n.offset)
            time = float(n.duration.quarterLength)
            pitch = n.pitch.midi
            measure_representation.append([note_on, time, pitch])
        note_based_representation.append(np.array(measure_representation))
        measure_idx += 1

    return np.array(note_based_representation)


def __create_filtration(measure, plot_pd=False):
    rips = Rips()
    pd = rips.fit_transform(measure)

    if plot_pd:
        rips.plot(pd)
        plt.show()

    return pd



def __preprocess_pd_for_pervert(pd):
    """ Pervert: Take care of infinity values and musical measures which only contain a single played note (since some vectorization algorithms fail with a single data point as input).
    :return: persistence diagram format which can be used as input for vectorization
    """

    pd_filtered = []

    for pd_d in pd:
        if pd_d.shape[0] != 0:
            pd_d[pd_d == np.Inf] = np.max(pd_d[pd_d != np.Inf]) * 10
            pd_filtered.append(pd_d)

    if len(pd_filtered) == 1:  # allow vectorization via Gaussian-mixture model
       pd_filtered = np.vstack((pd_filtered[0], pd_filtered[0]))
       print('x')

    return pd_filtered[0]  # only return dim=0 (i.e. components)

def __preprocess_pd_for_pi(pd):
    """ Persistent images: Take care of infinity values and musical measures which only contain a single played note (since some vectorization algorithms fail with a single data point as input).
    :return: persistence diagram format which can be used as input for vectorization
    """

    pd_filtered = []

    for pd_d in pd:
        if pd_d.shape[0] != 0:
            pd_d[pd_d == np.Inf] = np.max(pd_d[pd_d != np.Inf]) * 10
            pd_filtered.append(pd_d)

    return pd_filtered[0]  # only return dim=0 (i.e. components)


def __vectorize_PerVect(pds):
    return pervect.PersistenceVectorizer().fit_transform(pds)
    

def __vectorize_PI(pds):
    pimgr = persim.PersistenceImager(pixel_size=1)
    pimgr.fit(pds, skew=True)
    pimgr.birth_range = pimgr.birth_range[0], pimgr.birth_range[1] + 0.1
    persistent_vectors = pimgr.transform(pds, skew=True)

    prepared_vectors = []
    for vector in persistent_vectors:
        prepared_vectors.append(vector[0])
    prepared_vectors = np.array(prepared_vectors)

    return prepared_vectors


def __cluster_euclidean_vectors(vectors):
    labels = {}

    __kmeans_clustering(labels, vectors)
    __agglomerative_clustering(labels, vectors)
    __feature_agglomerative_clustering(labels, vectors)
    __dbscan_clustering(labels, vectors)
    __birch_clustering(labels, vectors)
    __mean_shift_clustering(labels, vectors)

    return labels


def __mean_shift_clustering(labels, vectors):
    mean_shift = MeanShift()
    mean_shift.fit(vectors)
    labels["mean_shift"] = mean_shift.labels_
    # __plot_clustering_result(mean_shift, "Blobs in smooth density (mean shift)", vectors)


def __birch_clustering(labels, vectors):
    birch = Birch()
    birch.fit(vectors)
    labels["birch"] = birch.labels_
    # __plot_clustering_result(birch, "Tree-based (BIRCH)", vectors)


def __dbscan_clustering(labels, vectors):
    db = DBSCAN(eps=0.8, min_samples=2)
    db.fit(vectors)
    labels["dbscan"] = db.labels_
    # __plot_clustering_result(db, "Density-based (DBSCAN)", vectors)


def __feature_agglomerative_clustering(labels, vectors):
    feature_agglo = FeatureAgglomeration(n_clusters=2)
    feature_agglo.fit_transform(vectors)
    labels["feature agglomeration"] = feature_agglo.labels_
    # __plot_clustering_result(feature_agglo, "Hierarchical (feature agglomerative)", vectors)


def __agglomerative_clustering(labels, vectors):
    agglo = AgglomerativeClustering(n_clusters=4)
    agglo.fit(vectors)
    labels["agglomeration"] = agglo.labels_
    # __plot_clustering_result(agglo, "Hierarchical (agglomerative)", vectors)


def __kmeans_clustering(labels, vectors):
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE).fit(vectors)
    labels["kmeans"] = kmeans.labels_
    __plot_clustering_result(kmeans, "Kmeans++", vectors)


def __plot_clustering_result(clustering_algorithm, clustering_name, vectors, dim_reduction="t-sne"):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'lightgreen', 'darkred']
    num_points_in_clusters = []

    for i in range(len(colors)):  # show found clusters in different colors
        mask = np.where(clustering_algorithm.labels_ == i, True, False)
        num_points_in_clusters.append(np.count_nonzero(mask))
        plt.scatter(vectors[mask, 0], vectors[mask, 1], label=f'cluster {i}', color=colors[i], s=4)

    for i in range(len(vectors)):  # add musical measure numbers
        plt.text(vectors[i, 0], vectors[i, 1], str(i))

    # plt.title(f'{clustering_name} n_clusters = {clustering_algorithm.n_clusters}: {dim_reduction} for 2d visualization')
    plt.title(f'{clustering_name}: {dim_reduction} for 2d visualization')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def __plot_vectorization_output(vectors_2d):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'lightgreen', 'darkred',
              'pink', 'salmon', 'darkgrey', 'paleturquoise', 'cornflowerblue', 'fuchsia', 'midnightblue', 'olive', 'dodgerblue']

    cm = plt.cm.get_cmap(lut=len(vectors_2d))

    for i in range(len(vectors_2d)):
        plt.plot(vectors_2d[i, 0], vectors_2d[i, 1], '.', label=i, color=colors[i])
        plt.text(vectors_2d[i, 0], vectors_2d[i, 1], str(i))

    plt.title('Dimensionality reduction: t-sne')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def __convert_euclidean_vectors_to_2d(vectors):
    return TSNE(perplexity=3).fit_transform(vectors)


if __name__ == '__main__':
    choral = __get_bach_choral(0)
    measures = __prepare_data(choral)
    # __plot_pianoroll_for_range(choral, 0, len(measures))

    pds_filtered = []
    for measure in measures:
        pd = __create_filtration(measure)
        pd_filtered = __preprocess_pd_for_pi(pd)
        # pd_filtered = __preprocess_pd_for_pervert(pd)
        pds_filtered.append(pd_filtered)

    vectors = __vectorize_PI(pds_filtered)
    # vectors = __vectorize_PerVect(pds_filtered)
    vectors_2d = __convert_euclidean_vectors_to_2d(vectors)
    __plot_vectorization_output(vectors_2d)
    labels = __cluster_euclidean_vectors(vectors_2d)

    print(labels)
