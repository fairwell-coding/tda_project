import matplotlib.pyplot as plt
import pervect
from music21 import corpus
import numpy as np
import persim
from sklearn.cluster import FeatureAgglomeration, KMeans, MeanShift, Birch, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE

from ripser import Rips


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


def __preprocess_pd_for_vectorization(pd):
    """ Take care of infinity values and musical measures which only contain a single played note (since some vectorization algorithms fail with a single data point as input).
    :return: persistence diagram format which can be used as input for vectorization
    """

    pd_filtered = []

    for pd_d in pd:
        if pd_d.shape[0] != 0:
            pd_d[pd_d == np.Inf] = np.max(pd_d[pd_d != np.Inf]) * 10
            pd_filtered.append(pd_d)

    #if len(pd_filtered) == 1:  # allow vectorization via Gaussian-mixture model
    #    pd_filtered = np.vstack((pd_filtered[0], pd_filtered[0]))
    #    print('x')

    return pd_filtered[0]  # only return dim=0 (i.e. components)


def __vectorize(pds):
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

    kmeans = KMeans(n_clusters=6)
    kmeans.fit(vectors)
    labels["kmeans"] = kmeans.labels_

    agglo = AgglomerativeClustering(n_clusters=4)
    agglo.fit(vectors)
    labels["agglomeration"] = agglo.labels_

    feature_agglo = FeatureAgglomeration(n_clusters=4)
    feature_agglo.fit_transform(vectors)
    labels["feature agglomeration"] = feature_agglo.labels_

    db = DBSCAN(eps=0.8, min_samples=2)
    db.fit(vectors)
    labels["dbscan"] = db.labels_

    birch = Birch()
    birch.fit(vectors)
    labels["birch"] = birch.labels_

    mean_shift = MeanShift()
    mean_shift.fit(vectors)
    labels["mean_shift"] = mean_shift.labels_

    return labels


def __plot_vectorization_output(vectors):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'lightgreen', 'darkred',
              'pink', 'salmon', 'darkgrey', 'paleturquoise', 'cornflowerblue', 'fuchsia', 'midnightblue', 'olive', 'dodgerblue']

    tsne = TSNE(perplexity=3).fit_transform(vectors)
    cm = plt.cm.get_cmap(lut=len(tsne))

    for i in range(len(tsne)):
        plt.plot(tsne[i, 0], tsne[i, 1], '.', label=i, color=colors[i])
        plt.text(tsne[i, 0], tsne[i, 1], str(i))
    # plt.scatter(tsne[:, 0], tsne[:, 1], label='2-dim t-sne (pca)', s=4)

    plt.title('Clustering: t-sne')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    choral = __get_bach_choral(0)
    measures = __prepare_data(choral)
    # __plot_pianoroll_for_range(choral, 0, len(measures))

    pds_filtered = []

    for measure in measures:
        pd = __create_filtration(measure)
        pd_filtered = __preprocess_pd_for_vectorization(pd)
        pds_filtered.append(pd_filtered)

    vectors = __vectorize(pds_filtered)
    __plot_vectorization_output(vectors)
    labels = __cluster_euclidean_vectors(vectors)

    print(labels)
