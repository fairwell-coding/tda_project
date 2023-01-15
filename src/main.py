from music21 import corpus
import numpy as np
import matplotlib.pyplot as plt
from ripser import *
import pervect

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


def __create_filtration(plot_pd=False):
    rips = Rips()
    pd = rips.fit_transform(measure)

    if plot_pd:
        rips.plot(pd)
        plt.show()

    return pd


def __preprocess_pd_for_vectorization():
    """ Take care of infinity values and musical measures which only contain a single played note (since some vectorization algorithms fail with a single data point as input).
    :return: persistence diagram format which can be used as input for vectorization
    """

    pd_filtered = []

    for pd_d in pd_components:
        if pd_d.shape[0] != 0:
            pd_d[pd_d == np.Inf] = np.max(pd_d[pd_d != np.Inf]) * 10
            pd_filtered.append(pd_d)

    if len(pd_filtered) == 1:  # allow vectorization via Gaussian-mixture model
        pd_filtered = np.vstack((pd_filtered[0], pd_filtered[0]))
        print('x')

    return pd_filtered


def __cluster_euclidean_vectors():
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


if __name__ == '__main__':
    choral = __get_bach_choral(0)
    # __plot_pianoroll_for_range(choral, 0, 17)
    measures = __prepare_data(choral)

    vectors = []

    for measure in measures:
        pd = __create_filtration()
        pd_components = pd[0]  # only use first dimension (i.e. components) to detect pattern classes (other dimensions add data noise for that downstream task)
        pd_filtered = __preprocess_pd_for_vectorization()
        vectors.append(pervect.PersistenceVectorizer(n_components=2).fit_transform(pd_filtered))

    __cluster_euclidean_vectors()

    print(vectors)
