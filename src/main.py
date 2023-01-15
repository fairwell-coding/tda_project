from music21 import corpus
import numpy as np
import matplotlib.pyplot as plt
from ripser import *
import pervect

def __get_bach_choral(index: int):
    chorales = corpus.search('bach')
    return chorales[index].parse()


def __plot_pianoroll_for_range(start_idx: int, end_idx: int):
    choral.measures(start_idx, end_idx).plot('pianoroll', figureSize=(10, 3))


def __prepare_data():
    note_based_representation = []
    measure_idx = 0

    while len(choral.measure(measure_idx).flat) != 0:
        measure_representation = []
        for n in choral.measure(measure_idx).flat.notes:
            measure_representation.append([float(n.offset), float(n.duration.quarterLength), n.pitch.midi])  # note-on, time, note
        note_based_representation.append(np.array(measure_representation))
        measure_idx += 1

    return np.array(note_based_representation)


if __name__ == '__main__':
    choral = __get_bach_choral(0)
    # __plot_pianoroll_for_range(1, 31)
    data = __prepare_data()

    rips = Rips()
    pd = rips.fit_transform(data[2])
    # rips.plot(pd)
    # plt.show()

    pd_filtered = []
    for pd_d in pd:
        if pd_d.shape[0] != 0:
            pd_d[pd_d == np.Inf] = np.max(pd_d[pd_d != np.Inf]) * 10
            pd_filtered.append(pd_d)
            
    vectors = pervect.PersistenceVectorizer(n_components=3).fit_transform(pd_filtered)

    print(vectors)
