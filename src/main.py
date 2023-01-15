from music21 import corpus


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
    pd = rips.fit_transform(data[1])
    rips.plot(pd)
    plt.show()

    pd = [np.nan_to_num(pd[0]), np.nan_to_num(pd[1])]
    pd[0][0, 1] = 10000
    vectors = pervect.PersistenceVectorizer().fit_transform(pd)

    print('x')