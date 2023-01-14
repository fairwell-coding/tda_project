from music21 import corpus

if __name__ == '__main__':
    chorales = corpus.search('bach')
    choral = chorales[0]
    score = choral.parse()

    # partStream = score.parts.stream()

    # score.measures(1, 31).plot('pianoroll', figureSize=(10, 3))

    note_based_representation = []
    for n in score.flat.notes:
        #print(f"time: {float(n.duration.quarterLength):.2f}, note: {n.pitch.nameWithOctave}:{n.pitch.midi}, velocity: {n.pitch.frequency:.2f}, note-on: {float(n.offset):.2f}")
        note_based_representation.append([float(n.duration.quarterLength), n.pitch.midi, float(n.offset)])

    data = np.array(note_based_representation)
    # data = data[:, [True, True, False, True]]

    rips = Rips()
    diagrams = rips.fit_transform(data)
    rips.plot(diagrams)
    plt.show()
