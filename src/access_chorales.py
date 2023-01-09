from music21 import corpus


if __name__ == '__main__':

    for score in corpus.chorales.Iterator():
        # score.plot('pianoroll', figureSize=(10, 3))
        # score.measures(1, 3).plot('pianoroll', figureSize=(10, 3))
        # score.measures(1, 3).plot('histogram', 'duration', figureSize=(10, 3))
        score.measures(1, 7).plot('pianoroll', figureSize=(10, 3))
        pass

    # chorales = corpus.search('bach', fileExtensions='xml')
    # bwv1 = chorales[0].parse()
    # bwv1.measures(0, 3).show()

    allBach = corpus.search('bach')

    x = allBach[0]
    p = x.parse()

    partStream = p.parts.stream()

    for n in p.flat.notes:
        print("Note: %s%d %0.1f" % (n.pitch.name, n.pitch.octave, n.duration.quarterLength))

    pass