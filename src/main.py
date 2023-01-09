from music21 import corpus

if __name__ == '__main__':
    chorales = corpus.search('bach')
    choral = chorales[0]
    score = choral.parse()

    partStream = score.parts.stream()

    #score.measures(0, 2).plot('pianoroll', figureSize=(10, 3))

    for n in score.flat.notes:
        # print("Note: %s%d %0.1f" % (n.pitch.name, n.pitch.octave, n.duration.quarterLength))
        print(f"time: {float(n.duration.quarterLength):.2f}, note: {n.pitch.nameWithOctave}:{n.pitch.midi}, velocity: {n.pitch.frequency:.2f}, note-on: {float(n.offset):.2f}")

