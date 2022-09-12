import numpy as np
import librosa
import midiutil


#matrica tranzicije - T[i, j] - verovatnoca da ce se iz stanja i preci u stanje j
def transition_matrix(note_min, note_max, p_stay_note, p_stay_silence):
    midi_min = librosa.note_to_midi(note_min)
    midi_max = librosa.note_to_midi(note_max)
    n_notes = midi_max - midi_min + 1
    p_ = (1-p_stay_silence)/n_notes 
    p__ = (1-p_stay_note)/(n_notes+1)
    
    T = np.zeros((2*n_notes+1, 2*n_notes+1))
    
    T[0,0] = p_stay_silence
    for i in range(n_notes):
        T[0, (i*2)+1] = p_ 
    
    for i in range(n_notes):
        T[(i*2)+1, (i*2)+2] = 1

    for i in range(n_notes):
        T[(i*2)+2, 0] = p__
        T[(i*2)+2, (i*2)+2] = p_stay_note
        for j in range(n_notes):        
            T[(i*2)+2, (j*2)+1] = p__ 

    return T


def probabilities(y, note_min, note_max, sr, frame_length, window_length, hop_length,
                  pitch_acc, voiced_acc, onset_acc, spread):

    fmin = librosa.note_to_hz(note_min) 
    fmax = librosa.note_to_hz(note_max) 
    midi_min = librosa.note_to_midi(note_min) 
    midi_max = librosa.note_to_midi(note_max)
    n_notes = midi_max - midi_min + 1

    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin*0.9, fmax*1.1, sr, frame_length, window_length, hop_length)
    
    tuning = librosa.pitch_tuning(f0)
    f0_ = np.round(librosa.hz_to_midi(f0-tuning)).astype(int)

    onsets = librosa.onset.onset_detect(y, sr=sr, hop_length=hop_length, backtrack=True) 

    P = np.ones( (n_notes*2 + 1, len(f0)) )

    for t in range(len(f0)):
        if voiced_flag[t]==False:
            P[0,t] = voiced_acc
        else:
            P[0,t] = 1-voiced_acc 

        for j in range(n_notes):
            if t in onsets:
                P[(j*2)+1, t] = onset_acc
            else:
                P[(j*2)+1, t] = 1-onset_acc
            
            if j+midi_min == f0_[t]:
                P[(j*2)+2, t] = pitch_acc 

            elif np.abs(j+midi_min-f0_[t])==1:
                P[(j*2)+2, t] = pitch_acc * spread 

            else:
                P[(j*2)+2, t] = 1-pitch_acc 

    return P

#tranksripcija procenjenih stanja u note koje su razumljive MIDI protokolom
def states_to_notes(states, note_min, note_max, hop_time):
    midi_min = librosa.note_to_midi(note_min) 
    midi_max = librosa.note_to_midi(note_max)
    
    states_ = np.hstack( (states, np.zeros(1))) 

    silence = 0
    onset = 1
    sustain = 2

    my_state = silence 
    output = []
    
    last_onset = 0
    last_offset = 0
    last_midi = 0
    for i in range(len(states_)):
        if my_state == silence:
            if int(states_[i]%2) != 0:
                last_onset = i * hop_time 
                last_midi = ((states_[i]-1)/2)+midi_min 
                last_note = librosa.midi_to_note(last_midi) 
                my_state = onset 

        elif my_state == onset:
            if int(states_[i]%2) == 0:
                my_state = sustain 

        elif my_state == sustain:
            if int(states_[i]%2) != 0:
                last_offset = i*hop_time 
                my_note = [last_onset, last_offset, last_midi, last_note] 
                output.append(my_note) 
                
                last_onset = i * hop_time 
                last_midi = ((states_[i]-1)/2)+midi_min 
                last_note = librosa.midi_to_note(last_midi) 
                my_state = onset 

            elif states_[i]==0:
                last_offset = i*hop_time 
                my_note = [last_onset, last_offset, last_midi, last_note] 
                output.append(my_note) 
                my_state = silence 

    return output



def notes_to_midi(y, notes):

    bpm = librosa.beat.tempo(y)[0] 
    quarter_note = 60/bpm 
    
    onsets = np.array([n[0] for n in notes]) 
    offsets = np.array([n[1] for n in notes]) 
    
    onsets = onsets / quarter_note 
    offsets = offsets  / quarter_note 
    durations = offsets-onsets 
    
    MyMIDI = midiutil.MIDIFile(1)
    MyMIDI.addTempo(0, 0, bpm)

    for i in range(len(onsets)):
        MyMIDI.addNote(0, 0, int(notes[i][2]), onsets[i], durations[i], 100)

    return MyMIDI


def run(file_in, file_out):
    note_min='A2' 
    note_max='E6' 
    voiced_acc = 0.9 
    onset_acc = 0.8 
    frame_length=2048 
    window_length=1024 
    hop_length=256 
    pitch_acc = 0.99 
    spread = 0.6 
    
    y, sr = librosa.load(file_in) 

    T = transition_matrix(note_min, note_max, 0.9, 0.2)

    P = probabilities(y, note_min, note_max, sr, frame_length, window_length,
                      hop_length, pitch_acc, voiced_acc, onset_acc, spread)
    p_init = np.zeros(T.shape[0]) 
    p_init[0] = 1 
    
    states = librosa.sequence.viterbi(P, T, p_init=p_init) 
    notes = states_to_notes(states, note_min, note_max, hop_length/sr) 
    MyMIDI = notes_to_midi(y, notes)

    with open(file_out, "wb") as output_file:
        MyMIDI.writeFile(output_file)

    

