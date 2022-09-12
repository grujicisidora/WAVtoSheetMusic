import PySimpleGUI as psg
import os
import subprocess
import midiConverter

working_directory = os.getcwd()

layout = [
    [psg.Text("Choose a WAV file:")],
    [psg.InputText(key = "-FILE_PATH-"),
     psg.FileBrowse(initial_folder = working_directory, file_types = [("Audio files", "*.wav")])],
    [psg.Button("Convert"), psg.Exit()]
    ]

window = psg.Window("Automatic Melody Transcripton", layout)

def start_conversion(file_address):
    extension = ".wav"
    wav_file_address = file_address
    midi_file_address = file_address.replace(extension, ".midi")
    ly_file_address = file_address.replace(extension, ".ly")
    midiConverter.run(wav_file_address, midi_file_address)
    cmd = 'python midi2ly.py ' + midi_file_address + ' -o ' + ly_file_address
    subprocess.call(cmd, shell = True)
    os.system(ly_file_address)

while True:
    event, values = window.read()
    if event in (psg.WIN_CLOSED, 'Exit'):
        break
    elif event == "Convert":
        file_address = values["-FILE_PATH-"]
        start_conversion(file_address)
        psg.popup("The transcription has finished successfully.")
    
window.close()
