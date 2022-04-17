# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 09:10:49 2022

@author: damini
"""

# video to text


import moviepy
import moviepy.editor
import speech_recognition as sr
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

r = sr.Recognizer()

class vid2txt:
    def extract_audio(file):
        
        # Replace the parameter with the location of the video
        video = moviepy.editor.VideoFileClip(file)
        audio = video.audio
        # Replace the parameter with the location along with filename
        #audio.write_audiofile("sample1.wav")
        return audio.write_audiofile("sample3.wav")


    # initialize the recognizer
    # create a speech recognition object


    # a function that split_on_silences the audio file into chunks
    # and applies speech recognition
    def get_large_audio_transcription(wav_file):
        
        # open the audio file using pydub
        sound = AudioSegment.from_wav(wav_file)  
        # split audio sound where silence is 700 miliseconds or more and get chunks
        chunks = split_on_silence(sound,
            # experiment with this value for your target audio file
            min_silence_len = 500,
            # adjust this per requirement
            silence_thresh = sound.dBFS-14,
            # keep the silence for 1 second, adjustable as well
            keep_silence=500,
        )
        folder_name = "audio-chunks"
        # create a directory to store the audio chunks
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        whole_text = ""
        # process each chunk 
        for i, audio_chunk in enumerate(chunks, start=1):
            # export audio chunk and save it in
            # the `folder_name` directory.
            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            # recognize the chunk
            with sr.AudioFile(chunk_filename) as source:
                audio_listened = r.record(source)
                # try converting it to text
                try:
                    text = r.recognize_google(audio_listened)
                except sr.UnknownValueError as e:
                    print("Error:", str(e))
                else:
                    text = f"{text.capitalize()}. "
                    print(chunk_filename, ":", text)
                    whole_text += text
        # return the text for all chunks detected
        return whole_text


