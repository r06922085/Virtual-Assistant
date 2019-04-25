import speech_recognition
from gtts import gTTS
from pygame import mixer
import pygame
import tempfile
from gtts import gTTS
import time

def speak(sentense):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text=sentense,lang='zh-tw')
        tts.save("{}.mp3".format(fp.name))
        pygame.mixer.init()
        pygame.mixer.music.load('{}.mp3'.format(fp.name))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): 
            pygame.time.Clock().tick(10)
def listen():   
 
    while True:
        r = speech_recognition.Recognizer()
        with speech_recognition.Microphone() as source:
            audio = r.listen(source)
            #sentense = "123"
        r.energy_threshold = 400;
        try:
            sentense = r.recognize_google(audio, language='zh-TW',show_all = False)
            break;
        except speech_recognition.UnknownValueError:
            print('我聽不到')
            speak('你為甚麼不講話?')
    return sentense