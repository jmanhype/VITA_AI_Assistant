from gtts import gTTS

class TextToSpeech:
    def speak(self, text):
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
