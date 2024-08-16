import speech_recognition as sr

class SpeechRecognition:
    def recognize_speech(self, audio_data):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
