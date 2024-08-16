import logging
import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
import torch.cuda.amp as amp
import torchaudio
from PIL import Image
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from torchmultimodal.models.flava.model import flava_model
from torchmultimodal.transforms.flava_transform import FLAVAImageTransform
from transformers import BertTokenizer, CLIPModel, CLIPProcessor, pipeline
import speech_recognition as sr
from gtts import gTTS
import os
import pygame
import mss
import cv2
import pyaudio
import usb.core
import usb.util
import tempfile
import wave
import io
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
from dotenv import load_dotenv
import librosa
import base64

load_dotenv()

class LazyLoader:
    def __init__(self, import_path, class_name):
        self.import_path = import_path
        self.class_name = class_name
        self._instance = None

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            module = __import__(self.import_path, fromlist=[self.class_name])
            class_ = getattr(module, self.class_name)
            self._instance = class_(*args, **kwargs)
        return self._instance

class RequestQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.processing = False

    def add_request(self, request):
        self.queue.put(request)
        if not self.processing:
            threading.Thread(target=self.process_requests).start()

    def process_requests(self):
        self.processing = True
        while not self.queue.empty():
            request = self.queue.get()
            response = vita.generate_response(**request)
            print(f"Processed request: {response}")
            self.queue.task_done()
        self.processing = False

class VITA:
    def __init__(self, model_name="OpenGVLab/VITA-7B-V1.0"):
        self.setup_logging()
        self.logger.info("Initializing VITA...")
        
        self.device = torch.device("cpu")
        self.logger.info(f"Using device: {self.device}")
        
        self.load_models(model_name)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize Silero VAD
        self.init_vad()
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        self.logger.info("VITA initialization complete.")

    def setup_logging(self):
        self.logger = logging.getLogger('VITA')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def load_models(self, model_name):
        try:
            self.logger.info("Loading FLAVA model...")
            self.flava_model = flava_model(pretrained=True).to(self.device)
            self.flava_image_transform = FLAVAImageTransform(is_train=False)
            
            self.logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            self.logger.info("Loading tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            
            # Enable automatic mixed precision for faster inference
            self.scaler = amp.GradScaler()
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

    def list_audio_devices(self):
        self.logger.info("Scanning for audio devices...")
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                device_info = {
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                }
                input_devices.append(device_info)
                self.logger.info(f"{i}: {device['name']} (Channels: {device['max_input_channels']}, Default SR: {device['default_samplerate']})")

        return input_devices

    def listen_audio_stream(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000  # Changed from 44100 to 16000
        RECORD_SECONDS = 5

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        self.logger.info("Listening for audio...")

        while True:
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            audio_data = b''.join(frames)
            self.audio_queue.put(audio_data)

            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    def visualize_audio(self, data):
        max_amplitude = np.max(np.abs(data))
        bar_length = int(max_amplitude * 50)  # Scale to 0-50 characters
        print(f"Audio level: {'|' * bar_length}")

    def process_audio_stream(self):
        while True:
            try:
                audio_chunk = self.audio_queue.get()
                self.logger.debug("Processing audio chunk")
                
                # Convert bytes to numpy array
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
                
                if self.detect_speech(audio_np):
                    self.logger.info("Speech detected")
                    text = self.transcribe(audio_chunk)
                    if text:
                        self.logger.info(f"Transcribed: {text}")
                        response = self.generate_response(text)
                        self.logger.info(f"Generated response: {response}")
                        self.response_queue.put(response)
                    else:
                        self.logger.warning("Failed to transcribe speech")
                else:
                    self.logger.debug("No speech detected in audio chunk")
            except Exception as e:
                self.logger.error(f"Error processing audio: {str(e)}")

    def init_vad(self):
        self.logger.info("Initializing Silero VAD...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=True)
        self.get_speech_timestamps = utils[0]
        self.logger.info("Silero VAD initialized.")

    def detect_speech(self, audio_waveform):
        try:
            # Convert to float32 if not already
            audio_float = audio_waveform.astype(np.float32) / 32768.0
            
            # Resample to 16000 Hz
            audio_resampled = librosa.resample(audio_float, orig_sr=16000, target_sr=16000)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_resampled).float()
            
            speech_timestamps = self.get_speech_timestamps(audio_tensor, 
                                                           self.vad_model, 
                                                           sampling_rate=16000)
            return len(speech_timestamps) > 0
        except Exception as e:
            self.logger.error(f"Error detecting speech: {str(e)}")
            return False

    def transcribe(self, audio_data):
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file_path = temp_file.name
            temp_file.close()

            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)  # Changed from 44100 to 16000
                wf.writeframes(audio_data)

            with open(temp_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            
            os.unlink(temp_file_path)
            return transcript.text
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {str(e)}")
            return None

    def generate_voice_audio(self, text: str):
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1-hd",
                voice="shimmer",
                input=text,
                response_format="mp3"
            )
            return response.content
        except Exception as e:
            self.logger.error(f"Error generating voice audio: {str(e)}")
            return None

    def speak(self, text: str):
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1-hd",
                voice="shimmer",
                input=text
            )
            audio = AudioSegment.from_mp3(io.BytesIO(response.content))
            play(audio)
        except Exception as e:
            self.logger.error(f"Error speaking text: {str(e)}")

    def process_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            flava_input = self.flava_image_transform(image)["image"].unsqueeze(0).to(self.device)
            clip_input = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            return {"flava": flava_input, "clip": clip_input}
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return None

    def process_text(self, text):
        try:
            flava_input = self.tokenizer(text, return_tensors="pt").to(self.device)
            clip_input = self.clip_processor(text=text, return_tensors="pt").to(self.device)
            return {"flava": flava_input, "clip": clip_input}
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return None

    def process_audio(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.to(self.device)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            return waveform
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            return None

    def should_capture_screen(self, query):
        try:
            decision_prompt = f"""
            Given the following user query, determine if a screenshot of the user's screen would be helpful in answering the question. Respond with only 'Yes' or 'No'.

            User query: "{query}"

            Consider factors such as:
            - Is the user asking about something visual on their screen?
            - Is the user referring to their current activity or something they're looking at?
            - Would seeing the screen provide valuable context for answering the query?

            Decision (Yes/No):
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that determines if a screenshot is needed to answer a user's query."},
                    {"role": "user", "content": decision_prompt}
                ],
                max_tokens=1,
                temperature=0.1
            )

            decision = response.choices[0].message.content.strip().lower()
            return decision == 'yes'
        except Exception as e:
            self.logger.error(f"Error determining if screenshot is needed: {str(e)}")
            return False

    def capture_screen(self):
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # Capture the primary monitor
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        except Exception as e:
            self.logger.error(f"Error capturing screen: {str(e)}")
            return None

    def generate_response(self, query):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant named VITA with the ability to see the user's screen when relevant."},
                {"role": "user", "content": query}
            ]
            
            if self.should_capture_screen(query):
                screen_img = self.capture_screen()
                if screen_img is not None:
                    screen_img_small = cv2.resize(screen_img, (800, 450))
                    _, img_encoded = cv2.imencode('.jpg', screen_img_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                    
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here's the current view of the user's screen:"},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                        ]
                    })
                    self.logger.info("Screen capture included in response generation")
                else:
                    self.logger.warning("Screen capture failed, proceeding without it")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content
            return response_text
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."

    def process_responses(self):
        while True:
            try:
                response = self.response_queue.get()
                self.logger.info(f"Speaking response: {response}")
                self.speak(response)
            except Exception as e:
                self.logger.error(f"Error processing response: {str(e)}")

    def run_interactive(self):
        try:
            self.logger.info("Starting VITA in interactive mode...")
            
            threads = [
                threading.Thread(target=self.listen_audio_stream),
                threading.Thread(target=self.process_audio_stream),
                threading.Thread(target=self.process_responses),
            ]
            
            for thread in threads:
                thread.start()
            
            print("VITA is now listening. Speak to interact.")
            
            while True:
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Error in run_interactive: {str(e)}")
        finally:
            self.logger.info("Stopping VITA...")
            # Add any cleanup code here if needed

app = FastAPI()

class QueryInput(BaseModel):
    query: str
    image_path: Optional[str] = None
    audio_path: Optional[str] = None

vita = VITA()

@app.post("/generate_response")
async def generate_response(query_input: QueryInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(vita.request_queue.add_request, query_input.dict())
    return {"message": "Request added to queue"}

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return {"filename": file.filename}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VITA in API or interactive mode")
    parser.add_argument("--mode", choices=["api", "interactive"], default="interactive", help="Run mode")
    args = parser.parse_args()

    vita = VITA()

    if args.mode == "api":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        vita.run_interactive()