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
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.init_vad()
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
                dev