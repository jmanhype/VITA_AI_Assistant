import logging
import os
import torch
from openai import OpenAI
from dotenv import load_dotenv
from torchmultimodal.models.flava.model import flava_model
from torchmultimodal.transforms.flava_transform import FLAVAImageTransform
from transformers import BertTokenizer, CLIPModel, CLIPProcessor

load_dotenv()

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
