
from transformers import BertTokenizer, CLIPProcessor
import logging
from logging_setup import setup_logging

class TextProcessor:
    def __init__(self):
        self.logger = setup_logging()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    def process_text(self, text):
        try:
            flava_input = self.tokenizer(text, return_tensors='pt')
            clip_input = self.clip_processor(text=text, return_tensors='pt')
            return {'flava': flava_input, 'clip': clip_input}
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return None
