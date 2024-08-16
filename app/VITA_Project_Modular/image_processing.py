
from PIL import Image
import logging
from logging_setup import setup_logging

class ImageProcessor:
    def __init__(self):
        self.logger = setup_logging()

    def process_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return None
