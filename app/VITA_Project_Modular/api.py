
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from audio_processing import AudioProcessor
from image_processing import ImageProcessor
from text_processing import TextProcessor

app = FastAPI()

audio_processor = AudioProcessor()
image_processor = ImageProcessor()
text_processor = TextProcessor()

@app.get('/api/audio/devices')
async def get_audio_devices():
    devices = audio_processor.list_audio_devices()
    return JSONResponse(content=devices)

@app.post('/api/image/process')
async def process_image(image_path: str):
    image = image_processor.process_image(image_path)
    return JSONResponse(content={'status': 'success', 'image': image})

@app.post('/api/text/process')
async def process_text(text: str):
    processed_text = text_processor.process_text(text)
    return JSONResponse(content={'status': 'success', 'data': processed_text})
