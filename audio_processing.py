import sounddevice as sd

class AudioProcessing:
    def list_audio_devices(self):
        self.logger.info("Scanning for audio devices...")
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append(device)
        return input_devices
