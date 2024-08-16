import queue
import threading

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
