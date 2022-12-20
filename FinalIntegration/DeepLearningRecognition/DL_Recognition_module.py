import os
import torch

class DeepLearningRecognition(object):
    def __init__(self, carlaWorld, config = None) -> None:
        if not config:
            config = dict()
        super().__init__()
        self.config = config
        self._carlaWorld = carlaWorld
        self._agent = carlaWorld.getPlayer()

        self.device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self._model = self._load_model("")
        self._model_name = "/"

        self.detections = None
        self.detected_image = None

    def detect(self):
        #load image from sensor
        sensor = self.carlaWorld.get_sensor("Camera")
        if sensor is None:
            return
        image = sensor.getState()
        if image is None:
            return

        tensor = torch.tensor(image).to(self.device)
        output = self._model.forward(tensor)

        self.detections = output
        self.detected_image = output

    def _load_model(self, filename):
        if not (os.path.exists(filename) and os.path.isfile(filename)):
            return None

        #model = torch.load(filename)
        #return model

    def getModelName(self) -> str:
        return self._model_name