import os
import torch


class DeepLearningRecognition(object):
    def __init__(self, carlaWorld, config=None) -> None:
        if not config:
            config = dict()
        super().__init__()
        self.config = config
        self._carlaWorld = carlaWorld
        self._agent = carlaWorld.getPlayer()

        self.device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self._model = self._load_model(config.get("model_path",""), config.get("hubconf_path",""))
        self._model_name = "/"

        self.detections = None
        self.detected_image = None

    def detect(self):
        # Load image from sensor
        sensor = self._carlaWorld.get_sensor("Camera")

        if sensor is None:
            return
        image = sensor.getState()
        if image is None:
            return

        tensor = torch.tensor(image.copy()).to(self.device)
        tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)
        output = self._model.forward(tensor)

        self.detections = output
        self.detected_image = output

    def _load_model(self, filename, hubconf_path):
        if not (os.path.exists(filename) and os.path.isfile(filename)):
            print("Model file not found")
            return None
        model = torch.hub.load(hubconf_path, 'custom', path=filename, source='local')
        return model

    def getModelName(self) -> str:
        return self._model_name






