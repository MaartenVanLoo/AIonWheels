
class DeepLearningRecognition(object):
    def __init__(self, carlaWorld, config = None) -> None:
        if not config:
            config = dict()
        super().__init__()
        self._model_name = "/"

    def getModelName(self) -> str:
        return self._model_name