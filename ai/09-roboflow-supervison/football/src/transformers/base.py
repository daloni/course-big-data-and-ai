from abc import ABC, abstractmethod

class Base(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def infer(self, frame, confidence=0.3):
        pass