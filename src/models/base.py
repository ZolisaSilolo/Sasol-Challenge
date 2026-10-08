from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass
        
    @abstractmethod
    def predict(self, X):
        pass
