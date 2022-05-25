# Import packages
import numpy as np



class Model:
    """
    Generic model class for two-sample tests
    """
    def __int__(self):
        raise NotImplementedError()

    def fit(self, X):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()



class AutoGluonTabularPredictor(Model):
    """
    Wrapper model for the Tabular Predictor of the AutoGluon
    package
    """
    def __int__(self):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass

