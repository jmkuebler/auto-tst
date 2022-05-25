# Import packages
import numpy as np

from .model import AutoGluonTabularPredictor



class AutoTST:
    """
    AutoML Two-Sample Test

    Documentation with example of the class goes here
    """
    def __int__(self, X, Y, model=AutoGluonTabularPredictor(), split_ratio=0.5):
        """
        Constructor
        Add doc for params here
        :param X:
        :param Y:
        :param model:
        :param split_ratio:
        :return:
        """
        pass

    def split_data(self):
        """
        Split & label data
        :return:
        """
        pass

    def fit_witness(self):
        """
        Fit witness
        :return:
        """
        pass

    def p_value_evaluate(self):
        """
        Evaluate p value
        :return:
        """
        pass

    def p_value(self):
        self.split_data()
        self.fit_witness()
        pval = self.p_value_evaluate()
        return pval