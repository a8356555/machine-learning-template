from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
from functools import partial

class ModelFactory:
    def __init__(self):
        self._models = {}

    def register_model(self, key, model):
        self._models[key] = model

    def create(self, key, **kwargs):
        model = self._models.get(key)
        if not model:
            raise ValueError(key)
        return model(**kwargs)

model_factory = ModelFactory()
model_factory.register_model('LGBM', lgb.LGBMClassifier)
model_factory.register_model('LR', LogisticRegression)
model_factory.register_model('RF', RandomForestClassifier)
model_factory.register_model('SVM', partial(SVC, probability=True))