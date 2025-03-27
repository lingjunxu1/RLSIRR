import importlib
from .arch import *


def make_model(name: str):
    base = importlib.import_module('models.' + name)
    model = getattr(base, 'DSRNetModel')
    return model
