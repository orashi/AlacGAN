from .standard import *


def get_model(config):
    return globals()[config['arch']](**config['kwargs'])
