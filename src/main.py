from api import *
from expirementUnetModel import build_model


model = build_model((160, 160, 3), 32)
