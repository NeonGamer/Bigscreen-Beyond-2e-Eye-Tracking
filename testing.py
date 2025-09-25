
import numpy as np
#    0 - 0.375 eye_openness
#0.375 - 0.75  eey_squint
# 0.75 - 1     eye_widen

def remap(value, fromMin, fromMax, toMin, toMax):
    return toMin + (value - fromMin) / (fromMax - fromMin) * (toMax - toMin)

x = .5

widen    = (np.clip(x, 0.75, 1) - 0.75) / 0.25
openness = np.clip(x, 0, 0.375) / 0.375
squint   = 0