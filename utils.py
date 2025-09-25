import numpy as np
from pythonosc import udp_client

def send_eye_osc(lx, ly, blink_l, rx, ry, blink_r, dilation):
    OSC_CLIENT.send_message("/LeftEyeX", float(lx))
    OSC_CLIENT.send_message("/LeftEyeY", float(ly))
    OSC_CLIENT.send_message("/LeftEyeLid", float(blink_l))
    OSC_CLIENT.send_message("/RightEyeX", float(rx))
    OSC_CLIENT.send_message("/RightEyeY", float(ry))
    OSC_CLIENT.send_message("/RightEyeLid", float(blink_r))
    OSC_CLIENT.send_message("/EyePupilDilation", float(dilation))

OSC_CLIENT = udp_client.SimpleUDPClient("127.0.0.1", 8888)

class OutputSmoother:
    def __init__(self, alpha=0.5, t=0.8, use_diff = False):
        self.alpha = alpha
        self.values = None
        self.t = t
        self.use_diff = use_diff

    def curve(self, x, t):
        return np.clip(np.array(x), 0, 1)**t

    def update(self, new_values):
        new_values = np.array(new_values, dtype=np.float32)
        if self.values is None:
            self.values = new_values
        else:
            alpha = self.alpha if self.use_diff else (1 - self.curve(np.abs(new_values - self.values), self.t)) * self.alpha
            self.values = alpha * self.values + (1 - alpha) * new_values
        return self.values.tolist()

def smin(a, b, k):
    h = b - a
    return 0.5 * (a + b - np.sqrt(h * h + k * k))

def smax(a,b,k):
    return -smin(-a, -b, k)

def blink_curve(x):
    x = np.array(x)
    k = 0.2
    a = 15 * x
    b = 2 * x - 1
    c = .5
    return smin(smax(b, c, k), a, k)

def remap(value, fromMin, fromMax, toMin, toMax):
    return toMin + (value - fromMin) / (fromMax - fromMin) * (toMax - toMin)