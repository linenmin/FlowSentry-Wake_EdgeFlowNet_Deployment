import tensorflow as tf

def shift_flow_tf(flow_xy):
    shiftted_flow = flow_xy + 50
    return shiftted_flow

def unshift_flow_tf(shiftted_flow):
    flow_xy = shiftted_flow - 50
    return flow_xy