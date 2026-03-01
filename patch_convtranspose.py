import argparse
import onnx

def get_attr(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return attr
    return None

def get_attr_ints(node, name):
    attr = get_attr(node, name)
    if attr is None:
        return None
    return list(attr.ints)

def set_attr(node, name, value):
    attr = get_attr(node, name)
    if attr is not None:
        node.attribute.remove(attr)
    node.attribute.append(onnx.helper.make_attribute(name, value))

def del_attr(node, name):
    attr = get_attr(node, name)
    if attr is not None:
        node.attribute.remove(attr)

def get_initializer_shape(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None

def infer_kernel_shape(node, model):
    ks = get_attr_ints(node, 'kernel_shape')
    if ks:
        return ks
    if len(node.input) < 2:
        return None
    w_shape = get_initializer_shape(model, node.input[1])
    if w_shape and len(w_shape) >= 4:
        return w_shape[-2:]
    return None

def infer_strides(node, rank):
    strides = get_attr_ints(node, 'strides')
    if strides:
        return strides
    return [1] * rank

def infer_dilations(node, rank):
    dilations = get_attr_ints(node, 'dilations')
    if dilations:
        return dilations
    return [1] * rank

def patch_convtranspose(model):
    changed = 0
    for node in model.graph.node:
        if node.op_type != 'ConvTranspose':
            continue
        kernel = infer_kernel_shape(node, model)
        if not kernel:
            continue
        rank = len(kernel)
        strides = infer_strides(node, rank)
        dilations = infer_dilations(node, rank)
        pads_begin = []
        pads_end = []
        for k, d in zip(kernel, dilations):
            k_eff = (k - 1) * d + 1
            p = k_eff // 2
            pads_begin.append(p)
            pads_end.append(p)
        pads = pads_begin + pads_end
        output_padding = []
        for s in strides:
            op = s - 1 if s > 1 else 0
            output_padding.append(op)
        set_attr(node, 'auto_pad', 'NOTSET')
        set_attr(node, 'pads', pads)
        set_attr(node, 'output_padding', output_padding)
        del_attr(node, 'output_shape')
        changed += 1
    return changed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input onnx path', default='./edgeflownet_384_512.onnx')
    parser.add_argument('output', help='output onnx path', default='./output_padding/edgeflownet_384_512.onnx')
    args = parser.parse_args()
    model = onnx.load(args.input)
    changed = patch_convtranspose(model)
    onnx.save(model, args.output)
    print(f'patched ConvTranspose: {changed}')
if __name__ == '__main__':
    main()