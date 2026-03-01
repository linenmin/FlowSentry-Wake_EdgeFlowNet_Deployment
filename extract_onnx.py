import argparse
import os
import sys
from typing import Optional
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(ROOT_DIR, 'code')
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
try:
    from network.MultiScaleResNet import MultiScaleResNet
    from misc.utils import AccumPreds
except ImportError as exc:
    raise SystemExit(f'Failed to import EdgeFlowNet code: {exc}')

def resolve_checkpoint(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isdir(path):
        return tf.train.latest_checkpoint(path)
    if os.path.exists(path):
        return path
    if os.path.exists(path + '.index'):
        return path
    return None

def detect_checkpoint_num_out(checkpoint_path):
    ckpt = resolve_checkpoint(checkpoint_path)
    if not ckpt:
        return None
    tf.compat.v1.reset_default_graph()
    try:
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
        var_names = reader.get_variable_to_shape_map()
        target_var = None
        for var_name in var_names:
            if 'ConvTranspose45' in var_name and 'bias' in var_name.lower():
                target_var = var_name
                break
        if target_var:
            bias_shape = reader.get_variable_to_shape_map()[target_var]
            if len(bias_shape) == 1:
                num_out = bias_shape[0]
                print(f'[检测] 从检查点检测到 NumOut = {num_out} (来自 {target_var})')
                return num_out
    except Exception as e:
        print(f'[检测] 检测 NumOut 时出错: {e}')
    print('[检测] 无法从检查点检测 NumOut，使用默认值')
    return None

def build_graph(args, checkpoint_num_out=None):
    tf.compat.v1.reset_default_graph()
    actual_num_out = checkpoint_num_out if checkpoint_num_out is not None else args.num_out
    if checkpoint_num_out is not None and checkpoint_num_out != args.num_out:
        print(f'[构建] 使用检查点的 NumOut={checkpoint_num_out} (而非命令行参数 {args.num_out})')
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, args.height, args.width, args.channels * 2], name='input')
    model = MultiScaleResNet(InputPH=input_ph, Padding=args.padding, NumOut=actual_num_out, InitNeurons=args.init_neurons, ExpansionFactor=args.expansion_factor, NumSubBlocks=args.num_sub_blocks, NumBlocks=args.num_blocks, Suffix='', UncType=None)
    multi_scale_outputs = model.Network()
    accum_output, _ = AccumPreds(multi_scale_outputs)
    output = accum_output[..., :args.output_channels]
    output = tf.identity(output, name='output')
    return (input_ph, output)

def freeze_graph(sess, output_tensor):
    output_node_name = output_tensor.name.split(':')[0]
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, [output_node_name])
    return frozen_graph_def

def _get_attr_ints(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None

def _get_dims(vi):
    return [d.dim_value if d.dim_value else d.dim_param or None for d in vi.type.tensor_type.shape.dim]

def _set_dims(vi, dims):
    shape = vi.type.tensor_type.shape
    del shape.dim[:]
    for val in dims:
        dim = shape.dim.add()
        if isinstance(val, int):
            dim.dim_value = val
        elif val is not None:
            dim.dim_param = str(val)

def _reorder_nhwc_to_nchw(dims):
    if len(dims) != 4:
        return dims
    return [dims[0], dims[3], dims[1], dims[2]]

def strip_io_transposes(model, input_height, input_width, output_channels):
    graph = model.graph
    input_name = graph.input[0].name
    output_name = graph.output[0].name
    output_to_node = {}
    for node in graph.node:
        for out_name in node.output:
            output_to_node[out_name] = node

    def _resolve_identity_chain(name):
        identities = []
        node = output_to_node.get(name)
        while node is not None and node.op_type == 'Identity':
            identities.append(node)
            if not node.input:
                break
            name = node.input[0]
            node = output_to_node.get(name)
        return (name, node, identities)
    input_transpose = None
    for node in graph.node:
        if node.op_type != 'Transpose':
            continue
        perm = _get_attr_ints(node, 'perm')
        if perm == [0, 3, 1, 2] and node.input and (node.input[0] == input_name):
            input_transpose = node
            break
    if input_transpose:
        trans_out = input_transpose.output[0]
        for node in graph.node:
            for idx, name in enumerate(node.input):
                if name == trans_out:
                    node.input[idx] = input_name
        graph.node.remove(input_transpose)
        in_dims = _get_dims(graph.input[0])
        _set_dims(graph.input[0], _reorder_nhwc_to_nchw(in_dims))
    orig_out_dims = _get_dims(graph.output[0])
    resolved_name, resolved_node, identity_nodes = _resolve_identity_chain(output_name)
    output_transpose = None
    if resolved_node is not None and resolved_node.op_type == 'Transpose':
        perm = _get_attr_ints(resolved_node, 'perm')
        if perm == [0, 2, 3, 1]:
            output_transpose = resolved_node
    if output_transpose:
        new_output = output_transpose.input[0]
        trans_out = output_transpose.output[0]
        for node in graph.node:
            for idx, name in enumerate(node.input):
                if name == trans_out:
                    node.input[idx] = new_output
        graph.output[0].name = new_output
        out_dims = None
        for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
            if vi.name == new_output:
                out_dims = _get_dims(vi)
                break
        if out_dims:
            _set_dims(graph.output[0], out_dims)
        else:
            _set_dims(graph.output[0], _reorder_nhwc_to_nchw(orig_out_dims))
        graph.node.remove(output_transpose)
        for node in identity_nodes:
            if node in graph.node:
                graph.node.remove(node)
    if len(graph.output) > 0:
        out_dims = _get_dims(graph.output[0])
        if not out_dims or any((dim is None for dim in out_dims)) or (len(out_dims) == 4 and out_dims[1] == input_height and (out_dims[2] == input_width) and (out_dims[3] == output_channels)):
            _set_dims(graph.output[0], [1, output_channels, input_height, input_width])
    return model

def _get_initializer_shape(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None

def _get_shape_from_value_info(model, name):
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == name:
            return _get_dims(vi)
    return None

def convert_auto_pad_to_explicit_pads(model):
    import math
    import onnx
    from onnx import shape_inference
    model = shape_inference.infer_shapes(model)
    for node in model.graph.node:
        if node.op_type != 'Conv':
            continue
        auto_pad = None
        for attr in node.attribute:
            if attr.name == 'auto_pad' and attr.s:
                auto_pad = attr.s.decode('utf-8')
        if auto_pad is None or auto_pad == 'NOTSET':
            continue
        in_shape = _get_shape_from_value_info(model, node.input[0])
        if not in_shape or len(in_shape) < 4 or any((dim is None for dim in in_shape[0:4])):
            continue
        kernel_shape = None
        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                kernel_shape = list(attr.ints)
                break
        if kernel_shape is None:
            weight_shape = _get_initializer_shape(model, node.input[1])
            if weight_shape and len(weight_shape) >= 4:
                kernel_shape = weight_shape[-2:]
        if not kernel_shape or len(kernel_shape) != 2:
            continue
        strides = None
        dilations = None
        for attr in node.attribute:
            if attr.name == 'strides':
                strides = list(attr.ints)
            elif attr.name == 'dilations':
                dilations = list(attr.ints)
        if not strides:
            strides = [1, 1]
        if not dilations:
            dilations = [1, 1]
        in_h, in_w = (in_shape[2], in_shape[3])
        k_h, k_w = kernel_shape
        s_h, s_w = strides
        d_h, d_w = dilations
        if auto_pad == 'VALID':
            pads = [0, 0, 0, 0]
        else:
            out_h = int(math.ceil(float(in_h) / float(s_h)))
            out_w = int(math.ceil(float(in_w) / float(s_w)))
            pad_h_total = max((out_h - 1) * s_h + (k_h - 1) * d_h + 1 - in_h, 0)
            pad_w_total = max((out_w - 1) * s_w + (k_w - 1) * d_w + 1 - in_w, 0)
            if auto_pad == 'SAME_LOWER':
                pad_top = int(math.ceil(pad_h_total / 2.0))
                pad_left = int(math.ceil(pad_w_total / 2.0))
            else:
                pad_top = int(math.floor(pad_h_total / 2.0))
                pad_left = int(math.floor(pad_w_total / 2.0))
            pad_bottom = pad_h_total - pad_top
            pad_right = pad_w_total - pad_left
            pads = [pad_top, pad_left, pad_bottom, pad_right]
        new_attrs = []
        for attr in node.attribute:
            if attr.name in ('auto_pad', 'pads'):
                continue
            new_attrs.append(attr)
        new_attrs.append(onnx.helper.make_attribute('pads', pads))
        new_attrs.append(onnx.helper.make_attribute('auto_pad', 'NOTSET'))
        node.attribute[:] = new_attrs
    return model

def convert_to_onnx(args, input_tensor, output_tensor, frozen_graph_def):
    import tf2onnx
    onnx_model, _ = tf2onnx.convert.from_graph_def(frozen_graph_def, input_names=[input_tensor.name], output_names=[output_tensor.name], opset=args.opset)
    if args.export_nchw:
        onnx_model = strip_io_transposes(onnx_model, args.height, args.width, args.output_channels)
    if args.force_explicit_pads:
        onnx_model = convert_auto_pad_to_explicit_pads(onnx_model)
    import onnx
    onnx.save(onnx_model, args.output)

def verify_onnx(args):
    try:
        import onnx
        import onnxruntime as ort
    except Exception as exc:
        print(f'ONNX verification skipped: {exc}')
        return
    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    sess = ort.InferenceSession(args.output)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    if args.export_nchw:
        test_input = np.random.randn(1, args.channels * 2, args.height, args.width).astype(np.float32)
    else:
        test_input = np.random.randn(1, args.height, args.width, args.channels * 2).astype(np.float32)
    result = sess.run([output_name], {input_name: test_input})[0]
    print(f'ONNX output shape: {result.shape}')

def main():
    parser = argparse.ArgumentParser(description='Export EdgeFlowNet (transpose conv) to ONNX')
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--channels', type=int, default=3, help='Channels per frame')
    parser.add_argument('--num-out', type=int, default=2)
    parser.add_argument('--output-channels', type=int, default=2)
    parser.add_argument('--init-neurons', type=int, default=32)
    parser.add_argument('--expansion-factor', type=float, default=2.0)
    parser.add_argument('--num-sub-blocks', type=int, default=2)
    parser.add_argument('--num-blocks', type=int, default=1)
    parser.add_argument('--padding', default='same')
    parser.add_argument('--checkpoint', default=os.path.join(ROOT_DIR, 'checkpoints', 'best.ckpt'))
    parser.add_argument('--output', default='edgeflownet_384_512.onnx')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--export-nchw', action='store_true', default=True)
    parser.add_argument('--no-export-nchw', dest='export_nchw', action='store_false')
    parser.add_argument('--force-explicit-pads', action='store_true', default=True)
    parser.add_argument('--no-force-explicit-pads', dest='force_explicit_pads', action='store_false')
    parser.add_argument('--verify', action='store_true', default=False)
    args = parser.parse_args()
    checkpoint_num_out = detect_checkpoint_num_out(args.checkpoint)
    actual_num_out = checkpoint_num_out if checkpoint_num_out is not None else args.num_out
    if args.output_channels > actual_num_out:
        raise SystemExit(f'output-channels ({args.output_channels}) must be <= num-out ({actual_num_out})')
    input_tensor, output_tensor = build_graph(args, checkpoint_num_out)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        ckpt = resolve_checkpoint(args.checkpoint)
        if ckpt:
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, ckpt)
            print(f'Loaded checkpoint: {ckpt}')
        else:
            print('Checkpoint not found; using random weights.')
        frozen_graph_def = freeze_graph(sess, output_tensor)
    convert_to_onnx(args, input_tensor, output_tensor, frozen_graph_def)
    print(f'Saved ONNX model to: {args.output}')
    if args.verify:
        verify_onnx(args)
if __name__ == '__main__':
    main()