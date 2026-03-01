import argparse
import os
import subprocess
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

def run_extract_onnx(height: int, width: int):
    print('\n' + '=' * 60)
    print(f'[1/3] 执行 extract_onnx.py (分辨率 {height}x{width}) ...')
    print('=' * 60)
    onnx_name = f'edgeflownet_{height}_{width}.onnx'
    ret = subprocess.run([sys.executable, str(SCRIPT_DIR / 'extract_onnx.py'), '--height', str(height), '--width', str(width), '--output', onnx_name], cwd=SCRIPT_DIR)
    if ret.returncode != 0:
        raise RuntimeError('extract_onnx.py 执行失败')
    print('[1/3] extract_onnx.py 完成\n')

def run_patch_convtranspose(height: int, width: int):
    print('=' * 60)
    print('[2/3] 执行 patch_convtranspose.py ...')
    print('=' * 60)
    onnx_name = f'edgeflownet_{height}_{width}.onnx'
    raw_onnx = SCRIPT_DIR / onnx_name
    out_dir = SCRIPT_DIR / 'output_padding'
    out_dir.mkdir(parents=True, exist_ok=True)
    patched_onnx = out_dir / onnx_name
    ret = subprocess.run([sys.executable, str(SCRIPT_DIR / 'patch_convtranspose.py'), str(raw_onnx), str(patched_onnx)], cwd=SCRIPT_DIR)
    if ret.returncode != 0:
        raise RuntimeError('patch_convtranspose.py 执行失败')
    print('[2/3] patch_convtranspose.py 完成\n')
    return patched_onnx

def flow_to_image(flow):
    import numpy as np
    UNKNOWN_FLOW_THRESH = 10000000.0

    def make_color_wheel():
        RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros([ncols, 3])
        col = 0
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
        col += RY
        colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
        colorwheel[col:col + YG, 1] = 255
        col += YG
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
        col += GC
        colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
        colorwheel[col:col + CB, 2] = 255
        col += CB
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
        col += BM
        colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
        colorwheel[col:col + MR, 0] = 255
        return colorwheel

    def compute_color(u, v):
        h, w = u.shape
        img = np.zeros([h, w, 3])
        nan_idx = np.isnan(u) | np.isnan(v)
        u = np.nan_to_num(u, nan=0.0)
        v = np.nan_to_num(v, nan=0.0)
        colorwheel = make_color_wheel()
        ncols = colorwheel.shape[0]
        rad = np.sqrt(u ** 2 + v ** 2)
        a = np.arctan2(-v, -u) / np.pi
        fk = (a + 1) / 2 * (ncols - 1) + 1
        k0 = np.floor(fk).astype(int)
        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0
        for i in range(3):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1
            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            col[~idx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nan_idx)))
        return img
    u = flow[:, :, 0].astype(np.float64)
    v = flow[:, :, 1].astype(np.float64)
    idx_unk = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH)
    u[idx_unk] = 0
    v[idx_unk] = 0
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, float(np.max(rad)))
    u = u / (maxrad + 1e-10)
    v = v / (maxrad + 1e-10)
    img = compute_color(u, v)
    idx = np.repeat(idx_unk[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    return np.uint8(img)

def run_video_inference(patched_onnx: Path, video_path: Path):
    print('=' * 60)
    print('[3/3] 视频光流推理与渲染 ...')
    print('=' * 60)
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError('请先 conda activate vela 并安装依赖: pip install onnxruntime opencv-python numpy') from e
    if not video_path.exists():
        raise FileNotFoundError(f'视频文件不存在: {video_path}')
    out_dir = video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    flow_vis_video = out_dir / f'{stem}_flow_vis_test.mp4'
    flow_raw_dir = out_dir / f'{stem}_flow_raw_test'
    flow_raw_dir.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(str(patched_onnx), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    _, _, model_h, model_w = [int(d) for d in input_shape]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(flow_vis_video), fourcc, fps, (model_w, model_h))
    ret_prev, frame_prev = cap.read()
    if not ret_prev:
        raise RuntimeError('无法读取视频第一帧')
    frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB)
    frame_prev = cv2.resize(frame_prev, (model_w, model_h))
    frame_idx = 0
    while True:
        ret_curr, frame_curr = cap.read()
        if not ret_curr:
            break
        frame_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2RGB)
        frame_curr = cv2.resize(frame_curr, (model_w, model_h))
        pair = np.concatenate([frame_prev, frame_curr], axis=2)
        pair = pair.astype(np.float32) / 255.0
        pair = np.transpose(pair, (2, 0, 1))
        pair = np.expand_dims(pair, axis=0)
        flow_out = sess.run(None, {input_name: pair})[0]
        flow = np.transpose(flow_out[0], (1, 2, 0))
        flow_vis = flow_to_image(flow)
        flow_vis_bgr = cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR)
        writer.write(flow_vis_bgr)
        np.save(flow_raw_dir / f'flow_{frame_idx:06d}.npy', flow)
        frame_prev = frame_curr
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f'  已处理 {frame_idx}/{total_frames} 帧')
    cap.release()
    writer.release()
    print(f'\n[3/3] 完成!')
    print(f'  光流渲染视频: {flow_vis_video}')
    print(f'  原始光流 npy: {flow_raw_dir}')
    print()

def main():
    parser = argparse.ArgumentParser(description='EdgeFlowNet 推理流水线')
    parser.add_argument('--video', type=str, default=str(SCRIPT_DIR / 'visualization' / 'video' / 'hand moving.mp4'), help='输入视频路径')
    parser.add_argument('--height', type=int, default=576, help='模型输入高度（默认 576）')
    parser.add_argument('--width', type=int, default=1024, help='模型输入宽度（默认 1024）')
    parser.add_argument('--skip-extract', action='store_true', help='跳过 extract_onnx（若 ONNX 已存在）')
    parser.add_argument('--skip-patch', action='store_true', help='跳过 patch_convtranspose（若已 patch 过）')
    args = parser.parse_args()
    video_path = Path(args.video).resolve()
    height, width = (args.height, args.width)
    try:
        if not args.skip_extract:
            run_extract_onnx(height, width)
        else:
            print('[跳过] extract_onnx.py\n')
        if not args.skip_patch:
            patched = run_patch_convtranspose(height, width)
        else:
            onnx_name = f'edgeflownet_{height}_{width}.onnx'
            patched = SCRIPT_DIR / 'output_padding' / onnx_name
            if not patched.exists():
                raise FileNotFoundError(f'未找到已 patch 的模型: {patched}')
        run_video_inference(patched, video_path)
    except Exception as e:
        print(f'\n错误: {e}')
        sys.exit(1)
if __name__ == '__main__':
    main()