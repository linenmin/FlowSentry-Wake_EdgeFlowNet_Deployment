<div align="center">

# FlowSentry-Wake EdgeFlowNet Deployment

<img src="./image/FlowSentry-Wake.jpg" alt="FlowSentry-Wake" width="600"/>

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.17.0-005CED?logo=onnx)](https://onnx.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**[English](#english) | [中文](#chinese)**

Deploying EdgeFlowNet to Orange Pi 5 Plus + Axelera Metis M.2 NPU via Voyager SDK.
<br>将 EdgeFlowNet 部署至 Orange Pi 5 Plus 搭载 Axelera Metis M.2 异构计算平台。

</div>

---

<a id="english"></a>
## 🇬🇧 English

### 📌 Introduction

This repository provides a clean, deployment-focused pipeline to export and patch the official **EdgeFlowNet** model for ONNX inference. It is specifically designed to target heterogeneous hardware: **Orange Pi 5 Plus paired with an Axelera Metis M.2 NPU accelerator**.

> **⚠️ CRITICAL LIMITATION**: All scripts in this repository (`extract_onnx.py`, `patch_convtranspose.py`, `run_inference_pipeline.py`) **MUST BE EXECUTED ON A HOST MACHINE (e.g., Windows/Linux PC)**. The target edge device is only used for the final compiled SDK execution.

### 🔗 Companion Repository

This repository acts as the upstream ONNX exporter. The exported and patched ONNX models are intended to be deployed using the companion system repository:
👉 **[FlowSentry-Wake](https://github.com/mm0806son/FlowSentry-Wake)**

### 🎥 Demo Video

Watch the real-time optical flow estimation running on the target device:
**[Demo Video Link](https://www.youtube.com/watch?v=MPbFn8jpanw)**

### 📂 Directory Structure

```text
FlowSentry-Wake_EdgeFlowNet_Deployment/
├── checkpoints/                 # Pre-trained EdgeFlowNet weights (from the official EdgeFlowNet repository)
├── code/                        # Cleaned model architecture source code (No training scripts)
├── image/                       # Cover images and assets
├── extract_onnx.py              # Step 1: Export TensorFlow checkpoint to ONNX
├── patch_convtranspose.py       # Step 2: Fix dynamic tensor padding for ONNX ConvTranspose nodes
├── run_inference_pipeline.py    # Complete Host-side pipeline (Export -> Patch -> Test Inference)
├── requirements.txt             # Clean conda dependencies for host machine
└── README.md                    # This file
```

### ⚙️ Environment Setup (Host Machine)

Clone the repository and create a dedicated `conda` environment on your host machine to run the export scripts:

```bash
git clone https://github.com/linenmin/FlowSentry-Wake_EdgeFlowNet_Deployment
cd FlowSentry-Wake_EdgeFlowNet_Deployment

conda create -n vela python=3.10 -y
conda activate vela
pip install -r requirements.txt
```

### 🛠️ Execution Pipeline (Host Machine)

Instead of running steps individually, you can validate the full export-and-test loop on your host CPU using the provided pipeline script. You can use your own recorded dynamic video for testing:

```bash
python run_inference_pipeline.py \
    --video "path/to/your/own_video.mp4" \
    --height 576 \
    --width 1024
```

This script will:
1. Export `"edgeflownet_576_1024.onnx"`.
2. Patch it and save it to `output_padding/edgeflownet_576_1024.onnx`.
3. Run standard ONNXRuntime inference on the video and generate `your_video_flow_vis_test.mp4`.

### 🚀 Edge Deployment (Orange Pi + Axelera)

Once you have verified the patched model (`output_padding/edgeflownet_576_1024.onnx`) on your host, follow these steps to deploy:

1. **Transfer the Model**: Copy the patched ONNX model to your Orange Pi running the **Axelera Voyager SDK**. Place it in the required directory for the `FlowSentry-Wake` repo (e.g., `~/FlowSentry-Wake/ax_models/custom/`).
2. **Compile and Run**: On the Orange Pi, navigate to the `FlowSentry-Wake` workspace and execute the deployment script.

```bash
# Executed ON the Orange Pi
./deploy.py edgeflownet-opticalflow
```
This triggers the Axios compiler to slice the graph and compile `.hef` binaries for the Metis NPU.

### 📜 Acknowledgements & License

This project is licensed under the MIT License. See the `LICENSE` file for details.
We sincerely thank the following open-source projects:
* **EdgeFlowNet**: [https://github.com/pearwpi/EdgeFlowNet](https://github.com/pearwpi/EdgeFlowNet)
* **Axelera Official Voyager SDK (v1.5)**: [https://github.com/axelera-ai-hub/voyager-sdk/tree/release/v1.5](https://github.com/axelera-ai-hub/voyager-sdk/tree/release/v1.5)

---

<br>

<a id="chinese"></a>
## 🇨🇳 中文

### 📌 引言

本仓库提供了一条纯净的、专注于部署的流水线，用于将官方的 **EdgeFlowNet** 模型导出并修补为 ONNX 格式。本工程专门面向异构硬件目标平台优化：**Orange Pi 5 Plus 开发板 搭配 Axelera Metis M.2 NPU 加速卡**。

> **⚠️ 关键限制与注意**: 本仓库内的所有脚本（`extract_onnx.py`、`patch_convtranspose.py`、`run_inference_pipeline.py`）**必须且只能在上位机（如 Windows 或 Linux 电脑主机）运行**。边缘板端仅用于最终 SDK 编译和运行环节。

### 🔗 关联仓库

本仓库作为上游 ONNX 模型导出器存在。导出并经过修补的 ONNX 模型最终应配合下游系统仓库进行板端部署：
👉 **[FlowSentry-Wake](https://github.com/mm0806son/FlowSentry-Wake)**

### 🎥 演示视频

查看在目标异构设备上运行光流估计的实际演示视频：
**[Demo 演示视频链接](https://www.youtube.com/watch?v=MPbFn8jpanw)**

### 📂 目录结构

```text
FlowSentry-Wake_EdgeFlowNet_Deployment/
├── checkpoints/                 # EdgeFlowNet 预训练权重 (由原 EdgeFlowNet 官方网络训练所得)
├── code/                        # 剔除注释的纯净模型架构源码 (不含训练脚本)
├── image/                       # 文档与产品主图资源
├── extract_onnx.py              # 第一步：将 TensorFlow 权重导出为 ONNX
├── patch_convtranspose.py       # 第二步：修复 ONNX 中 ConvTranspose 节点的动态 padding 维度
├── run_inference_pipeline.py    # 完整的主机侧流水线 (包含 导出 -> 修补 -> 推理验证)
├── requirements.txt             # 主机侧基于 vela 环境剥离的精简依赖
└── README.md                    # 本文件
```

### ⚙️ 环境配置 (上位主机执行)

克隆本仓库，并在您的电脑主机上创建一个纯净的 `conda` 虚拟环境以运行导出脚本：

```bash
git clone https://github.com/linenmin/FlowSentry-Wake_EdgeFlowNet_Deployment
cd FlowSentry-Wake_EdgeFlowNet_Deployment

conda create -n vela python=3.10 -y
conda activate vela
pip install -r requirements.txt
```

### 🛠️ 工作流执行 (上位主机执行)

您无需手动分步操作，可直接运行提供的流水线脚本。它将在您的上位机 CPU 上进行端到端的提取与验证测试。您可以使用自己录制的任意动态视频进行输入测试：

```bash
python run_inference_pipeline.py \
    --video "path/to/your/own_video.mp4" \
    --height 576 \
    --width 1024
```

该脚本将自动执行以下操作：
1. 导出原始的 `"edgeflownet_576_1024.onnx"` 模型。
2. 对齐模型进行修补处理，生成存放于 `output_padding/edgeflownet_576_1024.onnx` 的适用目标编译器的版本。
3. 调用主机 `onnxruntime` 对输入的视频执行完整光流计算，并输出渲染视频 `your_video_flow_vis_test.mp4` 用于检视精度。

### 🚀 边缘部署流程 (Orange Pi + Axelera)

在上位机确认 `output_padding/` 下修补后的 ONNX 模型精度无误后，即可进入边缘硬件部署流程：

1. **模型传输**：将修补后的 `.onnx` 模型上传到装有 **Axelera Voyager SDK** 的 Orange Pi 上，放入配套仓库 `FlowSentry-Wake` 中要求的目录（例如 `~/FlowSentry-Wake/ax_models/custom/`）。
2. **编译与运行**：登录 Orange Pi，进入 `FlowSentry-Wake` 工作区，使用官方部署工具打包。

```bash
# 于 Orange Pi 终端内执行
./deploy.py edgeflownet-opticalflow
```
此命令将触发底层 Axios 编译器进行算子分割、降标与重构，最终将模型编入 Metis NPU 专用 `.hef` 可执行镜像并启动应用。

### 📜 版权、许可证与致谢

本项目遵循 MIT 开源许可证。详见 `LICENSE` 文件声明。
诚挚感谢以下两项优秀的开源系统为本级部署方案提供的底层支持：
* 原版 **EdgeFlowNet** 算法：[https://github.com/pearwpi/EdgeFlowNet](https://github.com/pearwpi/EdgeFlowNet)
* **Axelera 原生 Voyager SDK (v1.5)**：[https://github.com/axelera-ai-hub/voyager-sdk/tree/release/v1.5](https://github.com/axelera-ai-hub/voyager-sdk/tree/release/v1.5)
