# Claude 4.6 Experiment Handoff Prompt

Below is a prompt you can give directly to Claude 4.6.

---

你现在接手一个本地代码实验项目。你对项目没有任何先验记忆，必须先基于仓库现状建立上下文，再执行。你的角色不是写论文，而是把实验代码、环境、数据和 baseline 跑通，并为后续大规模实验建立可复用的工程基础。

## 你的工作目录

- Windows 仓库根目录：`C:\mamba`
- 本机 shell：PowerShell
- 主训练环境：`WSL2 Ubuntu-24.04`
- 主 GPU：RTX 5090
- 远端补充算力：`ssh FatMachine`，远端也有 `WSL2 Ubuntu-24.04 + RTX 5090 D v2`

## 项目一句话

这是一个 BMVC 风格的检测项目，研究一个轻量频率解耦的 Hybrid Vision Mamba backbone：让 Mamba 处理低频全局信息，再用一个极轻高频细节分支补偿边缘、纹理和微小目标细节，场景锁定为航拍/无人机 tiny object detection。

GitHub-facing 项目名已经固定为：

`HybridMambaDet: Frequency-Decoupled Lightweight Mamba for Aerial Tiny Object Detection`

## 你必须先读的文件

按顺序阅读，不要跳过：

1. `C:\mamba\README.md`
2. `C:\mamba\.codex\RULES.md`
3. `C:\mamba\docs\status.md`
4. `C:\mamba\docs\roadmap.md`
5. `C:\mamba\docs\phases\phase-01-foundation-and-baseline.md`
6. `C:\mamba\docs\folder-map.md`
7. `C:\mamba\docs\data-preparation.md`
8. `C:\mamba\docs\experiment-matrix.md`
9. `C:\mamba\code\tinyvim\detection\README_hybrid.md`

然后阅读这些关键代码和脚本：

1. `C:\mamba\code\tinyvim\detection\model\hybridmamba.py`
2. `C:\mamba\code\tinyvim\detection\model\tinyvim.py`
3. `C:\mamba\scripts\setup_wsl_env.sh`
4. `C:\mamba\scripts\smoke_detection.py`
5. `C:\mamba\scripts\preflight_detection.py`
6. `C:\mamba\scripts\preflight_all_configs.py`
7. `C:\mamba\scripts\convert_visdrone_to_coco.py`
8. `C:\mamba\scripts\convert_aitod_to_coco.py`
9. `C:\mamba\scripts\convert_dota_hbb.py`

## 当前仓库状态

以下事实已经成立，不要重复设计：

- 主代码底座已经选定为 `TinyViM`，位于 `C:\mamba\code\tinyvim`
- 已新增 backbone：
  - `HybridMamba_Base_B`
  - `HybridMambaDet_B`
- 已新增 RetinaNet 配置：
  - `retinanet_tinyvim_b_fpn_1x_visdrone.py`
  - `retinanet_hybridmamba_base_b_fpn_1x_visdrone.py`
  - `retinanet_hybridmambadet_b_fpn_1x_visdrone.py`
  - `retinanet_tinyvim_b_fpn_1x_aitodv2.py`
  - `retinanet_hybridmambadet_b_fpn_1x_aitodv2.py`
  - `retinanet_tinyvim_b_fpn_1x_dotahbb.py`
  - `retinanet_hybridmambadet_b_fpn_1x_dotahbb.py`
- 三个数据转换脚本已经写好，并通过了最小合成样本 smoke
- 预检脚本已经能指出缺失项
- 论文脚手架已经建好，但这不是你的主任务

## 当前真实阻塞

当前真正未完成的是：

1. WSL 里的正式训练环境还没有装起来
2. `weights/tinyvim/tinyvim_b_300e.pth` 还没落到 `C:\mamba\weights\tinyvim\`
3. `VisDrone / AI-TOD-v2 / DOTA` 的真实数据还没有落到 `C:\mamba\data\...`
4. 因为 1-3 未完成，正式 baseline 还没跑

## 你的第一阶段任务

你只需要先完成 Phase 1，不要擅自扩展到论文写作或大规模 sweep。

### Phase 1 目标

把仓库变成一个可复现训练工作区，并拿到第一个有效的 `TinyViM + RetinaNet` 在 `VisDrone` 上的 smoke / baseline 结果。

### 你需要完成的具体任务

1. 处理 WSL 环境
   - 进入 `WSL2 Ubuntu-24.04`
   - 让 `scripts/setup_wsl_env.sh` 真正跑通，或者在必要时做最小修补
   - 目标环境必须是兼容 `TinyViM + MMDetection 2.28 + mmcv-full + torch 2.0` 的 Python 3.10 栈
   - 如果 `micromamba` 下载失败，允许你改脚本，但不要擅自把框架升级到 MMDetection 3.x

2. 安装 `selective_scan_cuda`
   - 优先沿用 TinyViM/VMamba 的兼容方案
   - 如果需要额外脚本或说明，可以新增到 `scripts/` 或 `docs/`
   - 目标是让 `code/tinyvim/detection/model/tvimblock.py` 的 `import selective_scan_cuda` 可用

3. 准备 checkpoint
   - 找到或下载 `tinyvim_b_300e.pth`
   - 放到 `C:\mamba\weights\tinyvim\tinyvim_b_300e.pth`
   - 如果你找不到，明确记录阻塞，不要伪造

4. 准备 VisDrone 数据
   - 按 `docs/data-preparation.md` 规定落盘
   - 将 train/val 原始图像和 txt 标注放到 `data/visdrone/`
   - 用 `scripts/convert_visdrone_to_coco.py` 转出：
     - `data/converted/visdrone/annotations/train_coco.json`
     - `data/converted/visdrone/annotations/val_coco.json`
   - 用 `scripts/summarize_coco.py` 验证转换结果

5. 跑配置预检
   - 先跑 `scripts/preflight_detection.py` 针对：
     - `code/tinyvim/detection/configs/retinanet_tinyvim_b_fpn_1x_visdrone.py`
   - 再跑 `scripts/preflight_all_configs.py`
   - 目标是确认至少 VisDrone baseline config 不再缺 checkpoint 和 annotation

6. 跑 detector smoke
   - 使用 `scripts/smoke_detection.py`
   - 目标 config：`retinanet_tinyvim_b_fpn_1x_visdrone.py`
   - 至少完成 10 iter
   - 必须生成：
     - `RUN_MANIFEST.json`
     - `train.log`
     - `eval_metrics.json`

7. 如果 smoke 通过，再启动第一个 formal baseline
   - detector 保持 `RetinaNet + FPN`
   - backbone 保持 `TinyViM_B`
   - 不要改 detector head、loss、neck、assigner

## 你不能擅自改变的边界

这些是硬约束：

- 不做 detector 创新
- 不做 loss / label assignment 创新
- 不做 rotated detection 主线
- 不把 DOTA 改成 OBB 主任务
- 不升级到 MMDetection 3.x
- 不把项目主线从 `TinyViM` 换成 `MobileMamba`
- 不手改任何论文表格数字
- 不虚构 checkpoint、结果或引用

## 你可以做的合理修改

这些修改是允许的，只要你说明理由：

- 修补 `scripts/setup_wsl_env.sh`
- 修补 `scripts/run_local_training.sh`
- 修补 remote 启动脚本
- 修补 config 路径、数据路径、checkpoint 路径
- 为 `selective_scan_cuda` 安装补一个可复用脚本或文档
- 为 smoke / baseline 增加更稳妥的日志输出

## 文件管理要求

遵守这些目录纪律：

- `code/` 放主代码改动
- `external/` 只读参考仓
- `data/` 放 raw 和 converted 数据
- `weights/` 放 checkpoint
- `artifacts/` 放 run 输出
- `docs/` 放环境、状态和数据说明
- 临时验证只能放 `artifacts/tmp_validation/`
- 不要再产生任何类似 `CUsersadmin` 这种错误目录

## 输出和汇报要求

每次阶段性汇报必须给我这些信息：

1. 你读了哪些关键文件
2. 你改了哪些文件
3. 环境是否可用
4. checkpoint 是否到位
5. VisDrone 数据是否已转换完成
6. preflight 缺什么
7. smoke 是否通过
8. 如果失败，失败在什么位置，下一步最小修复是什么

## 你完成本轮任务的验收标准

本轮只有下面这些算完成：

- WSL 环境可用
- `selective_scan_cuda` 可导入
- `tinyvim_b_300e.pth` 到位
- VisDrone COCO annotations 到位
- `retinanet_tinyvim_b_fpn_1x_visdrone.py` preflight 通过
- 10-iter smoke 通过并产出 manifest/log/metrics

如果以上没有全部做到，不要说“实验已完成”，只能说“Phase 1 已推进到哪一步”。

## 建议执行顺序

1. 阅读仓库文档和规则
2. 检查 WSL 当前 Python 与包状态
3. 跑通环境安装
4. 处理 `selective_scan_cuda`
5. 落 checkpoint
6. 落 VisDrone 数据并转换
7. 预检
8. smoke
9. formal baseline

## 你开始工作前的第一句话建议

“我会先读项目状态、规则和 Phase 1 文档，然后检查 WSL 环境、checkpoint 和 VisDrone 数据三项阻塞，先把 baseline smoke 跑通。”

---

End of prompt.
