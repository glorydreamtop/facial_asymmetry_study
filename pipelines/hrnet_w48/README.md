# HRNet-W48 Pipeline

这是一套与当前 `MobileNetV2` 工程并行存在的 `HRNet-W48 + Heatmap Regression` 新工程。

目标：

- 使用 `HRNet-W48` 高分辨率特征
- 将 `31` 点关键点回归改为 `31` 通道热图预测
- 面向后续远程 `CUDA` 服务器训练，而不是本机 `MPS`

## 技术要点

- Backbone: `timm` 的 `hrnet_w48`
- 输入尺寸: `384 x 384`
- 热图尺寸: `96 x 96`
- 监督方式: 高斯热图 + mask
- 主损失: masked heatmap MSE
- 辅助损失: 可微坐标解码后的 masked Wing loss
- 训练策略: warmup + cosine decay, early stopping, checkpoint resume
- 冻结策略: 前若干 epoch 冻结低层 backbone

## 目录结构

- `configs/cuda_hrnet_w48_384_29pts_coord010.yaml`: 推荐起始配置 (coord_weight=0.10)
- `requirements.txt`: 该管线额外依赖
- `hrnet_w48_pipeline/`: 训练/评估实现
- `scripts/01_train.py`: 训练入口
- `scripts/02_evaluate.py`: 评估入口
- `scripts/03_paper_figures.py`: 曲线与 best/worst 图导出

## 依赖安装

根目录基础依赖安装完成后，再安装本管线额外依赖：

```bash
source ".venv/bin/activate" && pip install -r pipelines/hrnet_w48/requirements.txt
```

## 推荐训练命令

```bash
source ".venv/bin/activate" && python pipelines/hrnet_w48/scripts/01_train.py --config pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml --device cuda --save_dir checkpoints/hrnet_w48_29pts_coord010 --log_dir logs/hrnet_w48_29pts_coord010
```

## 推荐评估命令

```bash
source ".venv/bin/activate" && python pipelines/hrnet_w48/scripts/02_evaluate.py --config pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml --checkpoint checkpoints/hrnet_w48_29pts_coord010_best.pth --output results/evaluation_hrnet_w48 --device cuda --split_file data/splits/test.json
```

## 推荐图表导出命令

```bash
source ".venv/bin/activate" && python pipelines/hrnet_w48/scripts/03_paper_figures.py --config pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml --checkpoint checkpoints/hrnet_w48_29pts_coord010_best.pth --log_dir logs/hrnet_w48_29pts_coord010 --summary results/evaluation_hrnet_w48/paper_metrics_summary.json --output results/evaluation_hrnet_w48 --device cuda
```

## 对称轴线性回归

新增了一条监督学习路线：

- 输入：当前 29 点模型输出的人脸关键点坐标
- 输出：人工标注的对称轴直线
- 模型：`Ridge` 线性回归

推荐命令：

```bash
source ".venv/bin/activate" && python pipelines/hrnet_w48/scripts/04_train_axis_regressor.py --config pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml --checkpoint checkpoints/hrnet_w48_29pts_coord010_best.pth --axis_dir "对称轴数据" --landmark_root data/raw --feature_source pred --output results/axis_regression --device cuda
```

如果想先看“人工点到对称轴”的上限基线，可以把 `--feature_source pred` 改成 `--feature_source gt`。

## 与当前主工程的关系

- 数据格式继续复用当前工程：`data/annotations/annotations_usable.json`、`data/splits/*.json`
- 关键点定义继续复用当前工程：`src/landmark/schema.py`
- 这是平行工程，不会覆盖现有 `MobileNetV2` 主链路
