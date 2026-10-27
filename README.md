# 面部不对称自动评分：完整工程代码

## 目录结构

```
.
├── src/                         # 核心库：数据、关键点、对称轴、指标、可视化
├── pipelines/hrnet_w48/         # HRNet-W48 训练/评估/轴回归管线
│   ├── configs/
│   ├── hrnet_w48_pipeline/
│   └── scripts/
├── scripts/                     # 数据过滤、预处理、MobileNet 训练/评估、283 主流程
├── 对比/                        # 关键点/对称轴四模型对比
│   ├── configs/
│   └── scripts/
├── configs/                     # 通用训练配置
├── analysis_204/                # 204 例分析：轴模型、Global FA、A/B/C、表图
│   └── data/                    # 仅放输入表（本包不附带数据）
├── requirements.txt
└── README.md
```

## 不包含的内容

- `data/raw/`、`data/annotations/`、`data/splits/`、`对称轴数据/`、`专家VAS评分/` 等原始或标注数据；
- `checkpoints/` 模型权重；
- `results/`、`figures/`、Excel/Word/PDF 结果文件；
- `analysis_204/data/` 下的输入表（格式见 `analysis_204/data/README.md`）；
- 非正式/临时脚本。

## 环境

```bash
python -m pip install -r requirements.txt
```

- HRNet 训练建议在 CUDA GPU 上运行；
- macOS 可用 MPS 做推理；
- 旧版 MobileNetV2 路线依赖 `onnxruntime` / `insightface`，仅作历史保留。

## 运行顺序

### 1. 原始数据过滤与预处理

```bash
python scripts/01_filter_raw.py \
  --source_dir 照片 \
  --target_dir data/raw

python scripts/02_preprocess.py \
  --raw_dir data/raw \
  --ann_dir data/annotations \
  --split_dir data/splits
```

### 2. HRNet-W48 训练与评估

```bash
python pipelines/hrnet_w48/scripts/01_train.py \
  --config pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml \
  --device cuda

python pipelines/hrnet_w48/scripts/02_evaluate.py \
  --config pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml \
  --checkpoint checkpoints/hrnet_w48_29pts_coord010_best.pth \
  --split_file data/splits/test.json \
  --device cuda \
  --output results/evaluation_hrnet_w48
```

更详细的 HRNet 说明见 `pipelines/hrnet_w48/README.md`。

### 3. 对称轴四模型对比（283 例路线）

```bash
python 对比/scripts/cache_pred_landmarks_283.py

python 对比/scripts/train_axis_compare_pred.py \
  --axis_dirs "对称轴数据/1.对称轴133张" "对称轴数据/2. 对称轴157张" \
  --model_type ridge \
  --config 对比/configs/hrnet_w48.yaml \
  --checkpoint checkpoints/hrnet_w48_29pts_coord010_best.pth \
  --split_file 对比/results/axis_final/split_manifest_seed11.json \
  --output 对比/results/axis_final/ridge
```

`--model_type` 可选 `linear` / `svr` / `rf` / `ridge`。脚本参数以 `--help` 为准。

### 4. 283 例主流程

```bash
python scripts/06b_pipeline_283.py
```

需要 `对称轴数据/`、`对比/results/axis_final/pred_landmarks_283.npz`、`对比/results/axis_final/专家VAS评分_283例.xlsx`。

### 5. 204 例分析

把输入表放入 `analysis_204/data/`（格式见该目录的 `README.md`）；自动关键点也可由 `analysis_204/predict_landmarks.py` 生成。然后：

```bash
cd analysis_204
bash run_all.sh
```

输出写入 `analysis_204/outputs/`：Excel 主表、Word 表图、结果报告、逐例 CSV、图。

## 204 分析的输入表

- `expert_landmarks_204.npz`：`names` (204,), `landmarks` (204,29,2)
- `automatic_landmarks_204.npz`：`names` (204,), `landmarks` (204,29,2)
- `axes_204.json`：`A1/A2/B1/B2/consensus` → `[m, b]`
- `vas_204.csv`：`stem, split, vas, vas_a, vas_b`
- `split_204.json`：`train` / `val` / `test`
- `feature_names_167.json`：167 个特征名

## 代码说明

- 关键点定义：`src/landmark/schema.py`（29 点、11 对双侧点、7 个中线点）。
- HRNet 热图管线：`pipelines/hrnet_w48/hrnet_w48_pipeline/`。
- 轴特征与回归模型：`src/symmetry/axis_regression.py`、`analysis_204/axis_features.py`。
- 区域 RAS / 统计：`src/symmetry/`、`analysis_204/common.py`。
- 283 主流程：`scripts/06b_pipeline_283.py`。
- 204 分析：`analysis_204/01_prepare.py` → `05_docx.py`。

所有脚本默认从本目录（仓库根）运行；`analysis_204` 从该子目录运行。
