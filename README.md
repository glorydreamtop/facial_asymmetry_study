# 面部不对称数据集与代码

## 目录结构

```
facial_asymmetry_study_v2/
├── data/
│   ├── annotations/
│   │   └── annotations_usable.json   # 1030 例训练标注（COCO 关键点格式）
│   ├── raw/                           # 训练图片（8 来源，1030 张）
│   ├── splits/                        # train/val/test 分割
│   └── 100例原始数据/                 # Part 1-4 管线输入（图片 + 关键点 + 对称轴 + VAS）
│       ├── 1-50关键点/
│       ├── 1-50补对称轴/
│       ├── 51-100关键点/
│       ├── 51-100对称轴/
│       └── 专家VAS评分.xlsx
├── checkpoints/
│   └── hrnet_w48_29pts_coord010_best.pth  # HRNet-W48 预训练权重
├── models/
│   └── axis_regression/              # 对称轴线性回归模型
│       ├── axis_ridge.joblib
│       └── axis_ridge_equation.json
├── pipelines/
│   └── hrnet_w48/                    # HRNet 训练/评估管线（独立可运行）
│       ├── configs/
│       ├── hrnet_w48_pipeline/
│       └── scripts/
├── inputs/                           # 输入数据模板
├── src/                              # 核心工具包
└── scripts/
    ├── 01_filter_raw.py              # 原始数据过滤
    ├── 02_preprocess.py              # COCO 标注生成
    ├── 06_pipeline.py                # Part 1-4 完整流程（主管线）
    └── pipelines/                    # HRNet 管线副本

## 命名约定

- **Part 2** 使用原始区域 RAS 值（Global FA Index (raw)）
- **Part 3 / Part 4** 使用 **Machine FA Score (calibrated)**（经回归校准后的 0–100 分）
- **Expert VAS** 为评分者的均分
