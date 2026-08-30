# 面部不对称数据集与代码

## 目录结构

```
facial_asymmetry_study_v2/
├── data/
│   ├── annotations/
│   │   └── annotations_usable.json   # 1030 例训练标注（COCO 关键点格式）
│   ├── raw/                           # HRNet 训练图片（1030 张，已匿名化为 HRNet_*.jpg）
│   └── splits/                        # train/val/test 分割（721/206/103）
├── checkpoints/
│   └── hrnet_w48_29pts_coord010_best.pth  # HRNet-W48 预训练权重
├── pipelines/
│   └── hrnet_w48/                    # HRNet 训练/评估管线（独立可运行）
│       ├── configs/
│       ├── hrnet_w48_pipeline/
│       └── scripts/
├── 对比/                              # 对比模型研究（关键点 + 对称轴 4 模型对比）
│   ├── configs/                      # 对比模型训练配置
│   ├── scripts/                      # 对比训练/评估/报告脚本
│   └── results/                      # 最终报告 + 模型评估数据
│       ├── 最终版/                   # 对比模型最终报告（Excel + 图 + CSV）
│       ├── models/                   # 4 关键点模型原始评估数据
│       └── axis_final/               # 对称轴 4 模型结果 + 283 例 VAS 数据
├── inputs/                           # 输入数据模板（283 例 VAS + 共识轴）
├── src/                              # 核心工具包
├── scripts/
│   ├── 01_filter_raw.py              # 原始数据过滤
│   ├── 02_preprocess.py              # COCO 标注生成
│   ├── 03_train.py                   # HRNet 训练
│   ├── 04_evaluate.py                # HRNet 评估
│   ├── 05_symmetry.py                # 对称性分析
│   └── 06b_pipeline_283.py           # 283 例 Pred 路线主流程（Part 1-4）
├── results/                          # 283 例管线输出
│   ├── part1_axis_consensus/         # 专家共识轴 vs AI 轴对比（283 例）
│   ├── part2_region_fa/              # 5 区域 RAS + Global FA（283 例）
│   ├── part3_weights_shap/           # 回归权重 + SHAP 分析（283 例）
│   └── part4_evaluation/             # 机器评分 vs 专家 VAS 评估（283 例）
└── archive_100例/                    # 归档：历史 100 例数据（已弃用）
```

## 研究路线（Pred 纯 AI 自证）

- **关键点来源**：HRNet-W48 预测（对 283 张对称轴图推理）
- **对称轴**：专家标注（283 张）
- **VAS**：基于 HRNet 预测关键点计算 5 区域 RAS，按预期权重合成（Facial Contour 35% / Mouth 25% / Eye 15% / Nose 15% / Eyebrow 10%）
- **数据规模**：283 例全流程（Part 1-4）

## 命名约定

- **Part 2** 使用原始区域 RAS 值（Global FA Index (raw)）
- **Part 3 / Part 4** 使用 **Machine FA Score (calibrated)**（经回归校准后的 0–100 分）
- **Expert VAS** 为评分者的均分

## 主要流程

1. **关键点检测**：HRNet-W48（4 模型对比中最优，MRE 12.46px / NME 2.12%）
2. **对称轴预测**：以 HRNet 预测关键点为特征，Ridge 回归（4 模型对比中角度精度最优）
3. **区域不对称评分**：5 区域 RAS + Global FA（Part 2）
4. **权重分析**：回归 + SHAP（Part 3）
5. **模型评估**：机器 FA vs 专家 VAS（Part 4）

主管线：`python scripts/06b_pipeline_283.py`

## 交付说明

- `0825交付说明.md` — 最新交付内容说明
- `对比/results/README.md` — 对比模型研究报告说明
