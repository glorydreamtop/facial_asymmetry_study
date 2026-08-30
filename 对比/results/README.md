# 对比模型研究报告

本目录包含对称轴预测对比模型研究的**最终报告**（`最终版/`）。
基于 Pred 路线（HRNet 预测关键点 → 回归轴 → 与人工轴对比），283 张对称轴数据。

## 研究背景

- **关键点检测**：HRNet-W48（MRE 12.46px, NME 2.12%），4 模型对比中最优
- **对称轴预测**：以 HRNet-W48 预测关键点为特征（coords_geom，167 维），4 种回归方案对比
- **数据划分**：seed=11，train 198 + val 28 / test 57（70/10/20）
- **特征路线**：HRNet + 原始照片 → 预测关键点 → 回归轴 → 与人工轴对比

## 对称轴 4 模型最终结果（test 57 张）

| 模型 | 角度偏差 (°) Mean±SD | 角度 Median | 位置偏差 (px) | MHD (px) |
|---|---|---|---|---|
| **Ridge (alpha=1000)** | **0.36 ± 0.41** | **0.16** | **3.05** | **3.86** |
| RandomForest (ne=500) | 0.39 ± 0.37 | 0.29 | 3.00 | 3.80 |
| SVR-RBF (C=10) | 0.41 ± 0.41 | 0.28 | 3.03 | 3.97 |
| Linear (无正则) | 0.84 ± 0.60 | 0.65 | 6.78 | 7.78 |

**结论**：Ridge 在主要指标（角度 Mean、Median）上为最优；位置/MHD 与 RandomForest、SVR 基本持平；Linear 全面垫底（证实 L2 正则的必要性）。

## 文件结构

```
最终版/
├── 对比模型补充数据.xlsx   # 7 个工作表
│   ├── 表2 关键点总体     # 4 关键点模型 MRE/NME/SDR/AUC
│   ├── 表3 区域性能       # 5 区域 × 4 模型 MRE/NME/SDR
│   ├── 表4 逐点误差       # 29 关键点 × 4 模型 MRE + SDR@15
│   ├── 表5 对称轴         # 4 轴模型角度/位置/MHD（Ridge 最优）
│   ├── 表6 复杂度         # 参数量/GFLOPs/内存/耗时/FPS
│   ├── 表7 训练条件       # 划分/输入/损失/优化器/种子等
│   └── 数据量核对         # 样本量统计
├── figures/
│   ├── sdr_curves.png           # 关键点 SDR 曲线
│   ├── mre_boxplot.png          # 关键点 MRE 箱线图
│   ├── region_mre.png           # 区域 MRE 分组柱状图
│   ├── per_point_heatmap.png    # 逐点误差热图
│   ├── axis_angle_boxplot.png   # 轴角度偏差箱线图
│   └── axis_position_boxplot.png# 轴位置偏差箱线图
└── figures_data/                # 所有图的原始数据 CSV（可自行配色重绘）
    ├── fig1_sdr_curves.csv
    ├── fig2_mre_boxplot.csv     # 逐样本 MRE（412 行）
    ├── fig3_region_mre.csv
    ├── fig4_per_point_heatmap.csv
    ├── axis_Ridge.csv           # 逐样本角度/位置/MHD（57 行）
    ├── axis_Linear.csv
    ├── axis_SVR-RBF.csv
    └── axis_RandomForest.csv
```

## 其他目录

- `models/` — 4 个关键点模型的原始评估数据（paper_metrics_summary.json + paper_error_details.json）
- `axis_final/` — 对称轴 4 模型最终结果（joblib 模型 + 逐样本指标 + split_manifest_seed11.json）

## 模型权重状态

- **HRNet-W48 权重**：`../../checkpoints/hrnet_w48_29pts_coord010_best.pth`（主模型，可用）
- **对比模型权重**（HRNet-W18 / ResNet50-SB / MobileNetV3-S）：在远程 GPU 服务器上训练完成，因服务器实例已释放未能下载回传，权重不可用。但各模型在 103 张测试集上的完整评估结果已保存于 `models/`，报告、图、CSV 均不受影响

## 说明

- 所有图与 CSV 均基于同一份真实实验数据
- 若需更换绘图配色，直接使用 `figures_data/` 中的 CSV 即可
