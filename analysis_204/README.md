# 面部不对称自动评分：204 例分析代码

本目录从 204 例输入表生成论文表、图和报告。输入表放在 `data/`，输出写到 `outputs/`。

## 目录

```
analysis_204/
├── data/                        # 输入表（本包不附带数据）
├── predict_landmarks.py         # 可选：用 HRNet 推理生成自动关键点
├── common.py                    # 几何、RAS、统计
├── axis_features.py             # 167 维特征与四个轴回归模型
├── 01_prepare.py                # 组装 cohort.pkl
├── 02_axis_models.py            # 四模型训练/验证/测试
├── 03_global_fa.py              # Global FA 公式、A/B/C 三条路线
├── 04_tables.py                 # Excel、图、结果报告
├── 05_docx.py                   # Word 论文表与图
├── run_all.sh
└── outputs/                     # 运行后生成
```

## 输入

放入 `data/`：

- `expert_landmarks_204.npz`：`names` (204,), `landmarks` (204,29,2)
- `automatic_landmarks_204.npz`：`names` (204,), `landmarks` (204,29,2)
- `axes_204.json`：`A1/A2/B1/B2/consensus` → `[m, b]`
- `vas_204.csv`：`stem, split, vas, vas_a, vas_b`
- `split_204.json`：`train` / `val` / `test`
- `feature_names_167.json`：167 个特征名

自动关键点可由 `predict_landmarks.py` 用 HRNet 生成；训练代码见 `../pipelines/hrnet_w48`。

## 运行

```bash
cd analysis_204
python -m pip install -r ../requirements.txt

# 可选：生成自动关键点
python predict_landmarks.py \
  --image_dir ../对称轴数据/1.对称轴133张 \
  --checkpoint ../checkpoints/hrnet_w48_29pts_coord010_best.pth \
  --config ../pipelines/hrnet_w48/configs/cuda_hrnet_w48_384_29pts_coord010.yaml \
  --output data/automatic_landmarks_204.npz

bash run_all.sh
```

输出：`outputs/` 下的 Excel、Word、结果报告、逐例 CSV、图。

## 口径

- 标注 216 例，双侧关键点完整纳入 204 例；12 例排除。
- 训练 143 / 验证 41 / 测试 20（7:2:1，seed=11）。
- 轴模型：Ridge（α=1000）、Linear、SVR-RBF（C=10, ε=0.01）、RF（500 棵，leaf=2）。
- 模型选择：训练+验证集 5 折 CV 平均绝对角度偏差；测试集只评估一次。
- Global FA：训练+验证集 OLS，截断 0–100；raw z 保留并报告越界。
- A：自动点+预测轴；B：专家点+预测轴；C：专家点+共识轴。
- 评分时轴线以双瞳中点为锚点做横向平移；轴误差评价使用原始轴线。
