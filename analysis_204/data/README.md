# 204 分析输入

本目录不附带数据。运行 `analysis_204` 前，请放入以下输入表：

- `expert_landmarks_204.npz`：`names` (204,), `landmarks` (204,29,2)
- `automatic_landmarks_204.npz`：`names` (204,), `landmarks` (204,29,2)
- `axes_204.json`：每个病例的 `A1/A2/B1/B2/consensus` → `[m, b]`
- `vas_204.csv`：`stem, split, vas, vas_a, vas_b`
- `split_204.json`：`train` / `val` / `test`
- `feature_names_167.json`：167 个特征名

格式说明见 `analysis_204/README.md`。
