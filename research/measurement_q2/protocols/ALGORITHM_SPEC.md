# PCU-DM-Field / DAS-FC Algorithm Specification

## Status

- 状态：`FROZEN_SPEC_EXECUTED_READY_TO_DRAFT_WITH_SCOPE_LIMITS`
- 冻结日期：2026-07-19
- 核心算法：**DAS-FC**（Domain-Adaptive Shrinkage Full-field Conformal）
- 基础位移估计器：非学习式支承感知刚体配准 + 分段结构场；作为共享测量前端，不再声称联合优化创新。
- 证据边界：物理合成真值用于绝对位移；公开真实扫描只用于表面一致性、跨设备差异和探索性失败排序，不用于位移精度或 conformal coverage 验证。

## Post-formal outcome ledger (does not rewrite the frozen protocol)

- DAS-FC 的已知域 coverage--width--interval-score 核心通过正式 v1；
- 原冻结多分数组合拒识器未优于 scale-only，作为阴性消融保留；
- scale-only 排序在 480 个全新 v2 测试扫描对上独立确认，成为最终拒识规则；
- 未见域 homoscedastic fallback 把整场经验覆盖从 0.275 提高到 0.829，但区间显著变宽且 interval score 变差，因此只能写成 coverage--efficiency 取舍和失败边界，不能写成无条件鲁棒性成功；
- 新 `rbf_kink` 变形族的 240 个盲测扫描对零失败、点估计 MAE 1.590 mm；fallback 将经验整场覆盖从 0.688 提高到 0.971，但 interval score 恶化 8.927 mm，结论仍为混合鲁棒性；
- 公开 3DPrintedShapes 无修改迁移中，Cascade-Strong 比 multiscale ICP 的平均表面不一致度高 0.316 mm（cluster bootstrap 95% CI +0.185 到 +0.476），作为阴性外部证据保留；
- 下文“Frozen success conditions”保留实验前原文，用于审计，不因结果而删除失败条件。

## Pilot-driven pivot

初始 `PCU-DM joint` 交替位姿—位移版本未通过开发门：在 30 个测试扫描对的 pilot 中，相对 `Cascade-Strong` 的已知域 interval score 平均恶化 12.224 mm，未见域也更差。该路线被保留为负结果/消融，不进入完整算法。

通过机制门的算法核是“四分割异方差学习 + 扫描对级整场校准”：

```text
tuning scan pairs      -> fit structural conditional scale
validation scan pairs  -> choose domain-wise shrinkage lambda
calibration scan pairs -> fit pair-max conformal quantile
test scan pairs        -> evaluate once; never refit
```

## Inputs and outputs

输入：参考点云 `P0`、受测点云 `P1`、几何分段 `g(x)`、支承候选 `c(x)`、可选已知域标签 `d`。

输出：

- 位移点估计 `u_hat(x)`；
- 原始局部尺度 `s_raw(x)` 与学习尺度 `s_ml(x)`；
- 域自适应收缩尺度 `s_das(x)`；
- 95% 扫描对整场同时区间；
- 扫描对级拒识分数和 `known / unseen / abstain` 状态；
- 位姿、匹配、支持概率和失败诊断。

## Shared measurement front-end

基础前端不使用真支承掩膜，只接收可能含漏检/误检的几何候选。它执行：

1. trimmed robust point-to-point pose estimation；
2. 各板/构件分段的结构基函数拟合，接缝处不共享系数；
3. 由局部拟合残差、最近匹配距离和支持概率形成 `s_raw(x)`；
4. 保留无输出、数值失败和低支持区域，不静默删除。

`Cascade-Strong` 与完整算法共享同一个点估计。这样正式主效应只能来自不确定性算法，不会由更强配准器偷换。

## Cross-fitted structural scale

在专用 tuning 扫描对上，以每个扫描对等总权重抽取场位置，拟合条件误差分位尺度：

```text
features(x) = [
  log raw local scale,
  log match distance,
  support probability,
  |predicted displacement|,
  panel-local x and z,
  normalized panel index
]

target(x) = log(|u_true(x) - u_hat(x)| + floor)
```

当前实现为 `HistGradientBoostingRegressor(loss="quantile", quantile=0.80)`。点级样本只用于拟合尺度函数；统计覆盖仍以完整扫描对为独立单位。

## Domain-adaptive shrinkage

共享尺度模型在不同设备/退化域可能过度或不足异方差化。对每个预定义已知域，在独立 validation 扫描对上从冻结网格

```text
lambda in {0, 0.25, 0.50, 0.75, 1.00}
```

选择最小化扫描对最大误差校准后平均宽度的系数：

```text
s_das(x; d) = exp((1-lambda_d) log s_homo + lambda_d log s_ml(x))
```

其中 `s_homo` 是该扫描对 `s_raw` 的场内中位数。选择完成后 `lambda_d` 冻结；正式 calibration/test 不再改变。

## Scan-pair simultaneous conformal

对每个独立 calibration 扫描对 `j`：

```text
R_j = max over frozen valid field x
      |u_true_j(x) - u_hat_j(x)| / max(s_das_j(x), epsilon)
```

组 `d` 的有限样本次序统计量为：

```text
k_d = ceil((n_d + 1) * (1 - alpha))
q_d = kth smallest {R_j in group d}
interval(x) = u_hat(x) +/- q_d * s_das(x)
```

覆盖事件是“一个完整扫描对的所有冻结有效场位置均被覆盖”。禁止把点、体素或网格位置当成 conformal 独立样本。

## Unseen-domain policy

- 若域未在 tuning/validation 中出现，`lambda=0`；
- 同时回退到 homoscedastic calibrator 的 pooled quantile，而不是只回退尺度；
- 该区间只报告 empirical coverage，不声称形式保证；
- 若尺度、区间宽度、支持不足或优化失败超过冻结阈值，则扫描对拒识；
- 正式 v1 否决多分数组合后，最终规则为经独立 v2 确认的 scale-only risk ranking；OOD 和 residual 分数只作为阴性/对照结果保留。

## Mandatory baselines and ablations

1. uncalibrated local scale；
2. pointwise marginal conformal（仅作失效对照）；
3. homoscedastic pair-max simultaneous conformal；
4. raw-local pair-max simultaneous conformal；
5. learned structural scale without shrinkage (`lambda=1`)；
6. DAS-FC without domain groups；
7. DAS-FC without unseen fallback；
8. DAS-FC without abstention；
9. full DAS-FC；
10. rejected joint pose-field estimator as a negative mechanism ablation。

## Frozen success conditions

- 已知域同时覆盖的 Wilson 区间与 0.95 名义水平相容，且逐域报告；
- 相对 homoscedastic pair-max 的配对 interval-score 效应在至少 240 个已知域测试扫描对上为负，95% CI 不跨零；
- 相对 raw-local pair-max 保持明显效率优势；
- 未见域不要求 95% 保证，但保守回退或拒识不能比无回退版本更差；
- AURC 优于 residual-only/width-only/OOD-only 对照；
- 所有失败留在分母内。
