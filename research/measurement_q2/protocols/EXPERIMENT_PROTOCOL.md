# Measurement 目标实验协议（DAS-FC 冻结前版本）

## 当前状态

- 状态：`PILOT_PASSED_FORMAL_PROTOCOL_FREEZE_IN_PROGRESS`
- 点估计前端：`Cascade-Strong`，所有不确定性方法共享；
- 完整不确定性算法：DAS-FC；
- 已否决：把 joint pose-field 优势作为论文主贡献。

## 独立实验单位

独立单位为“结构几何 × 变形工况 × 扫描域 × 随机种子”形成的完整扫描对，不是单个点或单个体素。统计分析不得把同一扫描对内的数百万点视为数百万独立样本。

## 域设计

合成域至少覆盖：

- 结构：板宽、高度、厚度、立柱间距、接缝形式；
- 变形：刚体偏移、整体倾斜、板弯曲、局部鼓包、模态组合、接缝错动；
- 采样：扫描角、点间距、列扫描速度、时序错位；
- 退化：距离相关噪声、入射角噪声、离群点、遮挡、缺失回波、密度变化；
- 标定：外参漂移、轴向尺度误差、粗配准误差。

先用空间填充设计覆盖连续参数，再固定一组留一域测试：未知结构、未知扫描器参数、未知退化组合。

正式合成域分为：

- 已知组：nominal、noise/density、occlusion/outlier、oblique/pose 四类，均有独立 tuning/validation/calibration/test；
- 未见组：复合极端退化、超范围位姿、支承候选严重失配、结构/幅值超范围；
- 未见组不参与尺度拟合、lambda 选择和 conformal 分位数估计。

## 基线

- C2C、C2M、M3C2；
- point-to-point ICP、point-to-plane ICP、GICP、NDT；
- CPD/BCPD 或同等级非刚性配准；
- 当前仓库差动算法；
- 去掉稳定区域、物理约束、异方差、不确定性校准和域鲁棒模块的消融版本。
- homoscedastic、raw-local、learned-unshrunk 与 DAS-FC 四个共享点估计的 pair-max simultaneous interval；
- pointwise marginal conformal 和 field-fraction calibration 失效对照；
- no-group、no-fallback、no-abstention、joint-negative 消融。

## 主指标

主指标必须在正式实验前冻结：

1. 位移向量 EPE、法向位移 MAE/RMSE 和系统偏差；
2. 扫描对级 95% 整场同时覆盖率、平均区间宽度、interval score；
3. risk-coverage/AURC 与拒识后保留率；
4. 按域报告的 worst-domain error，而非只报总体平均；
5. 运行时间和峰值内存。

阈值检测 F1/AUPRC 只能作为次要指标，且不能使用未经授权的真实铁路安全阈值。

## 数据划分和调参纪律

- 合成训练、校准和测试生成种子完全分离；
- 真实数据按结构/场景留出；
- 测试域禁止用于超参数选择和不确定性校准；
- 所有方法使用等价的输入信息和预处理预算；
- 记录失败、无输出和数值不收敛，禁止静默删除。
- tuning、validation、calibration、test 使用不相交 seed 系列；tuning 拟合尺度，validation 只选择 lambda，calibration 只估计分位数；
- 未见域固定 `lambda=0` 且回退 homoscedastic pooled quantile，或按冻结拒识规则不输出。

## 正式样本量冻结

- 已知域 test 合计至少 240 个独立扫描对，且每个主域不少于 60；
- 功效依据：双侧 `alpha=0.05/3`、power 0.90，对 `d_z=0.25` 需 220 对，对 `d_z=0.35` 需 114 对；
- 每个已知域 calibration 目标不少于 150，对 0.95 次序统计量记录精确 `n` 与 `k`；
- tuning 每已知域不少于 60，validation 每已知域不少于 40；
- 每个未见压力域 test 不少于 60，只报告经验覆盖和风险；
- 真实数据使用全部 38 个 specimen，结构 ID 是独立 cluster，三设备和 Raw/Processed 不是独立结构。

## 统计报告

- 对独立扫描对做配对比较；
- 报告效应量和置信区间，不只报告 p 值；
- 对多个方法/域进行多重比较校正；
- 同时给出平均域、最差域和逐域结果；
- 正式实验至少使用预先冻结的多个独立生成种子。

## 预注册式成功条件

数值阈值在完成数据审计和 pilot 后冻结。正式实验前至少要求：

1. 最强基线、PCU-DM 和所有关键消融均能端到端复现；
2. pilot 不用于最终论文显著性检验；
3. 95% 区间覆盖和宽度共同达标；
4. PCU-DM 在多数域及最差域均表现出稳定优势；
5. 公开真实数据结果与可证明范围严格一致。
6. DAS-FC 相对 homoscedastic pair-max 的 interval-score 配对效应 CI 不跨零；
7. AURC 优于单一 residual/width/OOD 拒识对照；
8. 未见域结果明确标为 empirical，且 fallback/no-fallback 消融完整。
