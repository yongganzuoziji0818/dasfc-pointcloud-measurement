# 公开真实点云数据清单

公开真实数据用于验证真实扫描伪影、刚体配准和跨设备泛化；它们不能替代真实铁路声屏障现场验证。

| 数据源 | 官方入口 | 用途 | 真值/监督 | 主要限制 | 接入优先级 |
|---|---|---|---|---|---|
| WHU-TLS | [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/WHU-TLS/Default.aspx) / [GitHub](https://github.com/WHU-USI3DV/WHU-TLS) | 真实 TLS 刚体配准、密度/遮挡/环境域迁移 | 扫描对真值变换与配准图 | 没有声屏障位移真值；数据需填写官方 Google Form 获取 | P0 |
| 3DPrintedShapes | [Zenodo 10.5281/zenodo.19471431](https://doi.org/10.5281/zenodo.19471431) | 真实结构扫描、跨 iPad/FARO/Creaform 设备测试 | 打印所用数字模型与三设备扫描可用于单时相表面误差和跨设备重复性 | 钢构件不是声屏障；没有共同未变形时相，不能直接评价双时相位移真值 | **P0 connected** |
| SynBench | [HeiDATA 10.11588/data/R9IKCF](https://doi.org/10.11588/data/R9IKCF) | 非刚性变形、噪声/离群/缺失和对应点真值 | 完整对应点真值 | 合成数据，不作为真实域证据 | P1 |
| UNAVCO TLS Archive | 数据项目级入口待筛选 | 多时相真实 TLS 和变化检测压力测试 | 部分项目含配准/地理参考信息 | 任务异质，通常没有毫米级位移真值 | P2 |

## 首批云端接入锚点

3DPrintedShapes v1（2026-04-08）固定为第一批公开真实点云：

- 38 个物理打印构件、114 组真实扫描；每个构件包括 iPad、FARO Focus 和 Creaform HandySCAN3D 三个设备域；
- 每个设备域同时提供 `Raw.pcd` 和 `Processed.pcd`，构件还提供打印所用 `1Digital_Model/model.stl`；
- 主归档 `3DPrintedShapes.7z` 精确大小为 5,270,430,109 字节，发布方 MD5 为 `44cf569120483cb7248092c5eb4e4aaa`；
- 数据许可为 CC BY 4.0；论文使用时必须引用数据集作者和 DOI；
- 云端接入已于 2026-07-19 通过主归档 MD5、`py7zr` 完整性、逐文件布局、STL 实读和三设备 Raw/Processed PCD 实读。

## 已核验的第一批证据

- 主归档 SHA-256：`fc589158d7dd6b49e3879f2b736eb729697487bb110550d6c57ff35b251d73af`；
- 解压布局：38 个 specimen、38 个 `model.stl`、228 个 PCD（每设备 38 Raw + 38 Processed），无缺失；
- 抽检的 3 个 STL 和 6 个 PCD 均为有限、非空几何；Processed PCD 与数字模型处于相同毫米尺度并已做发布方 position adjustment；
- 三个构件的无隐藏配准直接距离审计显示设备差异明确：Creaform 的双向对称均值约 1.46--1.74 mm，FARO 约 3.61--4.14 mm，iPad 约 5.47--6.73 mm。该数字是数据质量审计，不是 PCU-DM 结果；
- 原始证据：`evidence/data/3dprintedshapes_audit.json` 与 `evidence/data/3dprintedshapes_mesh_scan_alignment.json`。

## 全量真实审计

- 已完成全部 38 个物理构件 × 三设备的 114 个 Processed 扫描直接比较，独立统计单位为 specimen cluster；
- 严格的 Creaform < FARO < iPad 误差梯度出现在 33/38 个构件；
- 全量均值（无隐藏配准）分别为 Creaform 5.55 mm、FARO 7.49 mm、iPad 9.62 mm；
- 两个构件在三设备上均出现 17.7--54.2 mm 的共同大失配，作为真实发布坐标/配准失败样本保留；
- 全量证据：`evidence/data/3dprintedshapes_mesh_scan_alignment_full.json` 与 `evidence/data/3dprintedshapes_full_statistics.json`；
- 另有预先冻结的标准 ICP 与未改 Cascade-Strong 外部转移任务；在其成功标记和聚类统计完成前，不把它写成算法结果。

## 数据接入规则

1. 原始数据不提交 Git；只提交下载说明、许可证、校验和、解析器和派生清单。
2. 每个数据集建立 `dataset_manifest.json`，记录版本、来源 URL、许可、文件哈希、坐标单位和预处理步骤。
3. 不把同一结构的点或切片随机分到训练与测试；划分单位至少是扫描对、结构或采集场景。
4. 公开真实数据分为三类证据：
   - A：具有外部真值，可评价绝对误差；
   - B：具有配准真值或数值几何，只评价对应子任务；
   - C：无真值，只进行重复性、自一致性和失败检测。
5. 论文必须逐项说明每个数据集能证明什么、不能证明什么。

## 计划目录

```text
DATA/public/
  whu_tls/
  3d_printed_shapes/
  synbench/
  manifests/
```

3DPrintedShapes 已位于云端持久盘
`/workspace/sound-barrier-measurement/datasets/extracted/3dprintedshapes/3DPrintedShapes`；
原始和解压数据不进入 Git。其余数据源仍是候选，不宣称已经接入。
