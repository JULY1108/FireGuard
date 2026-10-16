# 本机训练：对照作者的 FireGuard 流程

入口是 `train_local.py`，直接读取原始 `FireGuard_Pipeline.ipynb` 的代码单元。
所有数据、配置、日志、图表和模型保存在 `outputs/training/`；原始笔记本保持完整。

## 保留的训练与微调过程

| 作者步骤 | 本机运行 |
|---|---|
| FASDD_CV + D-Fire | 同一 Hugging Face 数据源，约 6.53 GB 下载 |
| 标签映射 | FASDD 保持原标签；D-Fire `0↔1`；最终 `0=fire, 1=smoke` |
| 去重、事件分组 | 复用作者 dHash 算法、阈值 3/12 和分组逻辑 |
| 数据划分 | 随机种子 1337，按事件组 80/10/10 划分；实际图片比例随组大小变化 |
| 数据检查 | 复用作者审计、保存标注预览；增加完整性及非法标签阻断 |
| 初始化 | COCO 预训练 `yolo26s.pt`，80 类检测头改为两类，全网络微调 |
| CCTV 增强 | JPEG 压缩、降采样、灰度、运动模糊、亮度/对比度、CLAHE |
| 试运行 | 作者的 400 张训练 + 120 张验证、2 epoch 配置 |
| A/B 实验 | 与作者默认一样关闭；`--ablation` 可开启 |
| 主训练 | 72 epoch @512；从第一阶段 `best.pt` 开始 48 epoch @640 |
| 通用参数 | AutoBatch、AMP、optimizer=auto、lr0=0.001、lrf=0.01、3 epoch 预热、余弦调度、patience=40 |
| 续训 | 阶段内用 `last.pt` 恢复，阶段间新建训练；每 5 epoch 同步本机备份 |
| 后处理 | 独立测试集评估、验证集告警阈值扫描、速度对比、ONNX/TensorRT 导出尝试、模型卡 |

`optimizer=auto` 的实际优化器和学习率由安装的 Ultralytics 选择，以运行日志/args.yaml 为准。
这份仓库没有锁定作者原始运行的完整依赖和随机状态；这里复现公开流程，不能保证数值结果完全一致。

## 本机适配与修复

- Google Drive 和 `/content` 改为项目内持久目录，缓存也保存在该目录。
- 修复 D-Fire 的 Parquet 解包结果没有传给合并单元的问题，强制确认两套数据都参与训练。
- 实际数据存在零面积框等非法标注，整张图片隔离并记录在 `label_quarantine.json`，原文件保留；避免删除框后把真实正样本变成负样本。
- 修复准备数据中的 50 个越界框：只裁剪到图片边界，保留类别与图片；原标注备份在 `outputs/training/label_fixes/before/`。
- 已完成解包才记录完成标记，下载中断后可重试；ZIP 路径检查防止写出目标目录。
- Windows 训练 DataLoader 使用 `workers=0`，预处理线程仍按作者方式自动设置。
- 本机备份同步复制，阶段状态原子写入；避免云端后台复制方式在本机结束阶段时发生并发覆盖。
- 权重复制使用临时文件再替换；续训比较本机和备份的有效轮数，优先使用进度较新的检查点。已完成并清除优化器的阶段直接恢复完成状态，避免错误地设置 `resume=True`。
- 重用数据前检查每一份标签、图片与标签配对、各类别数量和标签指纹；试运行标记关联数据与增强库版本，缺失权重或数据变化时重新试运行，并保留上一份模型。
- 图表保存为 PNG；每个适配后的代码单元、原笔记本 SHA256 和运行配置可追溯。
- 去重候选集仅记录距离不大于 12 的有效配对，保持原算法的去重/分组结果，减少无关配对占用内存；桶内配对扫描仍有平方级耗时。
- 保留作者的 dHash 去重：相同 dHash 并不等于文件字节相同；近似事件识别也不保证覆盖所有相邻帧。
- 作者在测试集上选告警阈值，本机已改为使用完整验证集；测试集继续用于独立检测评估。阈值扫描统计的是图片触发率，不是检测框召回率，也不是每摄像头每日误报率。
- 去除“基础模型 mAP=0”的未测量结论；只确认 COCO 权重不含 fire/smoke 两个类别，并测量其速度。
- `erasing=0.20` 保留作者配置，但当前 Ultralytics 只在图像分类增强中使用它，此检测任务不会启用随机擦除。
- TensorRT 导出是否成功取决于本机 SDK，结果以生成的文件与日志为准。

## 运行

独立环境复用已有 CUDA PyTorch，并单独安装训练所需的新版本增强库。
本次环境为 PyTorch 2.5.1+cu121、Ultralytics 8.4.138、Albumentations 2.0.8、NumPy 1.26.4、OpenCV 4.11.0。
已核对 GitHub 当前生成脚本及训练笔记本，均与本地原文件逐字节相同。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe train_local.py --check
.\.venv\Scripts\python.exe outputs\training\review_checks.py
.\.venv\Scripts\python.exe -u train_local.py --phase prepare
.\.venv\Scripts\python.exe -u train_local.py --phase all
```

`prepare` 完成数据准备、两轮试运行和基础模型速度测量。
`all` 自动准备并训练；准备完成后重复执行会复用数据，训练中断后会按阶段恢复。
主训练最多 120 epoch，早停可能提前结束。

进度：`outputs/training/progress.json`。
数据审计：`outputs/training/prepare_summary.json`、`outputs/training/reports/`。
训练曲线和阶段权重：`outputs/training/runs/`。
最终模型：`outputs/training/checkpoints/fireguard_s_local_v1/final/best.pt`。

详细复核结论见 `TRAINING_REVIEW.md`。两轮试运行的已结束 `last.pt` 已清除优化器，可作为新一轮微调的起点，不能恢复两轮试运行的优化器状态。
