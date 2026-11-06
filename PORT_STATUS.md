# 原生 Windows / ROCm 10 移植记录

目标：将 parth-thakre/exl3-rdna4-recipe 的固定版本移植到 RX 9070 XT / gfx1201 的原生 Windows 环境。

## 源码

- Windows 包基线：BBQ2077/exl3-windows-gfx1201，`d20e8dd71c97f637e0a975c37a38583799ecb163`。
- ExLlamaV3：1.6.6，`bd1f519172406a9560bd5c90e61000691bb69a07`。
- recipe：`fe77c7b430b27232959d8fa19cb0de6f8640178c`，其组合补丁已应用。
- `app/exllamav3` 使用完整上游核心，额外保留 Windows 的 clang-cl AVX 适配和预编译扩展加载方式。
- 使用 recipe 的 2～32 行 WMMA、MLP 和注意力优化；Windows 包自行添加的 64 行/TN 调优与随机 DFlash 拒绝采样没有并入此候选版。
- 原有可用环境 `D:\LLM_local\tools\exl3-gfx1201` 和 WSL 环境不被改动。

## 环境与执行

运行时：复用现有 Windows 包中的 Python、PyTorch 2.13.0+rocm10.1.0、Triton 3.7.1.post27。

编译时：独立 `build/.venv`，使用 Windows 基线固定的 PyTorch/SDK 10.0.0rc4 组合。
现有免安装运行包删除了 Torch C++ 头文件和导入库，因此编译环境需要完整 Torch 开发文件；不下载模型。

```powershell
cd D:\LLM_local\ports\exl3-rdna4-win-rocm10
powershell -ExecutionPolicy Bypass -File tools\build\prepare_port.ps1
python tools\build\guarded_build.py --wait
```

构建默认单进程，可通过 `MAX_JOBS=2` 使用两个编译进程，最多两个；保持低优先级。启动要求可用 RAM 至少 10 GiB，运行中低于 5 GiB 则只终止本次构建进程树。
构建日志：`logs/build.log`。成功后扩展写入本目录 `native/exllamav3_ext.pyd`。

GPU 验证需先查看 WSL 与 Windows 占用。不要在 WSL 正在推理时加载 27B 模型。
待资源允许，先执行 `port.cmd verify` 的小规模 Torch/Triton/EXL3 检查，再执行 recipe 数值测试和模型生成测试。
`test_gemv32.py` 会加载完整模型，不适合与 WSL 推理同时运行。

## 当前结果：微调草稿与解码修改已验证

- 原生扩展仍为 `6ab6cf3a1872dbd38f4b1534670d04079fa29dd41fc89e46271b9508bd17c1f6`，本轮只修改 Python 解码路径，未重新编译 HIP。
- 默认草稿已切换到 `D:\LLM_local\Qwen3.8-27B-DFlash2-EXL3-3.0bpw-tuned`，深度 3。
- 同一 30K / 512 输出 / 零缓存三轮：原草稿 38.59、仅换微调草稿 41.51、实现修改后 **41.94 tok/s**。
- 本任务接受率 19.67% → 22.99%；代码改动额外约 1%，增幅较小。
- EOS mask 复用、稳定 minimum 窗口批量采样、纯 greedy 整窗采样、草稿输出头前缀计算已启用。
- 16 组完整模型对照、6 组选择器前缀对照、贪心与随机 API 均通过。三个 30K 输出与微调草稿修改代码前逐字节一致。
- 启动器启用 `-rcs 0.5`，作为 512 MiB recurrent checkpoint 的 RAM 预算。重复 6021 token 输入时复用 5888 token；冷基准仍是 `-rcs 0`。
- 注意力 split=8 的完整模型吞吐未提升，仍保留原 split 策略。
- 启动 `start_huihui.cmd`，localhost:18543；内存紧张可追加 `-rcs 0`。
- 报告：`D:\LLM_local\benchmarks\20261107-win-rocm10-tuned-headwindow\REPORT.md`。
- 修改前快照：`snapshots/20261107-before-deep-tuning`；解码开关：`EXL3_DECODE_PIPELINE=0`、`EXL3_DFLASH_HEAD_WINDOW=0`。
- 所有测试服务均已退出。原有 Windows 包、原草稿及 WSL 环境保留。

测试依赖：`uv pip install --target build/test-deps pytest`。
原生扩展已有的 432 项 WMMA 与 8 组注意力数值检查结果仍保留在 `logs/`。

后续 Agent 交接与优化任务：[HANDOFF_OPTIMIZATION.md](HANDOFF_OPTIMIZATION.md)。
