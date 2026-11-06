# ExLlamaV3 on native Windows + ROCm 10 — AMD RX 9070 XT (gfx1201)

将 [ExLlamaV3](https://github.com/turboderp/exllamav3) 1.6.6 与 [parth-thakre/exl3-rdna4-recipe](https://github.com/parth-thakre/exl3-rdna4-recipe) 的 RDNA4 优化补丁移植到 **Windows 原生 ROCm 10.1**(HIP 7.16),面向 **RX 9070 XT / gfx1201**——不需要 WSL 或 Linux 容器。

> Native Windows port of ExLlamaV3 1.6.6 + RDNA4 recipe optimizations, running on AMD RX 9070 XT (gfx1201) with ROCm 10.1 — no WSL needed.

## 实测性能(RX 9070 XT 16GB, Ryzen 7 7700)

27B EXL3 3.0bpw 主模型 + DFlash2 草稿(深度 3)、Q4 KV、32K 上下文、30K token 输入冷基准:

| 指标 | 数值 |
|---|---:|
| prefill | ~1300 tok/s |
| 推测解码(spec decode) | ~42 tok/s(接受率 ~23%) |
| 前缀缓存命中后 prefill | ~0.26 s / 133 token(比上游阈值默认快 4.3×) |

prefill 与 Linux ROCm 参考栈持平;decode 达 WSL 参考的 ~96%。

## 仓库内容

- `app/exllamav3` — 完整源码(含 RDNA4 recipe 补丁与 Windows clang-cl/AVX 适配、本轮的 reconstruct 阈值修正)
- `native/exllamav3_ext.pyd` — **预编译原生扩展**(gfx1201,含 WMMA GEMV / Q4 attention 优化),免编译直接用
- `runtime.py` / `server.cmd` / `start_huihui.cmd` — 运行入口与 API 启动器示例
- `tests/` — 数值等价性、kernel/attention 冒烟、30K 基准等验证脚本
- `tools/build/` — 受控构建脚本(`prepare_port.ps1` + `guarded_build.py`,改原生代码时用)
- `licenses/` — 上游许可(ExLlamaV3 MIT、ROCm、AGPL 组件等)

## 快速开始

1. 准备 Windows Python 运行时:PyTorch `2.13.0+rocm10.1.0` + Windows Triton(原生 ROCm for Windows),详见 [README.windows-base.md](README.windows-base.md) 与 [BUILD.md](BUILD.md)。
2. `python -I -X utf8 -u runtime.py script <你的脚本.py> -m <模型目录> ...`(`runtime.py` 负责加载预编译扩展、设置 ROCm/Triton 环境)。
3. API 服务示例:`server.cmd -m <模型> -dm <DFlash2 草稿> -ndt 3 -cs 32768 -cq 4`。
4. 验证安装:`port.cmd verify`(Torch/Triton/EXL3 数值检查)、`port.cmd script tests/port_kernel_smoke.py`。

模型不随仓库分发;量化转换用上游 `exllamav3.conversion.convert_model`。

## 说明与限制

- 预编译 `.pyd` 仅适用于 gfx1201;其他 RDNA4 GPU(gfx110x 等)需用 `tools/build` 重新编译。
- 仅在 RX 9070 XT + Windows 11 + ROCm 10.1 验证;上游 ExLlamaV3 支持的模型架构均保留(`supported-architectures.json`)。
- 上游 recipe 的部分实验特性(64 行 GEMV 调优、随机 DFlash 拒绝采样)未并入。

## 来源与致谢

- [turboderp/exllamav3](https://github.com/turboderp/exllamav3)(MIT)— 核心引擎
- [parth-thakre/exl3-rdna4-recipe](https://github.com/parth-thakre/exl3-rdna4-recipe)— RDNA4 decode 优化补丁集
- [BBQ2077/exl3-windows-gfx1201](https://github.com/BBQ2077/exl3-windows-gfx1201)— Windows 免安装包基线
- AMD ROCm SDK for Windows

本仓库为非官方社区移植,与 AMD 无关;按上游许可(MIT)分发,详见 [licenses/](licenses/)。
