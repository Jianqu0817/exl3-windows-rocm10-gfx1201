# RX 9070 XT 上的本地酒馆：Windows ROCm 10.1 + EXL3

在原生 Windows 上用 16 GB RX 9070 XT 运行 27B EXL3 模型，并通过 OpenAI 兼容接口连接 TauriTavern。引擎为 ExLlamaV3 1.6.6，含 RDNA4 内核、匹配的预编译扩展和 DFlash2 草稿支持。不需要 WSL。

**日常长聊天首选 64K。** 实际输入 64,000 token、连续生成 512 token 的两条中文 RP 分别为 **44.51 / 41.35 decode tok/s**。最低的连续 128-token 窗口为 **39.38**；另一次真实 64K 服务即使修复了输出头慢态，也只跑到 **32.58**。这些是具体负载下的结果，不是 40 tok/s 保底。

## 启动 64K

1. 克隆本仓库。准备 Windows x64 的 **Python 3.12、torch 2.13.0+rocm10.1.0、Windows Triton 3.7.1.post27** 配套环境。可从 [Windows 基线项目的 Releases](https://github.com/BBQ2077/exl3-windows-gfx1201/releases) 获取对应 A-python 包；使用本仓库的源码与 `native/`，不要再用旧 B-app 包覆盖。A 包解压后若有仓库根目录下的 `python/python.exe`，不必设置 `EXL3_PYTHON`。
2. 准备 [Huihui 27B EXL3 3.0 bpw 主模型](https://huggingface.co/grimlee/Huihui-Qwen3.8-27B-abliterated-EXL3-3.0bpw) 和匹配的 EXL3 DFlash2 草稿。实测使用本地中文 RP 微调草稿，仓库不含模型权重；换用原版草稿，生成速度可能不同。
3. 在仓库目录的 PowerShell 中运行（将路径换成自己的）：

```powershell
$env:EXL3_PYTHON    = 'D:\EXL3-runtime\python\python.exe'  # 若运行时不在仓库中
$env:EXL3_MODEL_DIR = 'D:\Models\Huihui-Qwen3.8-27B-abliterated-EXL3-3.0bpw'
$env:EXL3_DRAFT_DIR = 'D:\Models\Qwen3.8-27B-DFlash2-EXL3-3.0bpw-tuned'
.\start_tauritavern_64k.cmd
```

4. 看到 `EXL3_SERVER_READY` 后，打开 **TauriTavern 客户端** → 聊天补全 → 导入 [64K 预设](integrations/tauritavern/EXL3-Huihui-RP-64K.json)，选择「原样导入」并保留连接绑定。地址是 `http://127.0.0.1:18543/v1`，模型名 `huihui-qwen3.8-27b`，API key 留空。启动脚本启动的是本地后端，不会替你安装或修改客户端。

| 用途 | 后端入口 | [导入同容量的酒馆预设](integrations/tauritavern/) |
|---|---|---|
| 较短对话 / 30K 基准 | `start_tauritavern.cmd` | EXL3-Huihui-RP.json |
| 日常长对话 | `start_tauritavern_64k.cmd` | EXL3-Huihui-RP-64K.json |
| 需要更长历史 | `start_tauritavern_120k.cmd` | EXL3-Huihui-RP-120K.json |
| 主要验证容量 | `start_tauritavern_128k.cmd` | EXL3-Huihui-RP-128K.json |

**[完整工作流：显存准备 → 启动与慢态恢复 → 实际测速 → 切回游戏](docs/WORKFLOW.md)** · [显存工具说明](tools/cleanup_vram.md) · [编译原生扩展](BUILD.md)

基于 [ExLlamaV3](https://github.com/turboderp/exllamav3)、[RDNA4 recipe](https://github.com/parth-thakre/exl3-rdna4-recipe) 和 [Windows gfx1201 基线](https://github.com/BBQ2077/exl3-windows-gfx1201)。预编译扩展面向 gfx1201。上游版本与二进制来源见 [provenance.json](provenance.json)，各组件保留[原许可证](licenses/)。
