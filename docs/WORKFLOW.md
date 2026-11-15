# 从游戏桌面切到本地酒馆，再切回来

## 1. 把独显留给模型

单显示器双线连接时，主板 HDMI 用于核显桌面，独显 DP 用于游戏。先把显示器输入源切到 HDMI，再在 Windows「系统 → 显示」只启用核显对应的屏幕。只切显示器输入源，不一定会停掉 Windows 的 DP 输出。

本仓库也提供本机同类拓扑专用的工具（AMD 核显 HDMI + RX 9070 XT DP），普通电脑可直接使用 Windows 设置。首次切换会把**这台机器**的显示拓扑备份到 `state/display-before.json`：

```powershell
.\tools\python.cmd -X utf8 tools\display_mode.py status
.\tools\python.cmd -X utf8 tools\display_mode.py hdmi
```

模型加载前，可双击 `一键整理显存.cmd`：它为支持的桌面应用设置核显优先并刷新图形辅助进程，保留主窗口和模型。用 `--preview` 先查看。它不会关闭游戏、录像、降噪程序等无关负载，不能把驱动占用清零。若有不需要的大显存应用，先保存工作并自行退出。

## 2. 选容量，再看后端是否真的就绪

启动入口及预设见[首页](../README.md)。容量是**输入与输出合计**：32K / 64K / 120K / 128K 分别是 32,768 / 65,536 / 122,880 / 131,072。四个预设共用同一端口，同一时间启动一个。均为单并发、Q4 KV、DFlash2 深度 3；三个长上下文入口使用 chunk 512、64 页解量化暂存，历史没有被截成 16K。

```powershell
Invoke-RestMethod http://127.0.0.1:18543/health
(Invoke-RestMethod http://127.0.0.1:18543/v1/models).data
```

确认 `status=ok` 和实际 `context_length`。加载中可返回 503。酒馆 RP 默认关闭思考，启用句末保护；需要严格格式、代码或无标点短答时可在请求体设 `ensure_sentence_end: false`。工具请求跳过句末保护，输出硬上限和用户 stop 仍有效。

## 3. 识别 WDDM 启动慢态

32K 和 64K 的启动器会在接收请求前对当前 27B 主模型输出头测量三个 4 行 GEMV 的中位数。正常实测约 1.4–1.6 ms；若达到 **10 ms**，后端以 75 退出，启动器换一个**新进程**，优先加载约 909 MiB 的压缩输出头。仅重试一次。日志中 `DECODE_HEAD normal` 和 `DECODE_HEAD head-first` 是实际测量值；如果第二次仍慢，服务会明确提示并继续，不会重启循环。这个检测不改变模型、采样或上下文。

32K 还默认设 `EXL3_GEMV_TILES_T=2`；64K 只默认启用慢态检测，保留本身计算参数；120K/128K 默认保留旧加载策略。只回退检测，启动前在 PowerShell 设 `$env:EXL3_DECODE_PLACEMENT='stock'`。要同时还原 32K 的 tile 默认设置，另设 `$env:EXL3_FAST_DECODE='0'`，并移除手工设过的 `EXL3_GEMV_TILES_T`。关闭当前服务再启动才会生效。强制提前加载不是通用提速开关：有一次 30K 冷输入首 token 等了约 173 秒。

## 4. 用真实回复判断 decode

`decode tok/s = 实际输出 token / 生成阶段秒数`。不要用预填速度、健康检查或四个输出 token 判断持续出字。在酒馆发送一条长回复请求，看后端的「生成」「近 3 秒」及最终统计；非流式 API 返回 `usage` 和 `exl3.generate_tps`。先测短提示，再在**同一进程**测接近实际聊天的长提示。区分「整段平均 ≥40」与「每 128 个输出 token 都 ≥40」。

| 本机条件（RX 9070 XT 16 GB） | 实际生成结果 |
|---|---|
| 32K，30,000 入 / 512 出，greedy，三轮无前缀缓存 | 47.93 → **48.40 tok/s**；新配置三轮 48.57 / 48.34 / 48.40 |
| 启动慢态压力对照，1,024 入 / 256 出 | 14.50 → **48.15 tok/s**，新进程提前分配，输出相同；这是恢复掉速 |
| 同一个 64K 进程，1,024 入 / 512 出 | **50.28 tok/s** |
| 同一个 64K 进程，64,000 入 / 512 出，两条 RP | **44.51 / 41.35**；最低分段 **39.38**，严格门槛未过 |
| 后来一次 64K 实际服务，3,872 入 / 512 出 | 输出头 **64.87 → 1.59 ms**，decode **32.58**；修复输出头不等于整机能保 40 |
| 120K，120,000 入 / 512 出 | **13.92**，分段 13.42–15.36；接近 20 分钟的冷预填不计入 decode |
| 128K，129,992 入 / 4 出 | 正确取回开头识别码，只证明容量；没有足够输出测试持续速度 |

[数值记录](measurements.json)。64K 两条长回复的验证轮耗时都约 37.7 ms，但草稿接受数为 207 / 184，生成速度因而不同。120K 未做同负载的 128K 配对实验，不能据此推断 128K 存在固定性能断崖。

可复查 API（会真实占用一个生成槽，空闲时运行）：

```powershell
.\tools\python.cmd tools\check_tauritavern_api.py --out logs\api-check.json
```

要测当前酒馆服务实际生成速度，用同一套公开 RP 提示，可选 `--input-tokens 3872`、`64000` 或 `120000`。先检查已加载的容量是否能容纳提示和 512 个输出 token；64K 长提示的冷预填可能需要数分钟，120K 可能需要二十分钟。命令保存时间与计数，不保存对话。SSE 分段计时为观察边界估算区间，与精确的服务内逐 token 计时分开报告。

```powershell
.\tools\python.cmd -X utf8 tools\measure_rp.py --input-tokens 3872 --out logs\rp-01.json
```

单独跑 30K/512 三轮冷基准时先停服务，设置首页中的模型路径。下面的命令使用随库发布的合成叙事提示，不读私人聊天；每轮新 Generator。新输出目录不要复用。

```powershell
.\tools\python.cmd tools\run_30k_bench.py --out logs\bench-30k-01 --runs 3 --env EXL3_GEMV_M4_FDOT2=1 --env EXL3_GEMV_TILES_T=2 --env EXL3_DECODE_PLACEMENT=probe
```

若基准返回 75，将最后一项改为 `EXL3_DECODE_PLACEMENT=head-first`，用新目录再跑一次。日常酒馆启动器会自动处理。验证安装用 `verify.cmd`，检查发布文件完整性用 `tools\python.cmd tests\verify_manifest.py`。

## 5. 结束推理，恢复游戏

先结束生成。只想停后端，运行 `stop_tauritavern.cmd`。想同时释放模型显存并整理桌面，双击 **后台释放显存（会停止酒馆模型）.vbs**。它是手动运行一次的工具，不是定时任务；会中断该项目的生成，保留酒馆前端。不要在需要保留当前回复时运行。

显示器切到 DP，确保 Windows 已检测到 DP。若先前通过脚本切换，运行：

```powershell
.\tools\python.cmd -X utf8 tools\display_mode.py restore
.\tools\python.cmd -X utf8 tools\display_mode.py status
.\一键整理显存.cmd --restore
```

确认活动输出属于 RX 9070 XT、分辨率及 Hz 正确（本机为 2560×1440、180 Hz）。恢复使用本机生成的快照；如果系统已重启、接口变动或 DP 未被检测到，使用 Windows 显示设置选择独显 DP 并设支持的刷新率，不要导入别人的快照。桌面应用的 GPU 首选项恢复后，在对应应用下次启动时生效。
