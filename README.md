# 宝宝哭声监控（Windows 客户端）

拉取小蚁摄像头（yi-hack 固件）的 RTSP 流：窗口里看实时视频，
后台分析纯音频流，检测到宝宝哭声后 Windows 右下角弹窗提醒。

## 检测原理

两级流水线，全部离线、本机运行：

1. **能量门**（常开）：0.5 秒一块音频，音量超过阈值且持续 `loud_seconds` 秒才往下走——过滤安静时段，省 CPU。
2. **YAMNet AI 确认**：Google 的 4MB 离线小模型（`assets/yamnet.tflite`），自带"婴儿哭声"类别，
   对最近 3 秒音频打分，超过 `yamnet_threshold` 才触发。触发后冷却 `cooldown_seconds` 秒。

YAMNet 加载失败时自动降级为**纯音量模式**（界面会显示），功能不受影响，只是误报稍多。

## 安装

1. 从 [Releases](../../releases) 下载 `BabyCryMonitor.exe`，放到任意目录（建议单独建个文件夹）。
2. 双击运行，首次运行会自动生成 `config.json`。
3. 用记事本打开 `config.json`，把 `rtsp_video_url` / `rtsp_audio_url` 里的 IP 改成你摄像头的局域网 IP。
4. 点"开始监控"。

## 自检

点"检测音频流"：4 秒内收到数据且峰值正常 = 音频链路 OK。
在摄像头旁边拍手，峰值会明显跳动——这是最快的麦克风验证法。

如果连不上，检查两点：

- 摄像头 Web 后台 Configurations 页的 **RTSP Audio** 开关是否打开；
- `config.json` 里的 IP 对不对（和电脑在同一 WiFi 下）。

## 配置项（config.json）

| 项 | 说明 |
|---|---|
| `rtsp_video_url` | 视频流（高清 ch0_0 / 低清 ch0_1 二选一） |
| `rtsp_audio_url` | 纯音频流 ch0_2，只取声音不解视频 |
| `energy_threshold` | 音量门阈值（0~1），误报多就调高 |
| `loud_seconds` | 持续大声多少秒才送 AI 判定 |
| `yamnet_threshold` | AI 哭声分数阈值（0~1），误报多就调高 |
| `cooldown_seconds` | 触发后冷却秒数，防消息轰炸 |

## 常见问题

- **exe 被 Defender 拦截**：PyInstaller 打包的程序常被误报，点"仍要运行"即可（代码开源可查）。
- **只有视频没声音**：先跑"检测音频流"自检；大概率是摄像头 RTSP Audio 没开。
- **白天误报多**：调高 `yamnet_threshold`（如 0.65）或 `energy_threshold`。
- **日志**：`baby_cry_monitor.log` 在 exe 同目录，报错先看它。

## 构建

打 tag 即自动构建：`git tag v1.0.0 && git push origin v1.0.0`，
GitHub Actions（windows-latest）会自动打出 `BabyCryMonitor.exe` 并建 Release。
ffmpeg 通过 imageio-ffmpeg 静态二进制打包，无需另外安装。
