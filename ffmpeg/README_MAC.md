# Mac系统FFmpeg使用指南

## 概述

本文档说明如何在Mac系统上配置和使用FFmpeg，支持Intel和Apple Silicon架构。项目使用智能包装脚本方案，自动检测并使用系统安装的FFmpeg。

## 系统要求

- macOS 10.15 (Catalina) 或更高版本
- Intel x86_64 或 Apple Silicon ARM64 架构
- 系统已安装FFmpeg（推荐通过Homebrew安装）

## 安装FFmpeg

### 使用Homebrew安装（推荐）

```bash
# 安装Homebrew（如果未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装FFmpeg
brew install ffmpeg
```

### 验证安装

```bash
# 检查FFmpeg版本
ffmpeg -version | head -1
ffprobe -version | head -1
```

## 项目集成

### 自动平台检测

项目会自动检测Mac系统并使用正确的FFmpeg包装脚本：

```python
# config.py 自动处理逻辑
if system_platform == "darwin":
    # Mac系统，自动添加_darwin后缀
    Config.base['ffmpeg_path'] = f"{Config.base['ffmpeg_path']}_darwin"
    Config.base['ffprobe_path'] = f"{Config.base['ffprobe_path']}_darwin"
```

### 配置文件

在`config/base.json`中，使用标准路径：

```json
{
    "ffmpeg_path": "ffmpeg/ffmpeg",
    "ffprobe_path": "ffmpeg/ffprobe"
}
```

系统会自动转换为：
- `ffmpeg/ffmpeg_darwin` (Mac系统)
- `ffmpeg/ffmpeg_arm64` (Linux ARM64)
- `ffmpeg/ffmpeg_amd64` (Linux AMD64)

## 目录结构

```
ffmpeg/
├── ffmpeg_darwin          # Mac系统FFmpeg包装脚本
├── ffprobe_darwin         # Mac系统FFprobe包装脚本
├── ffmpeg_amd64           # Linux x86_64版本
├── ffmpeg_arm64           # Linux ARM64版本
├── ffprobe_amd64          # Linux x86_64版本
├── ffprobe_arm64          # Linux ARM64版本
└── README_MAC.md          # 本文档
```

## 包装脚本工作原理

### FFmpeg包装脚本 (`ffmpeg_darwin`)

```bash
#!/bin/bash
# Mac系统FFmpeg包装脚本
# 自动检测并使用系统FFmpeg

# 查找FFmpeg路径
FFMPEG_PATHS=(
    "/usr/local/bin/ffmpeg"
    "/opt/homebrew/bin/ffmpeg"
    "/usr/bin/ffmpeg"
    "$(which ffmpeg 2>/dev/null)"
)

FFMPEG_CMD=""
for path in "${FFMPEG_PATHS[@]}"; do
    if [ -x "$path" ]; then
        FFMPEG_CMD="$path"
        break
    fi
done

if [ -z "$FFMPEG_CMD" ]; then
    echo "❌ 错误: 未找到FFmpeg，请先安装FFmpeg"
    echo "安装命令: brew install ffmpeg"
    exit 1
fi

# 执行FFmpeg命令
exec "$FFMPEG_CMD" "$@"
```

### 特性

- **自动路径检测**: 支持Homebrew、系统安装等多种FFmpeg安装位置
- **错误处理**: 未找到FFmpeg时提供清晰的错误信息和安装指导
- **透明代理**: 完全透明地传递所有参数给系统FFmpeg
- **跨架构支持**: 自动适配Intel和Apple Silicon架构

## 使用方法

### 1. 直接使用包装脚本

```bash
# 使用Mac版本FFmpeg
./ffmpeg/ffmpeg_darwin -i input.mp4 output.mp4

# 使用Mac版本FFprobe
./ffmpeg/ffprobe_darwin input.mp4
```

### 2. 在代码中使用

```python
from config.config import Config

# 初始化配置（自动检测平台）
Config.init()

# 获取FFmpeg路径
ffmpeg_path = Config.base['ffmpeg_path']  # 自动为Mac系统添加_darwin后缀
ffprobe_path = Config.base['ffprobe_path']

print(f"FFmpeg路径: {ffmpeg_path}")
print(f"FFprobe路径: {ffprobe_path}")
```

### 3. 测试配置

```bash
# 测试FFmpeg功能
./ffmpeg/ffmpeg_darwin -version

# 测试FFprobe功能  
./ffmpeg/ffprobe_darwin -version
```

## 故障排除

### 问题1: 找不到FFmpeg

**错误信息**: `❌ 错误: 未找到FFmpeg，请先安装FFmpeg`

**解决方案**:
```bash
# 安装FFmpeg
brew install ffmpeg

# 验证安装
ffmpeg -version
```

### 问题2: 权限问题

**错误信息**: `Permission denied`

**解决方案**:
```bash
# 设置执行权限
chmod +x ffmpeg/ffmpeg_darwin
chmod +x ffmpeg/ffprobe_darwin
```

### 问题3: 路径问题

**错误信息**: `No such file or directory`

**解决方案**:
```bash
# 检查文件是否存在
ls -la ffmpeg/ffmpeg_darwin
ls -la ffmpeg/ffprobe_darwin

# 检查系统FFmpeg
which ffmpeg
which ffprobe
```

## 性能优化

### 1. 硬件加速

Mac系统支持多种硬件加速：

```bash
# 检查可用硬件加速
./ffmpeg/ffmpeg_darwin -hide_banner -hwaccels

# 使用VideoToolbox硬件加速
./ffmpeg/ffmpeg_darwin -hwaccel videotoolbox -i input.mp4 output.mp4
```

### 2. 多线程优化

```bash
# 使用多线程编码
./ffmpeg/ffmpeg_darwin -threads 0 -i input.mp4 output.mp4
```

## 版本信息

- **包装脚本版本**: 1.0
- **支持平台**: macOS 10.15+
- **支持架构**: Intel x86_64, Apple Silicon ARM64
- **FFmpeg版本**: 依赖系统安装版本（推荐7.0+）

## 更新日志

### v1.0 (2024-08-16)
- 实现智能包装脚本方案
- 支持自动平台检测
- 集成到config.py自动处理逻辑
- 删除过时的安装和下载脚本

---

**注意**: 此方案的优势是无需下载大型静态文件，直接利用系统安装的FFmpeg，确保最佳性能和兼容性。
