# 静态资源目录说明

本目录用于存放系统的静态资源文件，包括CSS样式、JavaScript脚本、图片、视频和音效。

## 目录结构

```
static/
├── css/           # CSS样式文件
│   └── style.css  # 主样式文件（待创建）
├── js/            # JavaScript脚本
│   └── app.js     # 主应用脚本（待创建）
├── images/        # 图片资源
│   └── floor_plan.png  # 平面图（需准备）
├── videos/        # 演示视频
│   ├── demo_fire.mp4   # 火灾场景视频（需准备）
│   ├── demo_smoke.mp4  # 烟雾场景视频（需准备）
│   └── demo_normal.mp4 # 正常场景视频（需准备）
└── sounds/        # 音效文件
    └── alert.mp3  # 告警音效（可选）
```

## 演示资源准备指南

### 1. 演示视频（必需）

**要求**：
- 格式：MP4（H.264编码）
- 分辨率：640x480或更高
- 帧率：15-30fps
- 时长：建议30秒以上（系统会循环播放）

**推荐场景**：
1. **demo_fire.mp4**：明显的火焰和烟雾场景
2. **demo_smoke.mp4**：仅有烟雾，无明显火焰
3. **demo_normal.mp4**：正常场景，无火灾和烟雾

**获取方式**：
- 使用现有的测试视频（如果有）
- 从datasets/fire_videos_organized/目录复制
- 使用ffmpeg转换格式：
  ```bash
  ffmpeg -i input.avi -c:v libx264 -crf 23 -preset medium -c:a aac output.mp4
  ```

### 2. 平面图（必需）

**要求**：
- 格式：PNG（支持透明背景）
- 分辨率：1200x800或更高
- 内容：古建筑的俯视平面图

**准备方式**：
- 使用建筑CAD图纸导出为PNG
- 使用在线工具绘制简单平面图
- 临时使用占位图（纯色背景+文字标注）

**占位图创建示例**（Python）：
```python
from PIL import Image, ImageDraw, ImageFont

# 创建1200x800的深灰色背景
img = Image.new('RGB', (1200, 800), color='#2C2C2E')
draw = ImageDraw.Draw(img)

# 添加文字
draw.text((600, 400), "古建筑平面图", fill='white', anchor='mm')

# 保存
img.save('static/images/floor_plan.png')
```

### 3. 告警音效（可选）

**要求**：
- 格式：MP3
- 时长：1-3秒
- 音量：适中，不刺耳

**获取方式**：
- 从免费音效网站下载（如freesound.org）
- 使用系统自带音效
- 暂时不使用音效（前端代码会检查文件是否存在）

## 快速开始

### 最小配置（仅用于测试）

如果暂时没有演示资源，可以：

1. **创建占位视频**：复制任意视频文件3次，重命名为demo_fire.mp4、demo_smoke.mp4、demo_normal.mp4

2. **创建占位平面图**：使用上面的Python脚本或任意图片

3. **跳过音效**：前端会自动处理音效文件不存在的情况

### 使用现有测试视频

如果项目中已有测试视频，可以从以下位置复制：

```bash
# 复制火灾视频
cp datasets/fire_videos_organized/fire/mivia_fire_fire1.avi static/videos/temp.avi
ffmpeg -i static/videos/temp.avi -c:v libx264 static/videos/demo_fire.mp4
rm static/videos/temp.avi

# 复制烟雾视频
cp datasets/fire_videos_organized/smoke/mivia_smoke_*.avi static/videos/temp.avi
ffmpeg -i static/videos/temp.avi -c:v libx264 static/videos/demo_smoke.mp4
rm static/videos/temp.avi

# 复制正常视频
cp datasets/fire_videos_organized/normal/mivia_normal_*.avi static/videos/temp.avi
ffmpeg -i static/videos/temp.avi -c:v libx264 static/videos/demo_normal.mp4
rm static/videos/temp.avi
```

## 配置文件更新

准备好资源后，需要更新配置文件中的路径：

1. **system_config.json**：更新floor_plan.image路径
2. **buildings.json**：更新各建筑的floor_plan.image路径
3. **demo_resources.json**：集中管理所有资源路径（推荐）

## 注意事项

1. **文件大小**：视频文件不要太大（建议<50MB），否则影响加载速度
2. **路径格式**：使用相对路径，以`/static/`开头
3. **文件命名**：使用英文和下划线，避免中文和特殊字符
4. **版权问题**：确保使用的资源有合法授权

## 故障排查

**问题：视频无法播放**
- 检查文件格式是否为MP4（H.264）
- 检查文件路径是否正确
- 检查浏览器控制台是否有错误信息

**问题：平面图不显示**
- 检查文件格式是否为PNG
- 检查文件路径是否正确
- 检查图片尺寸是否合理

**问题：音效不播放**
- 检查浏览器是否允许自动播放音频
- 检查文件格式是否为MP3
- 音效是可选的，不影响核心功能
