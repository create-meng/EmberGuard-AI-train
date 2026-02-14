# 全量重构追踪文档（Demo 展示优先 + 真实 YOLO+LSTM）

> 目标：不卡顿、秒开、效果好看；同时必须真实运行 YOLO+LSTM（pt 模型），并在前端稳定展示检测框与趋势传感器。

---

## 0. 当前问题复盘（必须明确根因）

### 0.1 现状
- 前端视频链路曾出现“长时间后才有数据”的现象。
- 画面出现过卡顿、延迟、检测框不稳定/不显示。
- 传感器数据经常为 0 或不更新。

### 0.2 关键根因假设（待验证/已验证）
- **高频渲染**：前端使用 `requestAnimationFrame` 循环 + `getBoundingClientRect` 频繁测量 + canvas 重绘，主线程压力大导致卡顿。
- **数据通道耦合**：视频传输/播放与推理/状态推送混在同一个通道和更新频率上，导致 UI 更新过密。
- **推理链路不稳定**：YOLO+LSTM 推理耗时不确定，且 LSTM 需要 30 帧缓冲，导致“早期无数据”。

---

## 1. 重构总体设计（最终要落地的架构）

### 1.1 控制面 / 数据面 / 渲染面分离
- **播放链路（数据面 - Video Plane）**：浏览器用 `<video>` 播放稳定的视频源（推荐 MP4 loop；也支持 RTSP->转码方案，但 Demo 先 MP4）。
- **推理链路（Compute Plane）**：后端独立 Worker 从同一个视频源抽帧，真实运行 YOLO+LSTM（pt）。
- **结果链路（Data Plane - Results）**：低频（建议 5Hz~10Hz）推送检测结果与传感器趋势（SSE 或 WebSocket）。
- **前端渲染（Render Plane）**：仅在“新结果到来/尺寸变化”时重绘 canvas，不做无限循环重绘。

### 1.2 真实模型使用权威参考
- 以 `scripts/8_detect_with_lstm.py` 为权威：
  - `FireDetectionPipeline(yolo_model_path, lstm_model_path, sequence_length=30)`
  - 每帧 `pipeline.detect_frame(frame, conf_threshold=...)`
  - 结果字段：`yolo_detections`, `lstm_prediction`, `lstm_class_name`, `lstm_confidence`, `lstm_probabilities`, `buffer_size`

---

## 2. 需要新增/改造的接口清单

### 2.1 后端 HTTP
- `GET /demo/video/<camera_id>`
  - 返回 MP4 文件（或静态目录映射），用于 `<video src>`。
- `GET /demo/events`
  - SSE：以固定频率推送 JSON（检测结果 + 传感器 + 告警摘要）。
- `POST /demo/control`
  - 控制 Demo 场景：开始/暂停推理、切换视频源、调整推理 FPS、切换传感器脚本。

### 2.2 前端
- 视频组件改为 `<video autoplay muted loop playsinline>`。
- 叠加框组件：canvas 覆盖 video，数据到来时重绘。
- 传感器展示：卡片 + 迷你折线（最近 N 秒），告警时有动效。

---

## 3. 实现计划（按阶段）

### 阶段 A：建立 Demo-only 展示链路（不影响现有接口）
- [ ] 新增 `REFACTOR_TRACKING.md`（本文档）并持续更新。
- [ ] 后端新增 `/demo/video/<camera_id>`（MP4 播放）。
- [ ] 前端新增 Demo 播放模式：不再依赖 MJPEG `<img>`。

### 阶段 B：真实 YOLO+LSTM 推理 Worker（抽帧、低频结果）
- [ ] 新增推理 Worker（每个 camera 独立 pipeline，避免 LSTM 缓冲串台）。
- [ ] 推理结果写入内存缓存（按 camera_id 保存 last_result + timestamp）。
- [ ] SSE/WS 推送：5Hz~10Hz（仅 JSON）。

### 阶段 C：传感器趋势模拟（好看）
- [ ] 传感器值不再是纯随机：改为“状态机 + 噪声 + 回落曲线”。
- [ ] 可选：由 LSTM 结果驱动传感器进入上升/回落段。

### 阶段 D：前端 UI 重做（好看 + 丝滑）
- [ ] 大屏布局：左传感器/中视频/右事件。
- [ ] 告警动效（CSS）：danger 红色呼吸、边框闪烁、顶部告警条。
- [ ] 绘制策略：仅 data-driven redraw。

---

## 4. 性能与稳定性约束（硬指标）
- 推理频率：默认 5fps（可调），绝不与播放帧率绑定。
- 结果推送频率：默认 5Hz（可调），避免 Vue 响应式更新过密。
- 前端绘制：无 `requestAnimationFrame` 无限循环重绘。

---

## 5. 验证清单（每完成一阶段都要勾选）
- [ ] 页面首开 2 秒内出现视频。
- [ ] 10 秒内出现首个 LSTM 结果（考虑 30 帧缓冲，必要时推理端预热/填充）。
- [ ] 检测框稳定显示（不闪烁/不漂移明显）。
- [ ] 传感器值非 0 且有趋势（上升/回落）。
- [ ] 长时间运行（10 分钟）不卡顿。

---

## 6. 变更记录（按提交/日期追加）

### 2026-02-14
- 建立本文档，确定全量重构目标与架构：播放/推理解耦 + 低频结果流 + data-driven redraw。

### 2026-02-14（Demo 新链路落地）
- 后端新增：`GET /demo/video/<camera_id>`：为前端 `<video>` 提供稳定视频播放源（来自建筑配置的 `demo_video` 文件路径）。
- 后端新增：`GET /demo/events?camera_id=...`：SSE 低频（默认 5Hz）推送 `last_detection` + `sensors` 快照。
- 前端迁移：视频弹窗从 MJPEG `<img>` 切换为 `<video>`，检测框绘制改为“数据到来才重绘”，移除无限 `requestAnimationFrame` 循环。
- CSS 调整：`.video-frame` 设置为 `position: relative` 以支持 canvas 覆盖层。
- 兼容性：若浏览器不支持播放视频格式（如 `.avi`），弹窗会自动从 `<video>` 回退到 MJPEG（`/stream/<camera_id>`）。
- SSE：若前端尚未加载传感器列表，收到 SSE 快照会直接填充传感器数组，避免长期显示 0。

### 2026-02-14（Demo-only 重写：弃用旧前后端）
- 后端重写：`backend/app.py` 变为 Demo-only Flask（不再使用 Socket.IO / APIRouter）。
- 后端仅保留路由：
  - `GET /`：渲染 `frontend/demo.html`
  - `GET /demo/cameras`：摄像头列表
  - `GET /demo/stream/<camera_id>`：MJPEG 视频流（必出画面）
  - `GET /demo/events?camera_id=...`：SSE 推送推理结果与传感器
  - `GET /stream/<camera_id>`：兼容别名
- 前端新增：`frontend/demo.html` + `static/js/demo.js`，不再加载旧 `static/js/app.js` 与 Vue 组件。
