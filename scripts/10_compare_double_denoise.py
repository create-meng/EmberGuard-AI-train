import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib


matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _apply_frame_ewma(frame_bgr: np.ndarray, state: Optional[np.ndarray], alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (denoised_frame, new_state)."""
    import cv2

    a = float(alpha)
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0

    if state is None or state.shape != frame_bgr.shape:
        state = frame_bgr.astype(np.float32)
    else:
        cv2.accumulateWeighted(frame_bgr, state, a)

    out = np.clip(state, 0, 255).astype(np.uint8)
    return out, state


@dataclass
class Series:
    t_sec: List[float]
    frame_diff_mae: List[float]
    yolo_max_conf: List[float]
    yolo_count: List[float]
    feat_conf: List[float]
    feat_area: List[float]
    feat_ratio: List[float]
    feat_cls: List[float]
    lstm_conf: List[float]
    lstm_raw_conf: List[float]


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def run_video_series(
    video_path: str,
    yolo_path: str,
    lstm_path: Optional[str],
    enable_feature_denoise: bool,
    enable_frame_denoise: bool,
    frame_denoise_alpha: float,
    conf_threshold: float,
    yolo_imgsz: int,
    log_every: int,
    max_frames: int,
    skip_frames: int = 0,
) -> Series:
    try:
        import cv2
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少依赖: cv2 (opencv-python)。请在你当前 python 环境里安装 opencv-python，或用已安装的环境运行脚本。"
        ) from e

    from emberguard.pipeline import FireDetectionPipeline

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0

    pipeline = FireDetectionPipeline(
        yolo_model_path=yolo_path,
        lstm_model_path=lstm_path,
        sequence_length=30,
        enable_feature_denoise=enable_feature_denoise,
    )
    try:
        pipeline.yolo_imgsz = int(yolo_imgsz)
    except Exception:
        pass
    pipeline.reset_buffer()

    t_sec: List[float] = []
    frame_diff_mae: List[float] = []
    yolo_max_conf: List[float] = []
    yolo_count: List[float] = []
    feat_conf: List[float] = []
    feat_area: List[float] = []
    feat_ratio: List[float] = []
    feat_cls: List[float] = []
    lstm_conf: List[float] = []
    lstm_raw_conf: List[float] = []

    frame_idx = 0
    ewma_state: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if max_frames > 0 and frame_idx >= max_frames:
            break

        # 统一尺寸（和你的 demo 后端一致）
        frame = cv2.resize(frame, (640, 480))

        if enable_frame_denoise:
            frame, ewma_state = _apply_frame_ewma(frame, ewma_state, frame_denoise_alpha)

        # 帧间差分指标（用于体现 frame 降噪效果，不依赖 YOLO 是否有框）
        # 注意：必须在“是否应用 frame denoise”之后计算，否则 raw/denoise 会完全重合。
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None or prev_gray.shape != gray.shape:
                diff_mae = 0.0
            else:
                diff = cv2.absdiff(gray, prev_gray)
                diff_mae = float(np.mean(diff))
            prev_gray = gray
        except Exception:
            diff_mae = 0.0

        res = pipeline.detect_frame(frame, conf_threshold=conf_threshold) or {}

        dets = res.get('yolo_detections')
        if isinstance(dets, list) and dets:
            try:
                max_conf = max(_safe_float(d.get('confidence'), 0.0) for d in dets if isinstance(d, dict))
            except Exception:
                max_conf = 0.0
            cnt = float(len(dets))
        else:
            max_conf = 0.0
            cnt = 0.0

        feats = res.get('features')
        if isinstance(feats, np.ndarray) and feats.size >= 8:
            f = feats.reshape(-1)
            f_area = _safe_float(f[4], 0.0)
            f_ratio = _safe_float(f[5], 0.0)
            f_conf = _safe_float(f[6], 0.0)
            f_cls = _safe_float(f[7], -1.0)
        else:
            f_area, f_ratio, f_conf, f_cls = 0.0, 0.0, 0.0, -1.0

        t_sec.append((frame_idx - skip_frames) / float(fps))
        frame_diff_mae.append(diff_mae)
        yolo_max_conf.append(max_conf)
        yolo_count.append(cnt)
        feat_conf.append(f_conf)
        feat_area.append(f_area)
        feat_ratio.append(f_ratio)
        feat_cls.append(f_cls)

        # LSTM：缓冲没满时字段可能不存在，统一填 0
        lstm_conf.append(_safe_float(res.get('lstm_confidence'), 0.0))
        lstm_raw_conf.append(_safe_float(res.get('lstm_raw_confidence'), 0.0))

        frame_idx += 1

        if int(log_every) > 0 and (frame_idx % int(log_every) == 0):
            try:
                print(
                    f"[{('denoise' if enable_frame_denoise or enable_feature_denoise else 'raw')}] "
                    f"frame={frame_idx} yolo_cnt={cnt:.0f} yolo_max_conf={max_conf:.3f} "
                    f"feat_conf={f_conf:.3f} lstm_conf={lstm_conf[-1]:.3f}"
                )
            except Exception:
                pass

    cap.release()

    return Series(
        t_sec=t_sec,
        frame_diff_mae=frame_diff_mae,
        yolo_max_conf=yolo_max_conf,
        yolo_count=yolo_count,
        feat_conf=feat_conf,
        feat_area=feat_area,
        feat_ratio=feat_ratio,
        feat_cls=feat_cls,
        lstm_conf=lstm_conf,
        lstm_raw_conf=lstm_raw_conf,
    )


def _plot_three_series(
    out_path: str,
    title: str,
    t1: List[float], y1: List[float], label1: str,
    t2: List[float], y2: List[float], label2: str,
    t3: List[float], y3: List[float], label3: str,
    y_label: str,
):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    except Exception:
        pass
    plt.figure(figsize=(14, 6), dpi=160)
    plt.plot(t1, y1, linewidth=1.2, alpha=0.90, label=label1)
    plt.plot(t2, y2, linewidth=1.2, alpha=0.90, linestyle='--', label=label2)
    plt.plot(t3, y3, linewidth=1.2, alpha=0.90, linestyle=':', label=label3)
    plt.title(title)
    plt.xlabel('time (sec)')
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    try:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(10))
        ax.yaxis.set_major_locator(MaxNLocator(8))
    except Exception:
        pass
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare double denoise (frame+feature) vs none on same video')

    parser.add_argument('--video', type=str, default=r"D:\a安建大\大二\下学期\比赛\挑战杯\院赛\AI消防\ultralytics-main\datasets\fire_videos_organized\fire\archive_fire2.mp4")
    parser.add_argument('--yolo', type=str, default='runs/detect/train2/weights/best.pt')
    parser.add_argument('--lstm', type=str, default='models/lstm/best.pt')

    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--imgsz', type=int, default=256, help='YOLO inference imgsz used in pipeline (default 256)')
    parser.add_argument('--max-frames', type=int, default=300, help='0 means full video; default 300 (~12s at 25fps)')
    parser.add_argument('--skip-frames', type=int, default=30, help='Skip first N frames (LSTM buffer warmup)')

    parser.add_argument('--frame-alpha', type=float, default=0.78, help='EWMA alpha for frame denoise (same as demo backend)')
    parser.add_argument('--outdir', type=str, default=os.path.join('scripts', 'outputs'))
    parser.add_argument('--log-every', type=int, default=0, help='print debug every N frames (0 disables)')

    args = parser.parse_args()

    video_path = args.video
    yolo_path = args.yolo
    lstm_path = args.lstm if args.lstm and os.path.exists(args.lstm) else None

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"YOLO模型不存在: {yolo_path}")

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Baseline: no frame denoise + no feature denoise
    s_raw = run_video_series(
        video_path=video_path,
        yolo_path=yolo_path,
        lstm_path=lstm_path,
        enable_feature_denoise=False,
        enable_frame_denoise=False,
        frame_denoise_alpha=args.frame_alpha,
        conf_threshold=args.conf,
        yolo_imgsz=args.imgsz,
        log_every=args.log_every,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )

    # 2. Only frame denoise (no feature denoise)
    s_frame = run_video_series(
        video_path=video_path,
        yolo_path=yolo_path,
        lstm_path=lstm_path,
        enable_feature_denoise=False,
        enable_frame_denoise=True,
        frame_denoise_alpha=args.frame_alpha,
        conf_threshold=args.conf,
        yolo_imgsz=args.imgsz,
        log_every=args.log_every,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )

    # 3. Double denoise: frame denoise + feature denoise
    s_double = run_video_series(
        video_path=video_path,
        yolo_path=yolo_path,
        lstm_path=lstm_path,
        enable_feature_denoise=True,
        enable_frame_denoise=True,
        frame_denoise_alpha=args.frame_alpha,
        conf_threshold=args.conf,
        yolo_imgsz=args.imgsz,
        log_every=args.log_every,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )

    def _stats(name: str, s: Series):
        try:
            det_ratio = float(sum(1 for x in s.yolo_count if x > 0)) / float(max(1, len(s.yolo_count)))
            print(f"[{name}] frames={len(s.t_sec)} det_frames_ratio={det_ratio:.3f} yolo_max_conf_max={max(s.yolo_max_conf) if s.yolo_max_conf else 0.0:.3f}")
        except Exception:
            pass

    _stats('raw', s_raw)
    _stats('frame-denoise-only', s_frame)
    _stats('double-denoise', s_double)

    # 图1：帧间差分能量（MAE）对比，用于体现 frame 降噪（不依赖 YOLO 是否有框）
    _plot_three_series(
        out_path=os.path.join(args.outdir, 'compare_denoise_frame_diff.png'),
        title='Frame difference (MAE) over time | 3 scenarios',
        t1=s_raw.t_sec, y1=s_raw.frame_diff_mae, label1='raw (no denoise)',
        t2=s_frame.t_sec, y2=s_frame.frame_diff_mae, label2='frame-denoise only',
        t3=s_double.t_sec, y3=s_double.frame_diff_mae, label3='double-denoise (frame+feature)',
        y_label='frame_diff_mae (0-255)',
    )

    # 图2：YOLO 指标对比（三组）
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=160, sharex=True)

    axes[0].plot(s_raw.t_sec, s_raw.yolo_max_conf, linewidth=1.0, alpha=0.90, label='raw')
    axes[0].plot(s_frame.t_sec, s_frame.yolo_max_conf, linewidth=1.0, alpha=0.90, linestyle='--', label='frame-denoise only')
    axes[0].plot(s_double.t_sec, s_double.yolo_max_conf, linewidth=1.0, alpha=0.90, linestyle=':', label='double-denoise')
    axes[0].set_title('YOLO max confidence over time')
    axes[0].set_ylabel('max_conf')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(s_raw.t_sec, s_raw.yolo_count, linewidth=1.0, alpha=0.90, label='raw')
    axes[1].plot(s_frame.t_sec, s_frame.yolo_count, linewidth=1.0, alpha=0.90, linestyle='--', label='frame-denoise only')
    axes[1].plot(s_double.t_sec, s_double.yolo_count, linewidth=1.0, alpha=0.90, linestyle=':', label='double-denoise')
    axes[1].set_title('YOLO detection count over time')
    axes[1].set_ylabel('count')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # delta plots: frame vs raw, double vs raw
    try:
        n = min(len(s_raw.t_sec), len(s_frame.t_sec), len(s_double.t_sec),
                len(s_raw.yolo_max_conf), len(s_frame.yolo_max_conf), len(s_double.yolo_max_conf))
        t = s_raw.t_sec[:n]
        dy_frame = [(s_frame.yolo_max_conf[i] - s_raw.yolo_max_conf[i]) for i in range(n)]
        dy_double = [(s_double.yolo_max_conf[i] - s_raw.yolo_max_conf[i]) for i in range(n)]
        dc_frame = [(s_frame.yolo_count[i] - s_raw.yolo_count[i]) for i in range(n)]
        dc_double = [(s_double.yolo_count[i] - s_raw.yolo_count[i]) for i in range(n)]
    except Exception:
        t, dy_frame, dy_double, dc_frame, dc_double = [], [], [], [], []

    axes[2].plot(t, dy_frame, linewidth=1.0, alpha=0.9, label='Δ max_conf (frame - raw)')
    axes[2].plot(t, dy_double, linewidth=1.0, alpha=0.9, linestyle='--', label='Δ max_conf (double - raw)')
    axes[2].axhline(0.0, color='#999', linewidth=1.0, alpha=0.6)
    axes[2].set_title('Delta max_conf over time (vs raw baseline)')
    axes[2].set_xlabel('time (sec)')
    axes[2].set_ylabel('delta')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    try:
        axes[0].xaxis.set_major_locator(MaxNLocator(10))
        axes[0].yaxis.set_major_locator(MaxNLocator(8))
        axes[1].xaxis.set_major_locator(MaxNLocator(10))
        axes[1].yaxis.set_major_locator(MaxNLocator(8))
        axes[2].xaxis.set_major_locator(MaxNLocator(10))
        axes[2].yaxis.set_major_locator(MaxNLocator(8))
    except Exception:
        pass

    plt.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.07, hspace=0.32)
    out2 = os.path.join(args.outdir, 'compare_denoise_yolo_metrics.png')
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out2)), exist_ok=True)
    except Exception:
        pass
    plt.savefig(out2)
    plt.close(fig)

    # 图3：LSTM 置信度对比 - 体现 Feature 降噪效果
    # Feature降噪作用于YOLO输出的特征向量，稳定LSTM时序输入
    fig3, ax3 = plt.subplots(figsize=(14, 6), dpi=160)
    ax3.plot(s_raw.t_sec, s_raw.lstm_conf, linewidth=1.2, alpha=0.90, label='raw')
    ax3.plot(s_frame.t_sec, s_frame.lstm_conf, linewidth=1.2, alpha=0.90, linestyle='--', label='frame-denoise only')
    ax3.plot(s_double.t_sec, s_double.lstm_conf, linewidth=1.2, alpha=0.90, linestyle=':', label='double-denoise')
    ax3.set_title('LSTM confidence over time | Feature denoise effect')
    ax3.set_xlabel('time (sec)')
    ax3.set_ylabel('lstm_conf')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    try:
        ax3.xaxis.set_major_locator(MaxNLocator(10))
        ax3.yaxis.set_major_locator(MaxNLocator(8))
    except Exception:
        pass
    out3 = os.path.join(args.outdir, 'compare_denoise_lstm_conf.png')
    plt.savefig(out3)
    plt.close(fig3)

    print('✓ 图已生成:')
    print(os.path.abspath(os.path.join(args.outdir, 'compare_denoise_frame_diff.png')))
    print(os.path.abspath(os.path.join(args.outdir, 'compare_denoise_yolo_metrics.png')))
    print(os.path.abspath(os.path.join(args.outdir, 'compare_denoise_lstm_conf.png')))


if __name__ == '__main__':
    main()
