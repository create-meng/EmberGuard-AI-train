(() => {
  const els = {
    statusText: document.getElementById('statusText'),
    cameraSelect: document.getElementById('cameraSelect'),
    videoImg: document.getElementById('videoImg'),
    overlay: document.getElementById('overlay'),
    camMeta: document.getElementById('camMeta'),
    lstmMeta: document.getElementById('lstmMeta'),
    lstmClass: document.getElementById('lstmClass'),
    lstmConf: document.getElementById('lstmConf'),
    yoloCount: document.getElementById('yoloCount'),
    yoloHint: document.getElementById('yoloHint'),
    alarmBadge: document.getElementById('alarmBadge'),
    sensorList: document.getElementById('sensorList'),
  };

  let cameras = [];
  let selectedCameraId = null;
  let eventSource = null;
  let lastDetectionSig = '';
  let lastDetection = null;

  function setBadge(level) {
    const el = els.alarmBadge;
    el.classList.remove('danger', 'warn', 'ok');
    if (level === 'fire') {
      el.textContent = '火焰告警';
      el.classList.add('danger');
      return;
    }
    if (level === 'smoke') {
      el.textContent = '烟雾预警';
      el.classList.add('warn');
      return;
    }
    el.textContent = '正常';
    el.classList.add('ok');
  }

  function pickAlarmFromDetection(det) {
    const pred = det?.lstm_prediction;
    if (pred === 2) return 'fire';
    if (pred === 1) return 'smoke';
    return 'normal';
  }

  function safeText(v, fallback = '-') {
    if (v === null || v === undefined || v === '') return fallback;
    return String(v);
  }

  function fmtPercent(x) {
    if (typeof x !== 'number' || Number.isNaN(x)) return '-';
    return `${Math.round(x * 100)}%`;
  }

  function ensureCanvasSize() {
    const rect = els.videoImg.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (els.overlay.width !== w) els.overlay.width = w;
    if (els.overlay.height !== h) els.overlay.height = h;
    return { w, h };
  }

  function drawBoxes(det) {
    const ctx = els.overlay.getContext('2d');
    if (!ctx) return;

    const { w, h } = ensureCanvasSize();
    ctx.clearRect(0, 0, w, h);

    const boxes = det?.yolo_detections;
    if (!Array.isArray(boxes) || boxes.length === 0) return;

    const srcW = 640;
    const srcH = 480;
    const sx = w / srcW;
    const sy = h / srcH;

    ctx.lineWidth = 2;
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.textBaseline = 'top';

    for (const b of boxes) {
      const bb = b?.bbox;
      if (!bb || bb.length !== 4) continue;
      const [x1, y1, x2, y2] = bb;
      const x = x1 * sx;
      const y = y1 * sy;
      const bw = (x2 - x1) * sx;
      const bh = (y2 - y1) * sy;

      const cls = b?.class_name || '';
      const conf = typeof b?.confidence === 'number' ? b.confidence : null;
      const label = conf === null ? cls : `${cls} ${Math.round(conf * 100)}%`;

      const color = cls === 'fire' ? '#ff3b30' : '#ffcc00';
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.strokeRect(x, y, bw, bh);

      if (label) {
        const pad = 3;
        const metrics = ctx.measureText(label);
        const th = 14;
        const tw = Math.ceil(metrics.width) + pad * 2;
        const tx = Math.max(0, Math.min(w - tw, x));
        const ty = Math.max(0, y - th);
        ctx.fillRect(tx, ty, tw, th);
        ctx.fillStyle = '#000';
        ctx.fillText(label, tx + pad, ty + 1);
      }
    }
  }

  function renderSensors(sensors) {
    if (!Array.isArray(sensors)) sensors = [];

    els.sensorList.innerHTML = '';

    for (const s of sensors) {
      const row = document.createElement('div');
      row.className = 'card';

      const name = safeText(s?.name || s?.id);
      const unit = safeText(s?.unit, '');
      const value = (s?.current_value ?? s?.value ?? 0);
      const status = safeText(s?.status, 'normal');
      const ts = safeText(s?.timestamp, '');

      row.innerHTML = `
        <div class="row">
          <div>
            <div class="k">${name}</div>
            <div class="v" style="font-size:18px">${safeText(value)}${unit}</div>
            <div class="s">${ts}</div>
          </div>
          <span class="badge2 ${status === 'alert' ? 'danger' : 'ok'}">${status === 'alert' ? '告警' : '正常'}</span>
        </div>
      `;

      els.sensorList.appendChild(row);
    }

    if (sensors.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'card';
      empty.innerHTML = '<div class="k">暂无传感器数据</div>';
      els.sensorList.appendChild(empty);
    }
  }

  function renderDetection(det) {
    if (!det) {
      lastDetection = null;
      els.lstmClass.textContent = '无数据';
      els.lstmConf.textContent = '置信度 -';
      els.yoloCount.textContent = '0';
      els.yoloHint.textContent = '-';
      setBadge('normal');
      drawBoxes(null);
      return;
    }

    lastDetection = det;

    const alarm = pickAlarmFromDetection(det);
    setBadge(alarm);

    const buf = typeof det.buffer_size === 'number' ? det.buffer_size : 0;
    if (buf > 0 && buf < 30) {
      els.lstmClass.textContent = `预热中 ${buf}/30`;
      els.lstmConf.textContent = 'LSTM 需要缓存 30 帧后才会稳定输出';
    } else {
      els.lstmClass.textContent = safeText(det.lstm_class_name, '无火');
      els.lstmConf.textContent = `置信度 ${fmtPercent(det.lstm_confidence)}`;
    }

    const yolo = Array.isArray(det.yolo_detections) ? det.yolo_detections : [];
    els.yoloCount.textContent = String(yolo.length);
    els.yoloHint.textContent = safeText(det.timestamp, '-');

    drawBoxes(det);
  }

  function stopEvents() {
    if (eventSource) {
      try { eventSource.close(); } catch (e) {}
      eventSource = null;
    }
  }

  function startEvents(cameraId) {
    stopEvents();
    const url = `/demo/events?camera_id=${encodeURIComponent(cameraId)}`;
    eventSource = new EventSource(url);

    eventSource.onmessage = (evt) => {
      if (!evt?.data) return;
      let payload;
      try {
        payload = JSON.parse(evt.data);
      } catch (e) {
        return;
      }

      const cam = payload?.camera;
      const det = cam?.last_detection;
      const sig = det ? JSON.stringify({
        t: det.timestamp,
        p: det.lstm_prediction,
        c: det.lstm_confidence,
        y: det.yolo_detections
      }) : '';

      if (sig !== lastDetectionSig) {
        lastDetectionSig = sig;
        renderDetection(det);
      }

      renderSensors(payload?.sensors);
      els.statusText.textContent = `SSE 在线 | ${safeText(payload?.ts)}`;
      els.camMeta.textContent = `${safeText(cam?.camera_id)} | ${safeText(cam?.status)}`;
      if (det && typeof det.buffer_size === 'number') {
        els.lstmMeta.textContent = `LSTM 缓冲: ${det.buffer_size}/30`;
      } else {
        els.lstmMeta.textContent = '';
      }
    };

    eventSource.onerror = () => {
      els.statusText.textContent = 'SSE 断开，自动重连中...';
    };
  }

  function setCamera(cameraId) {
    selectedCameraId = String(cameraId);
    els.videoImg.src = `/demo/stream/${encodeURIComponent(selectedCameraId)}`;
    lastDetectionSig = '';
    renderDetection(null);
    startEvents(selectedCameraId);
  }

  async function loadCameras() {
    els.statusText.textContent = '加载摄像头列表...';
    const resp = await fetch('/demo/cameras');
    const json = await resp.json();
    cameras = json?.data || [];

    els.cameraSelect.innerHTML = '';
    for (const c of cameras) {
      const opt = document.createElement('option');
      opt.value = String(c.id);
      opt.textContent = `${c.name || c.id}`;
      els.cameraSelect.appendChild(opt);
    }

    if (cameras.length > 0) {
      const first = String(cameras[0].id);
      els.cameraSelect.value = first;
      setCamera(first);
      els.statusText.textContent = 'Demo 已就绪';
    } else {
      els.statusText.textContent = '未发现摄像头（请检查 buildings/demo/config.json）';
    }
  }

  els.cameraSelect.addEventListener('change', (e) => {
    const v = e.target.value;
    if (!v) return;
    setCamera(v);
  });

  window.addEventListener('resize', () => {
    // resize 时重画一次框
    if (lastDetection) {
      drawBoxes(lastDetection);
    } else {
      drawBoxes(null);
    }
  });

  // 启动
  loadCameras().catch(() => {
    els.statusText.textContent = '初始化失败，请查看后端控制台';
  });
})();
