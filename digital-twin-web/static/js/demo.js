(() => {
  const els = {
    statusText: document.getElementById('statusText'),
    videoImg: document.getElementById('videoImg'),
    overlay: document.getElementById('overlay'),
    camMeta: document.getElementById('camMeta'),
    lstmClass: document.getElementById('lstmClass'),
    lstmConf: document.getElementById('lstmConf'),
    yoloCount: document.getElementById('yoloCount'),
    yoloHint: document.getElementById('yoloHint'),
    alarmBadge: document.getElementById('alarmBadge'),
    sensorList: document.getElementById('sensorList'),
    headerTime: document.getElementById('headerTime'),
    footerAlarm: document.getElementById('footerAlarm'),
    footerDecision: document.getElementById('footerDecision'),
    footerYolo: document.getElementById('footerYolo'),
    finalStatus: document.getElementById('finalStatus'),
    finalReason: document.getElementById('finalReason'),
  };

  const ALARM_CFG = {
    windowSize: 20,
    ewmaAlpha: 0.35,
    yoloFireMinConf: 0.5,
    yoloFireStrongConf: 0.72,
    yoloFireStrongMinArea: 0.008,
    yoloSmokeMinConf: 0.55,
    yoloSmokeStrongConf: 0.7,
    yoloSmokeStrongMinArea: 0.01,
    voteMinConfSmoke: 0.55,
    voteMinConfFire: 0.45,
    onFireScore: 0.62,
    onFireScoreNoYolo: 0.9,
    offFireScore: 0.42,
    onSmokeScore: 0.68,
    onSmokeScoreNoYolo: 0.85,
    offSmokeScore: 0.52,
    kFire: 2,
    kFireNoYolo: 6,
    kSmoke: 4,
    kSmokeNoYolo: 8,
    yoloSmokeConsecutive: 2,
    yoloStrongConsecutive: 1,
    holdFireMs: 2500,
    holdSmokeMs: 600,
  };

  let eventSource = null;
  let lastDetectionSig = '';
  let lastDetection = null;

  let streamRetryTimer = null;
  let streamRetryMs = 400;
  let streamRefreshTimer = null;

  let alarmState = 'normal';
  let alarmSince = 0;
  let hist = [];
  let ewmaFire = 0;
  let ewmaSmoke = 0;
  let lastDecisionReason = '-';

  function setBadge(level) {
    const el = els.alarmBadge;
    el.classList.remove('badge--danger', 'badge--warning', 'badge--success', 'badge--neutral');
    if (level === 'fire') {
      el.textContent = '火焰告警';
      el.classList.add('badge--danger');
      if (els.footerAlarm) els.footerAlarm.textContent = '火焰告警';
      return;
    }
    if (level === 'smoke') {
      el.textContent = '烟雾预警';
      el.classList.add('badge--warning');
      if (els.footerAlarm) els.footerAlarm.textContent = '烟雾预警';
      return;
    }
    el.textContent = '系统正常';
    el.classList.add('badge--success');
    if (els.footerAlarm) els.footerAlarm.textContent = '系统正常';
  }

  function alarmName(level) {
    if (level === 'fire') return '火焰告警';
    if (level === 'smoke') return '烟雾预警';
    return '系统正常';
  }

  function normalizeLstmName(name) {
    const s = safeText(name, '');
    if (!s) return '正常';
    if (s === '无火') return '正常';
    if (s.toLowerCase?.() === 'normal') return '正常';
    return s;
  }

  function yoloEvidence(det) {
    // 优先对齐 LSTM 的真实输入：后端 features 是 get_best_detection() 的 8 维向量
    // features: [cx, cy, w, h, area, ratio, conf, cls]
    const feat = det?.features;
    if (Array.isArray(feat) && feat.length === 8) {
      const conf = typeof feat[6] === 'number' ? feat[6] : 0;
      const cls = typeof feat[7] === 'number' ? feat[7] : -1;
      const areaNorm = typeof feat[4] === 'number' ? feat[4] : 0;
      const isSmoke = cls === 0;
      const isFire = cls === 1;
      const fire = (isFire && conf >= ALARM_CFG.yoloFireMinConf) ? 1 : 0;
      const fireStrong = (isFire && conf >= ALARM_CFG.yoloFireStrongConf && areaNorm >= ALARM_CFG.yoloFireStrongMinArea) ? 1 : 0;
      const smoke = (isSmoke && conf >= ALARM_CFG.yoloSmokeMinConf) ? 1 : 0;
      const smokeStrong = (isSmoke && conf >= ALARM_CFG.yoloSmokeStrongConf && areaNorm >= ALARM_CFG.yoloSmokeStrongMinArea) ? 1 : 0;
      return { fire, fireStrong, smoke, smokeStrong };
    }

    // fallback：使用 yolo_detections（用于兼容没有 features 的情况）
    const arr = Array.isArray(det?.yolo_detections) ? det.yolo_detections : [];
    let fire = 0;
    let fireStrong = 0;
    let smoke = 0;
    let smokeStrong = 0;
    for (const d of arr) {
      const cls = d?.class_name;
      const conf = typeof d?.confidence === 'number' ? d.confidence : 0;
      const bb = d?.bbox;
      let area = 0;
      if (Array.isArray(bb) && bb.length === 4) {
        const [x1, y1, x2, y2] = bb;
        const w = Math.max(0, x2 - x1);
        const h = Math.max(0, y2 - y1);
        area = (w * h) / (640 * 480);
      }

      if (cls === 'fire') {
        if (conf >= ALARM_CFG.yoloFireMinConf) fire += 1;
        if (conf >= ALARM_CFG.yoloFireStrongConf && area >= ALARM_CFG.yoloFireStrongMinArea) fireStrong += 1;
      }
      if (cls === 'smoke') {
        if (conf >= ALARM_CFG.yoloSmokeMinConf) smoke += 1;
        if (conf >= ALARM_CFG.yoloSmokeStrongConf && area >= ALARM_CFG.yoloSmokeStrongMinArea) smokeStrong += 1;
      }
    }
    return { fire, fireStrong, smoke, smokeStrong };
  }

  function countYoloFire(det) {
    return yoloEvidence(det).fire;
  }

  function resetAlarm() {
    alarmState = 'normal';
    alarmSince = 0;
    hist = [];
    ewmaFire = 0;
    ewmaSmoke = 0;
  }

  function updateAlarmFromDetection(det) {
    if (!det) {
      resetAlarm();
      lastDecisionReason = '-';
      return 'normal';
    }

    const now = Date.now();
    const pred = det?.lstm_prediction;
    const conf = typeof det?.lstm_confidence === 'number' ? det.lstm_confidence : 0;
    const y = yoloEvidence(det);
    const yoloFire = y.fire;
    const yoloSmoke = y.smoke;

    const prev = hist.length > 0 ? hist[hist.length - 1] : null;

    // YOLO 出现强火焰证据时快速触发（响应优先）
    if (y.fireStrong > 0) {
      alarmState = 'fire';
      alarmSince = now;
      lastDecisionReason = `YOLO 强证据(${y.fireStrong})`;
      return 'fire';
    }

    // YOLO fire 连续两帧出现时快速触发（避免等 1-3 秒累计）
    if (yoloFire > 0 && prev?.yoloFire > 0) {
      alarmState = 'fire';
      alarmSince = now;
      lastDecisionReason = 'YOLO 连续两帧火焰框';
      return 'fire';
    }

    const fireProb = pred === 2 ? conf : 0;
    const smokeProb = pred === 1 ? conf : 0;
    ewmaFire = ALARM_CFG.ewmaAlpha * fireProb + (1 - ALARM_CFG.ewmaAlpha) * ewmaFire;
    ewmaSmoke = ALARM_CFG.ewmaAlpha * smokeProb + (1 - ALARM_CFG.ewmaAlpha) * ewmaSmoke;

    hist.push({ pred, conf, yoloFire, yoloFireStrong: y.fireStrong, yoloSmoke, yoloSmokeStrong: y.smokeStrong, t: now });
    if (hist.length > ALARM_CFG.windowSize) hist.shift();

    let fireVotes = 0;
    let smokeVotes = 0;
    let yoloHits = 0;
    let yoloStrongHits = 0;
    let yoloSmokeHits = 0;
    let yoloSmokeStrongHits = 0;
    for (const h of hist) {
      if (h.yoloFire > 0) yoloHits += 1;
      if (h.yoloFireStrong > 0) yoloStrongHits += 1;
      if (h.yoloSmoke > 0) yoloSmokeHits += 1;
      if (h.yoloSmokeStrong > 0) yoloSmokeStrongHits += 1;
      if (h.pred === 2 && h.conf >= ALARM_CFG.voteMinConfFire) fireVotes += 1;
      if (h.pred === 1 && h.conf >= ALARM_CFG.voteMinConfSmoke) smokeVotes += 1;
    }

    const win = Math.max(1, hist.length);
    const voteFireScore = fireVotes / win;
    const voteSmokeScore = smokeVotes / win;
    const yoloScore = Math.min(1, yoloHits / win);
    const yoloStrongScore = Math.min(1, yoloStrongHits / win);
    const yoloSmokeScore = Math.min(1, yoloSmokeHits / win);
    const yoloSmokeStrongScore = Math.min(1, yoloSmokeStrongHits / win);

    const fireScore = Math.max(ewmaFire, 0.55 * voteFireScore + 0.30 * yoloScore + 0.15 * yoloStrongScore);
    const smokeScore = Math.max(ewmaSmoke, 0.75 * voteSmokeScore + 0.20 * yoloSmokeScore + 0.05 * yoloSmokeStrongScore);

    const elapsed = alarmSince ? (now - alarmSince) : 0;

    if (alarmState === 'fire') {
      if (elapsed < ALARM_CFG.holdFireMs) return 'fire';
      if (fireScore <= ALARM_CFG.offFireScore && fireVotes === 0 && yoloHits === 0) {
        alarmState = 'normal';
        alarmSince = now;
        lastDecisionReason = `解除: fireScore=${fireScore.toFixed(2)}`;
        return 'normal';
      }
      lastDecisionReason = `保持: fireScore=${fireScore.toFixed(2)} yolo=${yoloHits}`;
      return 'fire';
    }

    if (alarmState === 'smoke') {
      if (elapsed < ALARM_CFG.holdSmokeMs) return 'smoke';
      if (smokeScore <= ALARM_CFG.offSmokeScore && smokeVotes === 0) {
        alarmState = 'normal';
        alarmSince = now;
        lastDecisionReason = `解除: smokeScore=${smokeScore.toFixed(2)}`;
        return 'normal';
      }
      // YOLO 强证据连续出现时快速升级为火焰告警
      const recent = hist.slice(-ALARM_CFG.yoloStrongConsecutive);
      const strongConsecutive = recent.length >= ALARM_CFG.yoloStrongConsecutive && recent.every(x => x.yoloFireStrong > 0);
      if (strongConsecutive) {
        alarmState = 'fire';
        alarmSince = now;
        lastDecisionReason = 'YOLO 强证据(窗口)';
        return 'fire';
      }

      // 禁止 LSTM-only fire：没有 YOLO 火焰证据时不允许触发 fire（防止行人/光照误报）
      if (yoloHits > 0) {
        const fireOnThreshold = ALARM_CFG.onFireScore;
        const fireVotesNeed = ALARM_CFG.kFire;
        if (fireScore >= fireOnThreshold || fireVotes >= fireVotesNeed || yoloHits >= 2) {
          alarmState = 'fire';
          alarmSince = now;
          lastDecisionReason = `融合: fireScore=${fireScore.toFixed(2)} yolo=${yoloHits}`;
          return 'fire';
        }
      }
      lastDecisionReason = `保持烟雾: smokeScore=${smokeScore.toFixed(2)}`;
      return 'smoke';
    }

    // YOLO 强证据连续出现时快速触发火焰告警
    const recent = hist.slice(-ALARM_CFG.yoloStrongConsecutive);
    const strongConsecutive = recent.length >= ALARM_CFG.yoloStrongConsecutive && recent.every(x => x.yoloFireStrong > 0);
    if (strongConsecutive) {
      alarmState = 'fire';
      alarmSince = now;
      lastDecisionReason = 'YOLO 强证据(窗口)';
      return 'fire';
    }

    // 禁止 LSTM-only fire：必须有 YOLO 火焰证据
    if (yoloHits > 0) {
      if (fireScore >= ALARM_CFG.onFireScore || fireVotes >= ALARM_CFG.kFire || yoloHits >= 2) {
        alarmState = 'fire';
        alarmSince = now;
        lastDecisionReason = `融合: fireScore=${fireScore.toFixed(2)} yolo=${yoloHits}`;
        return 'fire';
      }
    }

    // Smoke: 允许“及时发现”，但禁止“偶发”：
    // - 如果窗口内有 YOLO smoke 证据且满足连续性，则使用较低门槛（更快）
    // - 如果没有 YOLO smoke 证据，则必须更高分数 + 更多投票（更稳）
    const hasYoloSmoke = yoloSmokeHits > 0;
    const recentSmoke = hist.slice(-ALARM_CFG.yoloSmokeConsecutive);
    const smokeConsecutive = recentSmoke.length >= ALARM_CFG.yoloSmokeConsecutive && recentSmoke.every(x => x.yoloSmoke > 0);

    const smokeOnScore = (hasYoloSmoke && smokeConsecutive) ? ALARM_CFG.onSmokeScore : ALARM_CFG.onSmokeScoreNoYolo;
    const smokeVotesNeed = (hasYoloSmoke && smokeConsecutive) ? ALARM_CFG.kSmoke : ALARM_CFG.kSmokeNoYolo;

    if (smokeScore >= smokeOnScore && smokeVotes >= smokeVotesNeed) {
      alarmState = 'smoke';
      alarmSince = now;
      lastDecisionReason = `融合: smokeScore=${smokeScore.toFixed(2)}`;
      return 'smoke';
    }

    lastDecisionReason = `normal: ewmaF=${ewmaFire.toFixed(2)} ewmaS=${ewmaSmoke.toFixed(2)}`;

    return 'normal';
  }

  // Backward-compatible wrapper (renderDetection still calls this)
  function pickAlarmFromDetection(det) {
    return updateAlarmFromDetection(det);
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
      const status = safeText(s?.status, 'normal');
      row.className = status === 'alert' ? 'sensor-item alert' : 'sensor-item';

      const name = safeText(s?.name || s?.id);
      const unit = safeText(s?.unit, '');
      const value = (s?.current_value ?? s?.value ?? 0);
      const ts = safeText(s?.timestamp, '');

      row.innerHTML = `
        <div class="sensor-info">
          <h4>${name}</h4>
          <p>${ts}</p>
        </div>
        <div class="sensor-value" style="color: ${status === 'alert' ? 'var(--color-error)' : 'var(--color-success)'}">
          ${safeText(value)}${unit}
        </div>
      `;

      els.sensorList.appendChild(row);
    }

    if (sensors.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'sensor-item';
      empty.innerHTML = '<div class="k">暂无传感器数据</div>';
      els.sensorList.appendChild(empty);
    }
  }

  function renderDetection(det) {
    if (!det) {
      lastDetection = null;
      els.lstmClass.textContent = '无数据';
      els.lstmClass.classList.remove('normal', 'smoke', 'fire');
      els.lstmConf.textContent = '置信度 -';
      els.yoloCount.textContent = '0';
      els.yoloHint.textContent = '-';
      setBadge('normal');
      if (els.footerDecision) els.footerDecision.textContent = '-';
      if (els.finalStatus) els.finalStatus.textContent = '-';
      if (els.finalReason) els.finalReason.textContent = '-';
      if (els.footerYolo) els.footerYolo.textContent = '0';
      drawBoxes(null);
      return;
    }

    lastDetection = det;

    const alarm = pickAlarmFromDetection(det);
    setBadge(alarm);

    if (els.finalStatus) {
      els.finalStatus.textContent = alarmName(alarm);
      els.finalStatus.classList.remove('normal', 'smoke', 'fire');
      if (alarm === 'fire') els.finalStatus.classList.add('fire');
      else if (alarm === 'smoke') els.finalStatus.classList.add('smoke');
      else els.finalStatus.classList.add('normal');
    }
    if (els.finalReason) els.finalReason.textContent = lastDecisionReason || '-';
    if (els.footerDecision) els.footerDecision.textContent = alarmName(alarm);

    els.lstmClass.classList.remove('normal', 'smoke', 'fire');
    if (alarm === 'fire') els.lstmClass.classList.add('fire');
    else if (alarm === 'smoke') els.lstmClass.classList.add('smoke');
    else els.lstmClass.classList.add('normal');

    const buf = typeof det.buffer_size === 'number' ? det.buffer_size : 0;
    if (buf > 0 && buf < 30) {
      els.lstmClass.textContent = `预热中 ${buf}/30`;
      els.lstmConf.textContent = 'LSTM 需要缓存 30 帧后才会稳定输出';
    } else {
      els.lstmClass.textContent = normalizeLstmName(det.lstm_class_name);
      els.lstmConf.textContent = `置信度 ${fmtPercent(det.lstm_confidence)}`;
    }

    

    const yolo = Array.isArray(det.yolo_detections) ? det.yolo_detections : [];
    const yoloFireCount = yolo.filter(d => d?.class_name === 'fire').length;
    els.yoloCount.textContent = String(yoloFireCount);
    els.yoloHint.textContent = det.infer_ms ? `帧耗时: ${det.infer_ms}ms` : safeText(det.timestamp, '-');
    if (els.footerYolo) els.footerYolo.textContent = String(yoloFireCount);

    drawBoxes(det);
  }

  function stopEvents() {
    try { eventSource.close(); } catch (e) {}
    eventSource = null;
  }

  function refreshStream({ immediate = true } = {}) {
    if (!els.videoImg) return;

    const apply = () => {
      const url = `/demo/stream?_t=${Date.now()}`;
      els.videoImg.src = url;
    };

    if (immediate) apply();
    else setTimeout(apply, 60);
  }

  function scheduleStreamRetry() {
    if (streamRetryTimer) return;
    const delay = Math.min(6000, streamRetryMs);
    streamRetryTimer = setTimeout(() => {
      streamRetryTimer = null;
      refreshStream({ immediate: true });
      streamRetryMs = Math.min(8000, Math.round(streamRetryMs * 1.6));
    }, delay);
  }

  function startStreamWatchdog() {
    // 部分浏览器在后端重启后 MJPEG 不一定触发 onerror；定期轻量刷新一次避免“卡住”。
    if (streamRefreshTimer) clearInterval(streamRefreshTimer);
    streamRefreshTimer = setInterval(() => {
      refreshStream({ immediate: true });
    }, 60000);
  }

  function initStreamHandlers() {
    if (!els.videoImg) return;

    els.videoImg.addEventListener('error', () => {
      scheduleStreamRetry();
    });

    els.videoImg.addEventListener('load', () => {
      streamRetryMs = 400;
    });
  }

  function startEvents() {
    stopEvents();
    const url = `/demo/events?_t=${Date.now()}`;
    eventSource = new EventSource(url);
    
    eventSource.onopen = () => {
      els.statusText.textContent = '实时同步中';
      els.statusText.style.color = 'var(--color-success)';

      // SSE 恢复时，顺便刷新一下 MJPEG（避免需要手动刷新页面）
      streamRetryMs = 400;
      refreshStream({ immediate: false });
    };

    eventSource.onerror = () => {
      els.statusText.textContent = '连接断开，重连中...';
      els.statusText.style.color = 'var(--color-error)';
      stopEvents();

      // 后端重启/断开时同时触发 stream 重试
      scheduleStreamRetry();
      setTimeout(startEvents, 1000);
    };

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
    };
  }

  function initSingleCamera() {
    initStreamHandlers();
    refreshStream({ immediate: true });
    startStreamWatchdog();
    lastDetectionSig = '';
    renderDetection(null);
    startEvents();
  }

  window.addEventListener('resize', () => {
    // resize 时重画一次框
    if (lastDetection) {
      drawBoxes(lastDetection);
    } else {
      drawBoxes(null);
    }
  });

  // 启动
  initSingleCamera();

  function updateClock() {
    if (!els.headerTime) return;
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    els.headerTime.textContent = `${hh}:${mm}:${ss}`;
  }

  updateClock();
  setInterval(updateClock, 1000);
})();
