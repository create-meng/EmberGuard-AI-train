(() => {
  const EXPERIMENT_PROFILE = (typeof window !== 'undefined' && window.EXPERIMENT_PROFILE) ? window.EXPERIMENT_PROFILE : 'yolo_lstm_denoise_fusion';

  const PROFILE_CFG = {
    yolo: { fusion: false, source: 'yolo' },
    yolo_lstm: { fusion: false, source: 'lstm' },
    yolo_lstm_denoise: { fusion: false, source: 'lstm' },
    yolo_lstm_fusion: { fusion: true, source: 'fusion' },
    yolo_lstm_denoise_fusion: { fusion: true, source: 'fusion' },
  };

  const _profile = PROFILE_CFG[EXPERIMENT_PROFILE] || PROFILE_CFG.yolo_lstm_denoise_fusion;
  const els = {
    videoImg: document.getElementById('videoImg'),
    overlay: document.getElementById('overlay'),
    siteIntroText: document.getElementById('siteIntroText'),
    sensorList: document.getElementById('sensorList'),
    headerTime: document.getElementById('headerTime'),
    finalStatus: document.getElementById('finalStatus'),
    finalReason: document.getElementById('finalReason'),
    envTemp: document.getElementById('envTemp'),
    envHum: document.getElementById('envHum'),
    metricYoloLatency: document.getElementById('metricYoloLatency'),
    metricFps: document.getElementById('metricFps'),
    metricLstm: document.getElementById('metricLstm'),
  };

  const showTechDetails = !!(typeof window !== 'undefined' && window.SHOW_TECH_DETAILS_DEFAULT);

  const dom = {
    gaugeArcs: document.querySelectorAll('.dt-gauge__arc'),
    sparkSvg: document.querySelector('.dt-sparkline'),
    sparkLine: null,
    sparkFill: null,
    sparkHint: null,
    donutShell: document.querySelector('.dt-donut-shell'),
    donutArc: null,
    donutValue: null,
    donutLabel: null,
  };

  if (dom.sparkSvg) {
    dom.sparkLine = dom.sparkSvg.querySelector('.dt-sparkline__line');
    dom.sparkFill = dom.sparkSvg.querySelector('.dt-sparkline__fill');
    dom.sparkHint = dom.sparkSvg.querySelector('.dt-sparkline__hint');
  }
  if (dom.donutShell) {
    dom.donutArc = dom.donutShell.querySelector('.dt-donut__glow');
    dom.donutValue = dom.donutShell.querySelector('.dt-donut__value');
    dom.donutLabel = dom.donutShell.querySelector('.dt-donut__label');
  }

  const ALARM_CFG = {
    windowSize: 20,
    onFireRatio: 0.35,
    offFireRatio: 0.18,
    onSmokeRatio: 0.45,
    offSmokeRatio: 0.22,
    yoloFireMinConf: 0.25,
    yoloFireStrongConf: 0.7,
    yoloFireStrongMinArea: 0.008,
    yoloSmokeMinConf: 0.25,
    yoloSmokeStrongConf: 0.7,
    yoloSmokeStrongMinArea: 0.01,
  };

  let eventSource = null;
  let lastDetectionSig = '';
  let lastDetection = null;

  let streamRetryTimer = null;
  let streamRetryMs = 400;
  let streamRefreshTimer = null;

  // 为每个摄像头维护独立的状态
  const cameraStates = {}; // key: cameraId, value: { alarmState, alarmSince, hist, ewmaFire, ewmaSmoke, lastDecisionReason, lastDetectionSig }

  // 当前选中摄像头的状态（用于UI显示）
  let alarmState = 'normal';
  let alarmSince = 0;
  let hist = [];
  let ewmaFire = 0;
  let ewmaSmoke = 0;
  let lastDecisionReason = '-';

  const ui = {
    videoModal: document.getElementById('videoModal'),
    videoModalClose: document.getElementById('videoModalClose'),
    videoModalTitle: document.getElementById('videoModalTitle'),
    videoModalContent: document.getElementById('videoModalContent'),
    videoModalDragHandle: document.getElementById('videoModalDragHandle'),
    camLayer: document.getElementById('camLayer'),
    alarmList: document.getElementById('alarmList'),
  };

  const elsExtra = {
    metricAlarmFire: document.getElementById('metricAlarmFire'),
    metricAlarmSmoke: document.getElementById('metricAlarmSmoke'),
    metricAlarmTotal: document.getElementById('metricAlarmTotal'),
  };

  let selectedCameraId = null;
  let lastUiAlarm = null;
  let currentAlarmLevel = 'normal';

  const TREND_CFG = {
    windowSize: 140,
    visiblePoints: 120,
    yMax: 6,
    ewmaAlpha: 0.22,
  };

  let trendSeries = [];

  const UI_TICK_MS = 500;
  let lastUiTick = 0;
  let lastHeavyRenderMs = 0;

  let trendPoints = [];
  let trendEwma = 0;
  let lastTrendPersistMs = 0;

  async function fetchTrendPoints() {
    try {
      const r = await fetch('/demo/alarm_trend', { cache: 'no-store' });
      const j = await r.json();
      const arr = Array.isArray(j?.points) ? j.points : [];
      trendPoints = arr.map(x => Number(x)).filter(x => Number.isFinite(x));
      if (trendPoints.length > TREND_CFG.windowSize) trendPoints = trendPoints.slice(-TREND_CFG.windowSize);
      // 使用最后一点做 ewma 初值，避免刚启动突兀
      if (trendPoints.length > 0) {
        trendEwma = Number(trendPoints[trendPoints.length - 1]) || 0;
      }
      renderTrendSparkline();
    } catch (e) {}
  }

  async function persistTrendPoints() {
    try {
      // 覆盖旧数据：只保留最新窗口点位（后端也会截断）
      const windowPoints = trendPoints.slice(-TREND_CFG.visiblePoints);
      await fetch('/demo/alarm_trend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'replace', points: windowPoints }),
      });
    } catch (e) {}
  }

  async function clearTrendPoints() {
    try {
      await fetch('/demo/alarm_trend', { method: 'DELETE' });
    } catch (e) {}
    trendPoints = [];
    trendEwma = 0;
    renderTrendSparkline();
  }

  function clamp(n, a, b) {
    const x = typeof n === 'number' ? n : 0;
    return Math.max(a, Math.min(b, x));
  }

  function setSvgDash(el, ratio, totalDash = 320) {
    if (!el) return;
    const r = clamp(ratio, 0, 1);
    const filled = Math.round(totalDash * r);
    el.style.strokeDasharray = `${filled} ${Math.max(0, totalDash - filled)}`;
  }

  function renderEnvGauges(sensors) {
    if (!Array.isArray(sensors)) sensors = [];
    const temp = sensors.find(s => s?.type === 'temperature_sensor') || null;
    const hum = sensors.find(s => s?.type === 'humidity_sensor') || null;

    if (els.envTemp) els.envTemp.textContent = safeText(temp?.current_value, '--');
    if (els.envHum) els.envHum.textContent = safeText(hum?.current_value, '--');

    // 更新圆弧进度：温度 0-80℃，湿度 0-100%
    const arcs = dom.gaugeArcs;
    const tRatio = temp ? clamp((Number(temp.current_value) || 0) / 80, 0, 1) : 0;
    const hRatio = hum ? clamp((Number(hum.current_value) || 0) / 100, 0, 1) : 0;
    if (arcs && arcs.length >= 2) {
      setSvgDash(arcs[0], tRatio, 320);
      setSvgDash(arcs[1], hRatio, 320);
    }
  }

  function renderOverallKpis({ cameras, sensors }) {
    const camArr = Array.isArray(cameras) ? cameras : [];
    const senArr = Array.isArray(sensors) ? sensors : [];

    let online = 0;
    let alarms = 0;
    for (const cam of camArr) {
      const cameraId = cam?.camera_id || null;
      if (!cameraId) continue;
      if (cam?.status === 'online') online += 1;
      const st = getCameraState(cameraId);
      if (st?.alarmState === 'fire' || st?.alarmState === 'smoke') alarms += 1;
    }

    let sensorAlerts = 0;
    for (const s of senArr) {
      if (s?.status === 'alert') sensorAlerts += 1;
    }

    if (els.metricYoloLatency) els.metricYoloLatency.textContent = camArr.length ? `${online}/${camArr.length}` : '--';
    if (els.metricFps) els.metricFps.textContent = String(alarms);
    if (els.metricLstm) els.metricLstm.textContent = String(sensorAlerts);
  }

  function updateTrendPoints(cameras) {
    const arr = Array.isArray(cameras) ? cameras : [];
    let fire = 0;
    let smoke = 0;
    for (const cam of arr) {
      const cameraId = cam?.camera_id || null;
      if (!cameraId) continue;
      const st = getCameraState(cameraId);
      const level = st?.alarmState;
      if (level === 'fire') fire += 1;
      else if (level === 'smoke') smoke += 1;
    }

    const severityRaw = Math.min(TREND_CFG.yMax, fire * 2 + smoke);
    trendEwma = TREND_CFG.ewmaAlpha * severityRaw + (1 - TREND_CFG.ewmaAlpha) * trendEwma;
    trendPoints.push(trendEwma);
    if (trendPoints.length > TREND_CFG.windowSize) trendPoints = trendPoints.slice(-TREND_CFG.windowSize);

    const now = Date.now();
    if ((now - lastTrendPersistMs) >= 5000) {
      lastTrendPersistMs = now;
      persistTrendPoints();
    }
  }

  function renderTrendSparkline() {
    const line = dom.sparkLine;
    const fill = dom.sparkFill;
    const hint = dom.sparkHint;
    if (!line || !fill || !hint) return;

    const dotsG = dom.sparkSvg ? dom.sparkSvg.querySelector('.dt-sparkline__dots') : null;

    const series = trendPoints.slice(-TREND_CFG.visiblePoints);
    if (!series || series.length < 2) {
      if (hint) hint.textContent = '暂无数据';
      if (dotsG) dotsG.innerHTML = '';
      return;
    }

    if (hint) hint.textContent = '';

    const x0 = 20;
    const x1 = 300;
    const yTop = 20;
    const yBot = 110;
    const spanX = x1 - x0;
    const spanY = yBot - yTop;
    const n = series.length;

    const pts = series.map((v, i) => {
      const x = x0 + (n === 1 ? 0 : (i * spanX) / (n - 1));
      const norm = clamp(v / TREND_CFG.yMax, 0, 1);
      const y = yBot - norm * spanY;
      return { x, y };
    });

    const dLine = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
    if (line) line.setAttribute('d', dLine);

    const dFill = `${dLine} L${pts[pts.length - 1].x.toFixed(1)} 120 L${pts[0].x.toFixed(1)} 120 Z`;
    if (fill) fill.setAttribute('d', dFill);

    if (dotsG) {
      let html = '';
      for (const p of pts) {
        html += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="1.6" fill="rgba(var(--color-accent-rgb), 0.55)" />`;
      }
      dotsG.innerHTML = html;
    }
  }

  function renderAreaSafetyDonut(cameras) {
    const valueEl = dom.donutValue;
    const labelEl = dom.donutLabel;
    const arc = dom.donutArc;
    if (!valueEl || !labelEl || !arc) return;
    const arr = Array.isArray(cameras) ? cameras : [];

    if (!arr || arr.length === 0) {
      if (valueEl) valueEl.textContent = '--';
      if (labelEl) labelEl.textContent = '暂无数据';
      setSvgDash(arc, 0, 320);
      return;
    }

    let safe = 0;
    for (const cam of arr) {
      const cameraId = cam?.camera_id || null;
      if (!cameraId) continue;
      const st = getCameraState(cameraId);
      if (st?.alarmState === 'normal') safe += 1;
    }
    const ratio = safe / Math.max(1, arr.length);
    if (valueEl) valueEl.textContent = `${Math.round(ratio * 100)}%`;
    if (labelEl) labelEl.textContent = '安全区域占比';
    setSvgDash(arc, ratio, 320);
  }

  // 获取或初始化摄像头状态
  function getCameraState(cameraId) {
    if (!cameraId) return null;
    if (!cameraStates[cameraId]) {
      cameraStates[cameraId] = {
        alarmState: 'normal',
        alarmSince: 0,
        hist: [],
        ewmaFire: 0,
        ewmaSmoke: 0,
        lastDecisionReason: '-',
        lastDetectionSig: '',
        lastUiAlarm: null, // 记录上次的告警级别，用于日志去重
      };
    }
    return cameraStates[cameraId];
  }

  const DEMO_ALARM_POINT_ID = 'CAM-03';

  const ROOM_IDS = {
    east: 'room-east',
    hall: 'room-hall',
    west: 'room-west',
    kitchen: 'room-kitchen',
    store: 'room-store',
  };

  const CAMERA_TO_ROOM = {
    'CAM-01': ROOM_IDS.east,
    'CAM-02': ROOM_IDS.hall,
    'CAM-03': ROOM_IDS.store,
  };

  function setRoomAlarmByCameraPoint(pointId, level) {
    const normLevel = (level === 'fire' || level === 'smoke') ? level : 'normal';
    const roomId = CAMERA_TO_ROOM[pointId];
    if (!roomId) return;
    const el = document.getElementById(roomId);
    if (!el) return;
    if (normLevel === 'normal') el.removeAttribute('data-alarm');
    else el.setAttribute('data-alarm', normLevel);
  }

  function hasCameraPoint(cameraId) {
    if (!ui.camLayer || !cameraId) return false;
    const sel = `.dt-cam-point[data-camera-id="${cameraId}"]`;
    return Boolean(ui.camLayer.querySelector(sel));
  }

  function pickFallbackCameraPointId() {
    if (!ui.camLayer) return null;
    if (hasCameraPoint(DEMO_ALARM_POINT_ID)) return DEMO_ALARM_POINT_ID;
    const first = ui.camLayer.querySelector('.dt-cam-point');
    return first ? (first.getAttribute('data-camera-id') || null) : null;
  }

  function resolvePointIdFromCameraId(cameraId) {
    const raw = safeText(cameraId, '').trim();
    if (!raw) return null;
    if (hasCameraPoint(raw)) return raw;

    // 兼容后端使用 demo_cam_001 / cam03 / camera-2 等格式：抽取末尾数字并映射到 CAM-XX
    const m = raw.match(/(\d{1,3})\s*$/);
    if (m) {
      const n = Number(m[1]);
      if (Number.isFinite(n) && n > 0) {
        const id = `CAM-${String(n).padStart(2, '0')}`;
        if (hasCameraPoint(id)) return id;
      }
    }

    return null;
  }

  function updateCameraPointsOnAlarm(cameraId, level) {
    if (!ui.camLayer) return;

    const normLevel = (level === 'fire' || level === 'smoke') ? level : 'normal';

    // 直接更新指定摄像头的状态，不清空其他摄像头
    if (!cameraId) return;

    const target = resolvePointIdFromCameraId(cameraId) || pickFallbackCameraPointId();
    if (target) {
      setCameraPointState(target, normLevel);
      setRoomAlarmByCameraPoint(target, normLevel);
    }
  }

  const ALARM_LOG_CFG = {
    maxItems: 20,
    visibleMax: 6,
  };

  let alarmLogCache = [];

  async function fetchAlarmLogs() {
    try {
      const r = await fetch('/demo/alarm_logs', { cache: 'no-store' });
      const j = await r.json();
      const arr = Array.isArray(j?.items) ? j.items : [];
      alarmLogCache = arr.slice(0, ALARM_LOG_CFG.maxItems);
      renderAlarmList();
      renderTodayAlarmSummary();
    } catch (e) {}
  }

  function localDateISO() {
    const d = new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${dd}`;
  }

  function renderTodayAlarmSummary() {
    if (!elsExtra.metricAlarmFire && !elsExtra.metricAlarmSmoke && !elsExtra.metricAlarmTotal) return;
    const today = localDateISO();
    let fire = 0;
    let smoke = 0;
    for (const it of (alarmLogCache || [])) {
      const day = typeof it?.date === 'string' ? it.date : today;
      if (day !== today) continue;
      if (it?.level === 'fire') fire += 1;
      else if (it?.level === 'smoke') smoke += 1;
    }
    const total = fire + smoke;
    if (elsExtra.metricAlarmFire) elsExtra.metricAlarmFire.textContent = String(fire);
    if (elsExtra.metricAlarmSmoke) elsExtra.metricAlarmSmoke.textContent = String(smoke);
    if (elsExtra.metricAlarmTotal) elsExtra.metricAlarmTotal.textContent = String(total);
  }

  async function postAlarmLog({ ts, cameraId, level }) {
    try {
      const date = localDateISO();
      await fetch('/demo/alarm_logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ts, date, cameraId, level }),
      });
    } catch (e) {}
  }

  async function clearAlarmLogs() {
    try {
      await fetch('/demo/alarm_logs', { method: 'DELETE' });
    } catch (e) {}
    alarmLogCache = [];
    renderAlarmList();
    renderTodayAlarmSummary();
  }

  function setBadge(level) {
    // 徽章已删除，此函数保留用于兼容性
  }

  function openVideoModal(cameraId) {
    setActiveCamera(cameraId || selectedCameraId);
    if (!ui.videoModal) return;
    ui.videoModal.classList.remove('hidden');
    ui.videoModal.setAttribute('aria-hidden', 'false');
    if (ui.videoModalTitle) {
      ui.videoModalTitle.textContent = selectedCameraId ? `${selectedCameraId} 实时监控画面` : '实时监控画面';
    }

    updateSiteIntro();

    // 弹窗打开时才拉流，减少后台无意义刷新
    refreshStream({ immediate: true });

    // 弹窗打开后延迟一次绘制（等 img/canvas 布局稳定）
    setTimeout(() => {
      if (!ui.videoModal || ui.videoModal.classList.contains('hidden')) return;
      if (lastDetection) drawBoxes(lastDetection);
      else drawBoxes(null);
    }, 80);
  }

  function closeVideoModal() {
    if (!ui.videoModal) return;
    ui.videoModal.classList.add('hidden');
    ui.videoModal.setAttribute('aria-hidden', 'true');

    // 弹窗关闭后停止画面更新，减少卡顿
    if (els.videoImg) {
      try { els.videoImg.src = ''; } catch (e) {}
    }
  }

  function bindUiEvents() {
    if (ui.videoModalClose) {
      ui.videoModalClose.addEventListener('click', closeVideoModal);
    }

    if (ui.videoModal) {
      ui.videoModal.addEventListener('click', (e) => {
        if (e.target === ui.videoModal) closeVideoModal();
      });
    }

    if (ui.camLayer) {
      ui.camLayer.addEventListener('click', (e) => {
        const btn = e.target?.closest?.('.dt-cam-point');
        if (!btn) return;
        const cameraId = btn.getAttribute('data-camera-id') || null;
        // 点位互不关联：点击即切换当前通道
        setActiveCamera(cameraId);
        openVideoModal(cameraId);
      });
    }

    // 悬浮窗拖拽：仅改变 UI 容器的位置，不触碰视频流逻辑
    if (ui.videoModalDragHandle && ui.videoModalContent) {
      let dragging = false;
      let startX = 0;
      let startY = 0;
      let startLeft = 0;
      let startTop = 0;

      const onMove = (ev) => {
        if (!dragging) return;
        const dx = ev.clientX - startX;
        const dy = ev.clientY - startY;

        const nextLeft = startLeft + dx;
        const nextTop = startTop + dy;

        const rect = ui.videoModalContent.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        const minLeft = 8;
        const minTop = 8;
        const maxLeft = Math.max(minLeft, window.innerWidth - w - 8);
        const maxTop = Math.max(minTop, window.innerHeight - h - 8);

        const clampedLeft = Math.min(maxLeft, Math.max(minLeft, nextLeft));
        const clampedTop = Math.min(maxTop, Math.max(minTop, nextTop));

        ui.videoModalContent.style.left = `${clampedLeft}px`;
        ui.videoModalContent.style.top = `${clampedTop}px`;
      };

      const stopDrag = () => {
        dragging = false;
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', stopDrag);
      };

      ui.videoModalDragHandle.addEventListener('mousedown', (ev) => {
        // 避免点击关闭按钮触发拖拽
        if (ev.target?.closest?.('#videoModalClose')) return;
        ev.preventDefault();

        const rect = ui.videoModalContent.getBoundingClientRect();
        dragging = true;
        startX = ev.clientX;
        startY = ev.clientY;
        startLeft = rect.left;
        startTop = rect.top;

        // 一旦用户拖拽，切换到绝对像素定位并移除 transform
        ui.videoModalContent.style.transform = 'none';
        ui.videoModalContent.style.left = `${startLeft}px`;
        ui.videoModalContent.style.top = `${startTop}px`;

        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', stopDrag);
      });
    }
  }

  function setCameraPointState(cameraId, level) {
    if (!ui.camLayer) return;
    const normLevel = (level === 'fire' || level === 'smoke') ? level : 'normal';

    const apply = (el) => {
      if (!el) return;
      if (normLevel === 'normal') el.removeAttribute('data-state');
      else el.setAttribute('data-state', normLevel);
    };

    if (!cameraId || cameraId === '*') {
      const els = ui.camLayer.querySelectorAll('.dt-cam-point');
      for (const el of els) apply(el);

      // 同步清空房间告警态
      for (const k of Object.keys(ROOM_IDS)) {
        const roomEl = document.getElementById(ROOM_IDS[k]);
        if (roomEl) roomEl.removeAttribute('data-alarm');
      }
      return;
    }

    const sel = `.dt-cam-point[data-camera-id="${cameraId}"]`;
    apply(ui.camLayer.querySelector(sel));
  }

  function loadAlarmLogs() {
    alarmLogCache = [];
    fetchAlarmLogs();
  }

  function renderAlarmList() {
    if (!ui.alarmList) return;
    ui.alarmList.innerHTML = '';

    if (!alarmLogCache || alarmLogCache.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = '暂无告警';
      ui.alarmList.appendChild(empty);
      return;
    }

    for (const it of alarmLogCache.slice(0, ALARM_LOG_CFG.visibleMax)) {
      const row = document.createElement('div');
      row.className = 'dt-alarm-row';
      row.setAttribute('data-level', it.level || 'placeholder');
      row.innerHTML = `
        <div class="dt-alarm-row__time">${safeText(it.ts, '--:--:--')}</div>
        <div class="dt-alarm-row__cam" title="${safeText(it.cameraId, '-')}">${safeText(it.cameraId, '-')}</div>
        <div class="dt-alarm-row__level">${it.level === 'fire' ? '火焰' : (it.level === 'smoke' ? '烟雾' : '-')}</div>
      `;
      ui.alarmList.appendChild(row);
    }
  }

  function ensureAlarmListEmptyCleared() {
    if (!ui.alarmList) return;
    const empty = ui.alarmList.querySelector('.empty-state');
    if (empty) empty.remove();
  }

  function appendAlarmLog({ ts, cameraId, level }) {
    if (!ui.alarmList) return;
    if (level !== 'fire' && level !== 'smoke') return;

    const date = localDateISO();
    alarmLogCache.unshift({ date, ts: safeText(ts, '--:--:--'), cameraId: safeText(cameraId, '-'), level });
    if (alarmLogCache.length > ALARM_LOG_CFG.maxItems) alarmLogCache = alarmLogCache.slice(0, ALARM_LOG_CFG.maxItems);
    renderAlarmList();
    renderTodayAlarmSummary();

    // 异步落盘到本地 JSON（便于清空/测试）
    postAlarmLog({ ts, cameraId, level });
  }

  function alarmName(level) {
    if (level === 'fire') return '火焰告警';
    if (level === 'smoke') return '烟雾告警';
    return '正常';
  }

  function fmtDuration(ms) {
    const sec = Math.max(0, Math.floor(ms / 1000));
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    if (m <= 0) return `${s}s`;
    return `${m}m ${s}s`;
  }

  function extinguishMethod(level) {
    if (level === 'fire') return '高压细雾喷雾';
    if (level === 'smoke') return '气溶胶';
    return '';
  }

  function updateSiteIntro() {
    if (!els.siteIntroText) return;
    const cam = safeText(selectedCameraId, '--');
    const level = safeText(currentAlarmLevel, 'normal');
    const status = alarmName(level);
    const since = alarmSince ? fmtDuration(Date.now() - alarmSince) : '-';
    const temp = els.envTemp ? safeText(els.envTemp.textContent, '--') : '--';
    const hum = els.envHum ? safeText(els.envHum.textContent, '--') : '--';

    const nowText = new Date().toLocaleString();

    let suggestion = '建议：继续观察，保持通风与通道畅通。';
    if (level === 'fire') suggestion = '建议：立即核查现场火源，启动应急预案并联动处置。';
    else if (level === 'smoke') suggestion = '建议：优先排查烟雾来源（厨房/电气/扬尘），必要时进行现场确认。';

    els.siteIntroText.textContent =
      `时间：${nowText}\n` +
      `摄像头：${cam}\n` +
      `状态：${status}${alarmSince ? `（已持续 ${since}）` : ''}\n` +
      `环境：温度 ${temp}℃｜湿度 ${hum}%\n` +
      `${suggestion}`;
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

  function resetAlarm(cameraState = null) {
    const state = cameraState || { alarmState, alarmSince, hist, ewmaFire, ewmaSmoke };
    state.alarmState = 'normal';
    state.alarmSince = 0;
    state.hist = [];
    state.ewmaFire = 0;
    state.ewmaSmoke = 0;
    if (!cameraState) {
      alarmState = state.alarmState;
      alarmSince = state.alarmSince;
      hist = state.hist;
      ewmaFire = state.ewmaFire;
      ewmaSmoke = state.ewmaSmoke;
    }
  }

  function updateAlarmFromDetection(det, cameraState = null) {
    // Prefer backend decision if provided (ensures UI matches server-side EXPERIMENT_PROFILE)
    if (det && (det.final_alarm !== null && det.final_alarm !== undefined)) {
      const state = cameraState || { alarmState, alarmSince, hist, ewmaFire, ewmaSmoke, lastDecisionReason };
      const now = Date.now();
      const level = String(det.final_alarm || 'normal');
      state.lastDecisionReason = (det.final_reason !== null && det.final_reason !== undefined) ? String(det.final_reason) : '-';
      state.alarmState = level;
      state.alarmSince = (level === 'normal') ? 0 : (state.alarmSince || now);

      if (!cameraState) {
        alarmState = state.alarmState;
        alarmSince = state.alarmSince;
        lastDecisionReason = state.lastDecisionReason;
      }
      return level;
    }

    if (!_profile.fusion) {
      const state = cameraState || { alarmState, alarmSince, hist, ewmaFire, ewmaSmoke, lastDecisionReason };
      const now = Date.now();
      if (!det) {
        resetAlarm(state);
        state.lastDecisionReason = '-';
        if (!cameraState) {
          lastDecisionReason = state.lastDecisionReason;
        }
        return 'normal';
      }

      let level = 'normal';
      if (_profile.source === 'yolo') {
        const yolo = Array.isArray(det?.yolo_detections) ? det.yolo_detections : [];
        const hasFire = yolo.some(x => x && x.class_name === 'fire');
        const hasSmoke = yolo.some(x => x && x.class_name === 'smoke');
        level = hasFire ? 'fire' : (hasSmoke ? 'smoke' : 'normal');
        state.lastDecisionReason = level === 'normal' ? 'YOLO: 无框' : 'YOLO: 有框';
      } else {
        const hasLstm = det != null && det.lstm_prediction !== null && det.lstm_prediction !== undefined;
        if (!hasLstm) {
          const yolo = Array.isArray(det?.yolo_detections) ? det.yolo_detections : [];
          const hasFire = yolo.some(x => x && x.class_name === 'fire');
          const hasSmoke = yolo.some(x => x && x.class_name === 'smoke');
          level = hasFire ? 'fire' : (hasSmoke ? 'smoke' : 'normal');
          state.lastDecisionReason = 'LSTM未就绪: YOLO兜底';
        } else {
          const pred = Number(det?.lstm_prediction);
          if (pred === 2) level = 'fire';
          else if (pred === 1) level = 'smoke';
          else level = 'normal';
          state.lastDecisionReason = 'LSTM: 直接输出';
        }
      }

      state.alarmState = level;
      state.alarmSince = (level === 'normal') ? 0 : (state.alarmSince || now);

      if (!cameraState) {
        alarmState = state.alarmState;
        alarmSince = state.alarmSince;
        lastDecisionReason = state.lastDecisionReason;
      }
      return level;
    }

    // 如果提供了cameraState，使用它；否则使用全局状态（当前选中摄像头）
    const state = cameraState || { alarmState, alarmSince, hist, ewmaFire, ewmaSmoke, lastDecisionReason };
    
    if (!det) {
      resetAlarm(state);
      state.lastDecisionReason = '-';
      if (!cameraState) {
        lastDecisionReason = state.lastDecisionReason;
      }
      return 'normal';
    }

    const now = Date.now();
    const pred = det?.lstm_prediction;
    const conf = typeof det?.lstm_confidence === 'number' ? det.lstm_confidence : 0;
    const y = yoloEvidence(det);

    // Scheme A: sliding window vote ratio + hysteresis
    // - vote source: LSTM if ready, otherwise YOLO
    // - fast trigger: YOLO strong fire evidence

    if (y.fireStrong > 0) {
      state.alarmState = 'fire';
      state.alarmSince = now;
      state.lastDecisionReason = `YOLO 强证据(${y.fireStrong})`;
      if (!cameraState) {
        alarmState = state.alarmState;
        alarmSince = state.alarmSince;
        lastDecisionReason = state.lastDecisionReason;
      }
      return 'fire';
    }

    const hasLstm = pred !== null && pred !== undefined;
    let voteFire = 0;
    let voteSmoke = 0;
    let src = 'YOLO';
    if (hasLstm) {
      src = 'LSTM';
      voteFire = Number(pred) === 2 ? 1 : 0;
      voteSmoke = Number(pred) === 1 ? 1 : 0;
    } else {
      voteFire = y.fire > 0 ? 1 : 0;
      voteSmoke = y.smoke > 0 ? 1 : 0;
    }

    state.hist.push({ fire: voteFire, smoke: voteSmoke, src, t: now, conf: hasLstm ? conf : null });
    if (state.hist.length > ALARM_CFG.windowSize) state.hist.shift();

    const n = state.hist.length;
    let fireVotes = 0;
    let smokeVotes = 0;
    for (const x of state.hist) {
      fireVotes += x?.fire ? 1 : 0;
      smokeVotes += x?.smoke ? 1 : 0;
    }
    const fireRatio = n ? (fireVotes / n) : 0;
    const smokeRatio = n ? (smokeVotes / n) : 0;

    // Decision with hysteresis
    let next = state.alarmState || 'normal';
    if (next === 'fire') {
      if (fireRatio <= ALARM_CFG.offFireRatio) {
        next = 'normal';
        state.lastDecisionReason = `解除fire: ratio=${fireRatio.toFixed(2)} src=${src}`;
      } else {
        state.lastDecisionReason = `保持fire: ratio=${fireRatio.toFixed(2)} src=${src}`;
      }
    } else if (next === 'smoke') {
      // smoke -> fire upgrade if fire rises
      if (fireRatio >= ALARM_CFG.onFireRatio) {
        next = 'fire';
        state.lastDecisionReason = `升级fire: ratio=${fireRatio.toFixed(2)} src=${src}`;
      } else if (smokeRatio <= ALARM_CFG.offSmokeRatio) {
        next = 'normal';
        state.lastDecisionReason = `解除smoke: ratio=${smokeRatio.toFixed(2)} src=${src}`;
      } else {
        state.lastDecisionReason = `保持smoke: ratio=${smokeRatio.toFixed(2)} src=${src}`;
      }
    } else {
      if (fireRatio >= ALARM_CFG.onFireRatio) {
        next = 'fire';
        state.lastDecisionReason = `触发fire: ratio=${fireRatio.toFixed(2)} src=${src}`;
      } else if (smokeRatio >= ALARM_CFG.onSmokeRatio) {
        next = 'smoke';
        state.lastDecisionReason = `触发smoke: ratio=${smokeRatio.toFixed(2)} src=${src}`;
      } else {
        state.lastDecisionReason = `normal: fire=${fireRatio.toFixed(2)} smoke=${smokeRatio.toFixed(2)} src=${src}`;
      }
    }

    state.alarmState = next;
    if (next === 'normal') state.alarmSince = 0;
    else state.alarmSince = state.alarmSince || now;

    if (!cameraState) {
      alarmState = state.alarmState;
      alarmSince = state.alarmSince;
      lastDecisionReason = state.lastDecisionReason;
    }
    return next;
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
    if (!els.sensorList) return;
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
      currentAlarmLevel = 'normal';
      setBadge('normal');
      if (els.finalStatus) els.finalStatus.textContent = '-';
      if (els.finalReason) els.finalReason.textContent = '-';
      updateSiteIntro();
      drawBoxes(null);
      return;
    }

    lastDetection = det;

    const alarm = pickAlarmFromDetection(det);
    currentAlarmLevel = alarm;
    setBadge(alarm);

    if (els.finalStatus) {
      els.finalStatus.textContent = alarmName(alarm);
      els.finalStatus.classList.remove('normal', 'smoke', 'fire');
      if (alarm === 'fire') els.finalStatus.classList.add('fire');
      else if (alarm === 'smoke') els.finalStatus.classList.add('smoke');
      else els.finalStatus.classList.add('normal');
    }
    if (els.finalReason) {
      const method = extinguishMethod(alarm);
      if (!showTechDetails) {
        els.finalReason.textContent = method ? `最终灭火方式：${method}` : '-';
      } else {
        els.finalReason.textContent = method
          ? `${safeText(lastDecisionReason, '-')}` + `\n最终灭火方式：${method}`
          : `${safeText(lastDecisionReason, '-')}`;
      }
    }

    updateSiteIntro();

    // 只有弹窗打开时才绘制 bbox，避免后台无意义重绘
    if (ui.videoModal && !ui.videoModal.classList.contains('hidden')) {
      drawBoxes(det);
    }
  }

  function stopEvents() {
    try { eventSource.close(); } catch (e) {}
    eventSource = null;
  }

  function refreshStream({ immediate = true } = {}) {
    if (!els.videoImg) return;

    // 仅弹窗打开时刷新流
    if (ui.videoModal?.classList?.contains('hidden')) return;

    const apply = () => {
      const cam = selectedCameraId ? `&camera_id=${encodeURIComponent(selectedCameraId)}` : '';
      const url = `/demo/stream?_t=${Date.now()}${cam}`;
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
    const cam = selectedCameraId ? `&camera_id=${encodeURIComponent(selectedCameraId)}` : '';
    const url = `/demo/events?_t=${Date.now()}${cam}`;
    eventSource = new EventSource(url);
    
    eventSource.onopen = () => {
      // SSE 恢复时，顺便刷新一下 MJPEG（避免需要手动刷新页面）
      streamRetryMs = 400;
      refreshStream({ immediate: false });
    };

    eventSource.onerror = () => {
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

      const now = Date.now();
      if ((now - lastUiTick) >= UI_TICK_MS) lastUiTick = now;

      const shouldRenderHeavy = (now - lastHeavyRenderMs) >= UI_TICK_MS;
      if (shouldRenderHeavy) lastHeavyRenderMs = now;

      // 处理多个摄像头的数据
      const cameras = Array.isArray(payload?.cameras) ? payload.cameras : [];
      
      // 处理当前选中摄像头的检测数据（用于UI显示）
      const selectedCam = cameras.find(c => {
        const camId = c?.camera_id || '';
        const pointId = resolvePointIdFromCameraId(camId);
        return pointId === selectedCameraId;
      }) || (cameras.length > 0 ? cameras[0] : null);
      
      const det = selectedCam?.last_detection;
      const sig = det ? `${safeText(det.timestamp, '')}|${safeText(det.infer_ms, '')}|${safeText(det.buffer_size, '')}|${safeText(det.lstm_prediction, '')}|${safeText(det.lstm_confidence, '')}|${Array.isArray(det.yolo_detections) ? det.yolo_detections.length : 0}` : '';
      if (sig !== lastDetectionSig) {
        lastDetectionSig = sig;
        if (shouldRenderHeavy) renderDetection(det);
      }

      if (shouldRenderHeavy) {
        renderSensors(payload?.sensors);
        renderEnvGauges(payload?.sensors);
      }

      // 为每个摄像头独立计算告警状态并更新UI
      for (const cam of cameras) {
        const cameraId = cam?.camera_id || null;
        if (!cameraId) continue;

        // 获取该摄像头的独立状态
        const camState = getCameraState(cameraId);
        const camDet = cam?.last_detection || null;

        // 为该摄像头独立计算告警级别
        const level = updateAlarmFromDetection(camDet, camState);

        // 更新该摄像头在主页面的状态显示
        updateCameraPointsOnAlarm(cameraId, level);

        // 记录告警日志：所有摄像头的告警状态变化都记录
        const pointId = resolvePointIdFromCameraId(cameraId);
        if (pointId && level !== camState.lastUiAlarm && level !== 'normal') {
          camState.lastUiAlarm = level;
          appendAlarmLog({ ts: payload?.ts, cameraId: pointId, level });
        } else if (pointId && level === 'normal' && camState.lastUiAlarm !== 'normal') {
          // 告警解除时也更新状态，但不记录日志（避免日志过多）
          camState.lastUiAlarm = level;
        }

        // 如果是当前选中的摄像头，同步状态到全局变量
        if (pointId === selectedCameraId) {
          // 同步状态到全局变量（用于UI显示）
          alarmState = camState.alarmState;
          alarmSince = camState.alarmSince;
          hist = camState.hist;
          ewmaFire = camState.ewmaFire;
          ewmaSmoke = camState.ewmaSmoke;
          lastDecisionReason = camState.lastDecisionReason;
          currentAlarmLevel = level;
          lastUiAlarm = level; // 同步当前选中摄像头的告警状态
        }
      }

      updateTrendPoints(cameras);

      if (shouldRenderHeavy) {
        renderOverallKpis({ cameras, sensors: payload?.sensors });
        renderTrendSparkline();
        renderAreaSafetyDonut(cameras);
      }
    };
  }

  function setActiveCamera(cameraId) {
    if (!cameraId) return;
    if (selectedCameraId === cameraId) return;
    selectedCameraId = cameraId;
    lastUiAlarm = null;
    if (ui.videoModalTitle && !ui.videoModal?.classList?.contains('hidden')) {
      ui.videoModalTitle.textContent = `${selectedCameraId} 实时监控画面`;
    }
    // 切换通道时，同步全局状态到当前选中摄像头的状态
    const backendCamId = resolvePointIdFromCameraId(cameraId);
    if (backendCamId) {
      // 将CAM-01转换为demo_cam_001格式
      const m = backendCamId.match(/CAM-(\d+)/);
      if (m) {
        const num = m[1];
        const camId = `demo_cam_${num.padStart(3, '0')}`;
        const camState = getCameraState(camId);
        // 同步状态到全局变量（用于UI显示）
        alarmState = camState.alarmState;
        alarmSince = camState.alarmSince;
        hist = camState.hist;
        ewmaFire = camState.ewmaFire;
        ewmaSmoke = camState.ewmaSmoke;
        lastDecisionReason = camState.lastDecisionReason;
      }
    }
    // 切换通道时重连（后端即使是单路也不会影响前端结构）
    lastDetectionSig = '';
    stopEvents();
    refreshStream({ immediate: true });
    startEvents();
  }

  function initSingleCamera() {
    // 默认选中一个点位，避免告警落到兜底点位
    if (!selectedCameraId && ui.camLayer) {
      const first = ui.camLayer.querySelector('.dt-cam-point');
      if (first) selectedCameraId = first.getAttribute('data-camera-id') || null;
    }
    loadAlarmLogs();
    renderAlarmList();
    initStreamHandlers();
    // 默认不拉流，避免未打开弹窗也占用资源
    startStreamWatchdog();
    lastDetectionSig = '';
    renderDetection(null);
    startEvents();

    fetchTrendPoints();
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
  bindUiEvents();
  initSingleCamera();

  // 方便你手动测试/清空：在控制台执行 window.clearAlarmLogs()
  window.clearAlarmLogs = clearAlarmLogs;
  window.clearTrendPoints = clearTrendPoints;

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
