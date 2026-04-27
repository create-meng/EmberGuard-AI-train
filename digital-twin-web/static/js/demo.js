(() => {
  const EXPERIMENT_PROFILE = (typeof window !== 'undefined' && window.EXPERIMENT_PROFILE)
    ? window.EXPERIMENT_PROFILE
    : 'yolo_lstm_denoise_fusion';

  const els = {
    headerTime: document.getElementById('headerTime'),
    headerOnlineCount: document.getElementById('headerOnlineCount'),
    headerAlertCount: document.getElementById('headerAlertCount'),
    headerRiskLevel: document.getElementById('headerRiskLevel'),
    envTemp: document.getElementById('envTemp'),
    envHum: document.getElementById('envHum'),
    metricYoloLatency: document.getElementById('metricYoloLatency'),
    metricFps: document.getElementById('metricFps'),
    metricLstm: document.getElementById('metricLstm'),
    metricAlarmFire: document.getElementById('metricAlarmFire'),
    metricAlarmSmoke: document.getElementById('metricAlarmSmoke'),
    metricAlarmTotal: document.getElementById('metricAlarmTotal'),
    currentViewTitle: document.getElementById('currentViewTitle'),
    currentViewHint: document.getElementById('currentViewHint'),
    detailPanelTitle: document.getElementById('detailPanelTitle'),
    btnBackToOverview: document.getElementById('btnBackToOverview'),
    overviewLayer: document.getElementById('overviewLayer'),
    overviewViewport: document.getElementById('overviewViewport'),
    mapBaseGrid: document.getElementById('mapBaseGrid'),
    overviewWorld: document.getElementById('overviewWorld'),
    detailFloatPanel: document.getElementById('detailFloatPanel'),
    detailLayer: document.getElementById('detailLayer'),
    camLayer: document.getElementById('camLayer'),
    alarmList: document.getElementById('alarmList'),
    buildingArchiveName: document.getElementById('buildingArchiveName'),
    archiveYear: document.getElementById('archiveYear'),
    archiveProtectionLevel: document.getElementById('archiveProtectionLevel'),
    archiveStructureType: document.getElementById('archiveStructureType'),
    archiveArea: document.getElementById('archiveArea'),
    archiveKeyParts: document.getElementById('archiveKeyParts'),
    aiRiskPanel: document.getElementById('aiRiskPanel'),
    aiRiskScore: document.getElementById('aiRiskScore'),
    aiRiskLevel: document.getElementById('aiRiskLevel'),
    aiRiskSource: document.getElementById('aiRiskSource'),
    aiRiskReason: document.getElementById('aiRiskReason'),
    aiRiskTrend: document.getElementById('aiRiskTrend'),
    videoModal: document.getElementById('videoModal'),
    videoModalClose: document.getElementById('videoModalClose'),
    videoModalTitle: document.getElementById('videoModalTitle'),
    videoModalContent: document.getElementById('videoModalContent'),
    videoModalDragHandle: document.getElementById('videoModalDragHandle'),
    videoImg: document.getElementById('videoImg'),
    overlay: document.getElementById('overlay'),
    finalStatus: document.getElementById('finalStatus'),
    finalReason: document.getElementById('finalReason'),
    siteIntroText: document.getElementById('siteIntroText'),
  };

  const dom = {
    gaugeArcs: document.querySelectorAll('.dt-gauge__arc'),
    sparkSvg: document.querySelector('.dt-sparkline'),
    sparkLine: null,
    sparkFill: null,
    sparkHint: null,
  };

  if (dom.sparkSvg) {
    dom.sparkLine = dom.sparkSvg.querySelector('.dt-sparkline__line');
    dom.sparkFill = dom.sparkSvg.querySelector('.dt-sparkline__fill');
    dom.sparkHint = dom.sparkSvg.querySelector('.dt-sparkline__hint');
  }

  const state = {
    currentView: 'overview',
    selectedBuildingId: null,
    selectedCameraId: null,
    overviewConfig: null,
    payload: null,
    eventSource: null,
    trendPoints: [],
    trendEwma: 0,
    lastTrendPersistMs: 0,
    alarmLogCache: [],
    cameraStates: {},
    streamRetryTimer: null,
    streamRetryMs: 400,
    streamRefreshTimer: null,
    lastDetection: null,
    lastModelKey: '',
    mapView: { scale: 1, offsetX: 0, offsetY: 0, dragging: false, startX: 0, startY: 0, baseX: 0, baseY: 0 },
  };

  const ALARM_LOG_CFG = {
    maxItems: 20,
    visibleMax: 6,
  };

  const TREND_CFG = {
    windowSize: 140,
    visiblePoints: 120,
    yMax: 6,
    ewmaAlpha: 0.22,
  };

  const STATIC_BUILDINGS = [
    {
      building_id: 'building_jingxiu',
      name: '敬修堂',
      model_url: '/static/models/qinghe-building.glb',
      anchors_url: '/static/models/qinghe-building.anchors.json',
      archive: { year: '清代', protection_level: '县级文物保护单位', structure_type: '砖木结构', area: '862.3㎡', key_parts: '屋顶、梁架、正厅明间' },
      camera_ids: ['demo_cam_001'],
      sensor_ids: []
    },
    {
      building_id: 'building_zhenxing',
      name: '振兴堂',
      model_url: '/static/models/qinghe-building.glb',
      anchors_url: '/static/models/qinghe-building.anchors.json',
      archive: { year: '民国', protection_level: '一般保护建筑', structure_type: '穿斗式木构', area: '745.6㎡', key_parts: '戏台、厢房、木柱节点' },
      camera_ids: ['demo_cam_002'],
      sensor_ids: []
    },
    {
      building_id: 'building_wenchang',
      name: '文昌阁',
      model_url: '/static/models/qinghe-building.glb',
      anchors_url: '/static/models/qinghe-building.anchors.json',
      archive: { year: '清末', protection_level: '历史建筑', structure_type: '木结构', area: '512.4㎡', key_parts: '阁楼、斗拱、檐口部位' },
      camera_ids: ['demo_cam_003'],
      sensor_ids: []
    }
  ];

  const STATIC_OVERVIEW = { canvas: { width: 1400, height: 920 } };

  function safeText(v, fallback = '-') {
    if (v === undefined || v === null || v === '') return fallback;
    return String(v);
  }

  function clamp(n, a, b) {
    const x = typeof n === 'number' ? n : 0;
    return Math.max(a, Math.min(b, x));
  }

  function normalizeCameraId(cameraId) {
    const raw = safeText(cameraId, '').trim();
    if (!raw) return null;
    if (/^demo_cam_\d{3}$/i.test(raw)) return raw.toLowerCase();
    const m = raw.match(/(\d{1,3})\s*$/);
    if (!m) return raw;
    return `demo_cam_${String(Number(m[1])).padStart(3, '0')}`;
  }

  function shortCameraId(cameraId) {
    const raw = normalizeCameraId(cameraId) || '';
    const m = raw.match(/(\d{1,3})$/);
    if (!m) return raw || '-';
    return `CAM-${String(Number(m[1])).padStart(2, '0')}`;
  }

  function getCameraState(cameraId) {
    const id = normalizeCameraId(cameraId);
    if (!id) return null;
    if (!state.cameraStates[id]) {
      state.cameraStates[id] = {
        alarmState: 'normal',
        finalReason: '-',
      };
    }
    return state.cameraStates[id];
  }

  function setSvgDash(el, ratio, totalDash = 320) {
    if (!el) return;
    const r = clamp(ratio, 0, 1);
    const filled = Math.round(totalDash * r);
    el.style.strokeDasharray = `${filled} ${Math.max(0, totalDash - filled)}`;
  }

  function localDateISO() {
    const d = new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${dd}`;
  }

  function buildingList() {
    return STATIC_BUILDINGS;
  }

  function getBuilding(buildingId) {
    return STATIC_BUILDINGS.find((b) => b.building_id === buildingId) || null;
  }

  function cameraSnapshotsMap() {
    const result = new Map();
    const cameras = Array.isArray(state.payload?.cameras) ? state.payload.cameras : [];
    for (const cam of cameras) {
      const id = normalizeCameraId(cam?.camera_id);
      if (!id) continue;
      result.set(id, cam);
    }
    return result;
  }

  function sensorSnapshots() {
    return Array.isArray(state.payload?.sensors) ? state.payload.sensors : [];
  }

  function sensorMap() {
    const map = new Map();
    for (const sensor of sensorSnapshots()) {
      if (sensor?.id) map.set(sensor.id, sensor);
    }
    return map;
  }

  function buildingSnapshot(building) {
    const cams = cameraSnapshotsMap();
    const sensors = sensorMap();
    const cameraSnapshots = (building?.camera_ids || [])
      .map((id) => cams.get(normalizeCameraId(id)))
      .filter(Boolean);
    const sensorSnapshotsList = (building?.sensor_ids || [])
      .map((id) => sensors.get(id))
      .filter(Boolean);
    return { building, cameraSnapshots, sensorSnapshots: sensorSnapshotsList };
  }

  function deriveCameraAlarmLevel(cameraSnapshot) {
    const det = cameraSnapshot?.last_detection || null;
    const level = safeText(det?.final_alarm, '').toLowerCase();
    if (level === 'fire') return 'fire';
    if (level === 'smoke') return 'smoke';
    return 'normal';
  }

  function aggregateBuildingStatus(building) {
    const snap = buildingSnapshot(building);
    let primarySource = 'Sensor';
    let primaryReason = '无异常';

    for (const cam of snap.cameraSnapshots) {
      const level = deriveCameraAlarmLevel(cam);
      const camState = getCameraState(cam.camera_id);
      if (camState) {
        camState.alarmState = level;
        camState.finalReason = safeText(cam?.last_detection?.final_reason, '-');
      }
      if (level === 'fire') {
        return {
          status: 'alarm',
          primarySource: shortCameraId(cam.camera_id),
          primaryReason: 'Fusion / ' + safeText(cam?.last_detection?.final_reason, 'YOLO 视觉识别'),
        };
      }
      if (level === 'smoke') {
        primarySource = shortCameraId(cam.camera_id);
        primaryReason = 'Fusion / ' + safeText(cam?.last_detection?.final_reason, 'LSTM 趋势判定');
      }
    }

    if (primarySource !== 'Sensor') {
      return {
        status: 'warning',
        primarySource,
        primaryReason,
      };
    }

    const sensorAlert = snap.sensorSnapshots.find((sensor) => sensor?.status === 'alert');
    if (sensorAlert) {
      return {
        status: 'attention',
        primarySource: safeText(sensorAlert.name, 'Sensor'),
        primaryReason: safeText(sensorAlert.type, 'sensor') + ' 超过阈值',
      };
    }

    return {
      status: 'normal',
      primarySource: '多源数据',
      primaryReason: 'YOLO / LSTM / Sensor 均正常',
    };
  }

  function statusLabel(status) {
    if (status === 'alarm') return '高风险';
    if (status === 'warning') return '中高风险';
    if (status === 'attention') return '中风险';
    if (status === 'handling') return '处理中';
    return '低风险';
  }

  function statusScore(status) {
    if (status === 'alarm') return 92;
    if (status === 'warning') return 72;
    if (status === 'attention') return 48;
    if (status === 'handling') return 80;
    return 18;
  }

  function markerTone(status) {
    if (status === 'alarm') return 'alarm';
    if (status === 'warning') return 'warning';
    if (status === 'attention') return 'attention';
    return 'normal';
  }

  function updateMapViewport(reset = false) {
    if (!els.overviewViewport || !els.overviewWorld) return;
    const viewportRect = els.overviewViewport.getBoundingClientRect();
    const worldWidth = els.overviewWorld.offsetWidth || viewportRect.width;
    const worldHeight = els.overviewWorld.offsetHeight || viewportRect.height;

    if (reset || !state.mapView.scale) {
      state.mapView.scale = 1;
      state.mapView.offsetX = 0;
      state.mapView.offsetY = 0;
    }

    const scaledWidth = worldWidth * state.mapView.scale;
    const scaledHeight = worldHeight * state.mapView.scale;
    const limitX = Math.max(0, (scaledWidth - viewportRect.width) / 2);
    const limitY = Math.max(0, (scaledHeight - viewportRect.height) / 2);

    state.mapView.offsetX = Math.max(-limitX, Math.min(limitX, state.mapView.offsetX));
    state.mapView.offsetY = Math.max(-limitY, Math.min(limitY, state.mapView.offsetY));

    els.overviewWorld.style.transform = `translate(${state.mapView.offsetX}px, ${state.mapView.offsetY}px) scale(${state.mapView.scale})`;
  }

  function bindMapViewportDrag() {
    if (!els.overviewViewport || !els.overviewWorld) return;
    const onMove = (ev) => {
      if (!state.mapView.dragging) return;
      const dx = ev.clientX - state.mapView.startX;
      const dy = ev.clientY - state.mapView.startY;
      state.mapView.offsetX = state.mapView.baseX + dx;
      state.mapView.offsetY = state.mapView.baseY + dy;
      updateMapViewport(false);
    };

    const stopDrag = () => {
      state.mapView.dragging = false;
      els.overviewViewport.classList.remove('is-dragging');
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', stopDrag);
    };

    els.overviewViewport.addEventListener('mousedown', (ev) => {
      if (ev.target.closest('.simple-building')) return;
      ev.preventDefault();
      state.mapView.dragging = true;
      state.mapView.startX = ev.clientX;
      state.mapView.startY = ev.clientY;
      state.mapView.baseX = state.mapView.offsetX;
      state.mapView.baseY = state.mapView.offsetY;
      els.overviewViewport.classList.add('is-dragging');
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', stopDrag);
    });

    els.overviewViewport.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const nextScale = ev.deltaY < 0 ? state.mapView.scale * 1.12 : state.mapView.scale / 1.12;
      state.mapView.scale = Math.max(1, Math.min(2.8, nextScale));
      updateMapViewport(false);
    }, { passive: false });
  }

  function renderOverview() {
    return;
    if (!els.overviewLayer || !state.overviewConfig) return;
    const cfg = state.overviewConfig;
    els.overviewLayer.innerHTML = '';
    if (els.overviewWorld) {
      els.overviewWorld.style.setProperty('--overview-width', `${cfg.canvas?.width || 1400}px`);
      els.overviewWorld.style.setProperty('--overview-height', `${cfg.canvas?.height || 920}px`);
    }

    for (const road of (cfg.roads || [])) {
      const el = document.createElement('div');
      el.className = 'dt-overview-road';
      el.style.left = `${road.x}px`;
      el.style.top = `${road.y}px`;
      el.style.width = `${road.width}px`;
      el.style.height = `${road.height}px`;
      el.style.transform = `rotate(${Number(road.rotation) || 0}deg)`;
      el.innerHTML = `<span>${safeText(road.name, '消防通道')}</span>`;
      els.overviewLayer.appendChild(el);
    }

    for (const source of (cfg.water_sources || [])) {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'dt-overview-water';
      el.style.left = `${source.x}px`;
      el.style.top = `${source.y}px`;
      el.innerHTML = `<span class="dt-overview-water__dot"></span><span>${safeText(source.name, '水源点')}</span>`;
      els.overviewLayer.appendChild(el);
    }

    for (const area of (cfg.key_areas || [])) {
      const el = document.createElement('div');
      el.className = 'dt-overview-keyarea';
      el.style.left = `${area.x}px`;
      el.style.top = `${area.y}px`;
      el.style.width = `${area.width}px`;
      el.style.height = `${area.height}px`;
      el.innerHTML = `<span>${safeText(area.name, '重点保护区域')}</span>`;
      els.overviewLayer.appendChild(el);
    }

    for (const building of buildingList()) {
      const aggregate = aggregateBuildingStatus(building);
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'dt-overview-building';
      btn.dataset.buildingId = building.building_id;
      btn.dataset.status = aggregate.status;
      if (building.building_id === state.selectedBuildingId) btn.classList.add('is-selected');
      btn.style.left = `${building.x}px`;
      btn.style.top = `${building.y}px`;
      btn.style.width = `${building.width}px`;
      btn.style.height = `${building.height}px`;
      btn.innerHTML = `
        <span class="dt-overview-building__name">${safeText(building.name, '未命名建筑')}</span>
        <span class="dt-overview-building__status">${statusLabel(aggregate.status)}</span>
        <span class="dt-overview-building__meta">${safeText(building.archive?.protection_level, '-')}</span>
      `;
      els.overviewLayer.appendChild(btn);
    }
  }

  function syncSimpleBuildingStates() {
    document.querySelectorAll('.simple-building[data-building-id]').forEach((btn) => {
      const buildingId = btn.getAttribute('data-building-id');
      const building = getBuilding(buildingId);
      const status = aggregateBuildingStatus(building).status;
      btn.dataset.status = status;
      btn.dataset.tone = markerTone(status);
    });
  }

  function renderBuildingArchive(building) {
    const archive = building?.archive || {};
    if (els.buildingArchiveName) els.buildingArchiveName.textContent = safeText(building?.name, '--');
    const archiveBuildingNameMirror = document.getElementById('archiveBuildingNameMirror');
    if (archiveBuildingNameMirror) archiveBuildingNameMirror.textContent = safeText(building?.name, '--');
    if (els.archiveYear) els.archiveYear.textContent = safeText(archive.year, '--');
    if (els.archiveProtectionLevel) els.archiveProtectionLevel.textContent = safeText(archive.protection_level, '--');
    if (els.archiveStructureType) els.archiveStructureType.textContent = safeText(archive.structure_type, '--');
    if (els.archiveArea) els.archiveArea.textContent = safeText(archive.area, '--');
    if (els.archiveKeyParts) els.archiveKeyParts.textContent = safeText(archive.key_parts, '--');
  }

  function renderCamLayer(building) {
    if (!els.camLayer) return;
    const cameraIds = Array.isArray(building?.camera_ids) ? building.camera_ids : [];
    const cameraAnchorMap = building?.camera_anchor_map && typeof building.camera_anchor_map === 'object'
      ? building.camera_anchor_map
      : {};
    els.camLayer.innerHTML = '';

    cameraIds.forEach((cameraId, index) => {
      const normalizedCameraId = normalizeCameraId(cameraId);
      const shortId = shortCameraId(cameraId);
      const snap = cameraSnapshotsMap().get(normalizedCameraId);
      const level = deriveCameraAlarmLevel(snap);
      const anchorKey = safeText(cameraAnchorMap[normalizedCameraId], shortId);
      const btn = document.createElement('button');
      btn.className = 'dt-cam-point';
      btn.type = 'button';
      btn.setAttribute('data-camera-id', normalizedCameraId);
      btn.setAttribute('data-short-camera-id', shortId);
      btn.setAttribute('data-anchor-key', anchorKey);
      if (level === 'fire' || level === 'smoke') btn.setAttribute('data-state', level);
      btn.title = `${shortId} 监控点位`;
      btn.style.left = `${36 + index * 14}%`;
      btn.style.top = `${42 + index * 7}%`;
      btn.innerHTML = `
        <span class="dt-cam-dot"></span>
        <span class="dt-cam-label">${shortId}</span>
      `;
      els.camLayer.appendChild(btn);
    });
  }

  async function loadBuildingModel(building) {
    if (!building) return;
    const modelKey = `${building.model_url || ''}|${building.anchors_url || ''}`;
    if (state.lastModelKey === modelKey) return;
    if (els.detailLayer) els.detailLayer.classList.remove('is-model-ready');
    if (window.TwinModelViewer && typeof window.TwinModelViewer.load === 'function') {
      await window.TwinModelViewer.load({
        modelUrl: building.model_url,
        anchorsUrl: building.anchors_url || '',
      });
      if (els.detailLayer) els.detailLayer.classList.add('is-model-ready');
      state.lastModelKey = modelKey;
    }
  }

  function renderDetailView(building) {
    if (!building) return;
    renderBuildingArchive(building);
    renderCamLayer(building);
    if (state.currentView === 'building_detail') {
      loadBuildingModel(building).catch(() => {});
    }
  }

  function riskTrendText(score) {
    if (score >= 90) return '未来 5 分钟内仍处高位';
    if (score >= 70) return '短时间内有升级风险';
    if (score >= 45) return '存在轻度波动';
    return '整体保持稳定';
  }

  function renderAiRisk(building) {
    if (!building) return;
    const aggregate = aggregateBuildingStatus(building);
    const score = statusScore(aggregate.status);
    if (els.aiRiskScore) els.aiRiskScore.textContent = `${score}/100`;
    if (els.aiRiskLevel) els.aiRiskLevel.textContent = statusLabel(aggregate.status);
    if (els.aiRiskSource) els.aiRiskSource.textContent = safeText(aggregate.primarySource, '--');
    if (els.aiRiskReason) els.aiRiskReason.textContent = safeText(aggregate.primaryReason, '--');
    if (els.aiRiskTrend) els.aiRiskTrend.textContent = riskTrendText(score);
    if (els.aiRiskPanel) els.aiRiskPanel.setAttribute('data-risk-level', aggregate.status);
  }

  function renderCurrentSelection() {
    const building = getBuilding(state.selectedBuildingId) || buildingList()[0] || null;
    if (!building) return;
    if (!state.selectedBuildingId) state.selectedBuildingId = building.building_id;
    syncSimpleBuildingStates();
    renderBuildingArchive(building);
    renderAiRisk(building);
    renderDetailView(building);
  }

  function renderViewState() {
    const isOverview = state.currentView === 'overview';
    const detailFloatPanel = document.getElementById('detailFloatPanel');
    if (els.overviewLayer) els.overviewLayer.classList.remove('hidden');
    if (els.detailLayer) els.detailLayer.classList.toggle('hidden', isOverview);
    if (detailFloatPanel) detailFloatPanel.classList.toggle('hidden', isOverview);
    if (els.btnBackToOverview) els.btnBackToOverview.classList.toggle('hidden', isOverview);
    if (document.body) document.body.setAttribute('data-platform-view', state.currentView);

    const building = getBuilding(state.selectedBuildingId) || buildingList()[0] || null;
    if (els.detailPanelTitle) {
      els.detailPanelTitle.textContent = safeText(building?.name, '建筑');
    }
    if (els.currentViewTitle) {
      els.currentViewTitle.textContent = '古村落总览（GIS数字孪生）';
    }
    if (els.currentViewHint) {
      els.currentViewHint.textContent = isOverview
        ? '点击建筑块后，在右上角显示 BIM 数字孪生悬浮窗。'
        : `${safeText(building?.name, '当前建筑')} 的 BIM 数字孪生悬浮窗已打开，可点击监控点位查看实时画面。`;
    }

    document.querySelectorAll('.dt-nav-btn').forEach((btn) => {
      const view = btn.getAttribute('data-nav-view') || 'overview';
      const active = view === state.currentView || (view === 'building_detail' && state.currentView === 'building_detail');
      btn.classList.toggle('is-active', active);
    });
  }

  function setCurrentView(nextView) {
    if (nextView === 'building_detail' && !state.selectedBuildingId) {
      const first = buildingList()[0];
      if (first) state.selectedBuildingId = first.building_id;
    }
    state.currentView = nextView === 'building_detail' ? 'building_detail' : 'overview';
    renderViewState();
    renderCurrentSelection();
  }

  function selectBuilding(buildingId, nextView = 'building_detail') {
    state.selectedBuildingId = buildingId;
    state.lastModelKey = '';
    setCurrentView(nextView);
  }

  function renderEnvGauges(sensors) {
    const temp = (sensors || []).find((s) => s?.type === 'temperature_sensor') || null;
    const hum = (sensors || []).find((s) => s?.type === 'humidity_sensor') || null;
    if (els.envTemp) els.envTemp.textContent = safeText(temp?.current_value, '--');
    if (els.envHum) els.envHum.textContent = safeText(hum?.current_value, '--');
    const arcs = dom.gaugeArcs;
    const tRatio = temp ? clamp((Number(temp.current_value) || 0) / 80, 0, 1) : 0;
    const hRatio = hum ? clamp((Number(hum.current_value) || 0) / 100, 0, 1) : 0;
    if (arcs && arcs.length >= 2) {
      setSvgDash(arcs[0], tRatio, 320);
      setSvgDash(arcs[1], hRatio, 320);
    }
  }

  function renderOverallKpis() {
    const cameras = Array.isArray(state.payload?.cameras) ? state.payload.cameras : [];
    const sensors = sensorSnapshots();
    const online = cameras.filter((cam) => cam?.status === 'online').length;
    const currentAlerts = buildingList().filter((building) => {
      const status = aggregateBuildingStatus(building).status;
      return status === 'warning' || status === 'alarm';
    }).length;
    const sensorAlerts = sensors.filter((sensor) => sensor?.status === 'alert').length;
    const highestStatus = buildingList().reduce((best, building) => {
      const status = aggregateBuildingStatus(building).status;
      const order = { normal: 1, attention: 2, warning: 3, alarm: 4, handling: 5 };
      return order[status] > order[best] ? status : best;
    }, 'normal');

    if (els.metricYoloLatency) els.metricYoloLatency.textContent = `${online}/${cameras.length || 0}`;
    if (els.metricFps) els.metricFps.textContent = String(currentAlerts);
    if (els.metricLstm) els.metricLstm.textContent = String(sensorAlerts);
    if (els.headerOnlineCount) els.headerOnlineCount.textContent = String(online + sensors.length);
    if (els.headerAlertCount) els.headerAlertCount.textContent = String(currentAlerts);
    if (els.headerRiskLevel) els.headerRiskLevel.textContent = statusLabel(highestStatus);
  }

  async function fetchAlarmLogs() {
    try {
      const r = await fetch('/demo/alarm_logs', { cache: 'no-store' });
      const j = await r.json();
      state.alarmLogCache = Array.isArray(j?.items) ? j.items.slice(0, ALARM_LOG_CFG.maxItems) : [];
      renderAlarmList();
      renderTodayAlarmSummary();
    } catch (e) {}
  }

  function renderAlarmList() {
    if (!els.alarmList) return;
    els.alarmList.innerHTML = '';
    if (!state.alarmLogCache.length) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = '暂无告警';
      els.alarmList.appendChild(empty);
      return;
    }

    for (const item of state.alarmLogCache.slice(0, ALARM_LOG_CFG.visibleMax)) {
      const row = document.createElement('div');
      row.className = 'dt-alarm-row';
      row.setAttribute('data-level', safeText(item.level, 'placeholder'));
      row.innerHTML = `
        <div class="dt-alarm-row__time">${safeText(item.ts, '--:--:--')}</div>
        <div class="dt-alarm-row__cam" title="${safeText(item.cameraId, '-')}">${safeText(item.cameraId, '-')}</div>
        <div class="dt-alarm-row__level">${item.level === 'fire' ? '??' : (item.level === 'smoke' ? '??' : '-')}</div>
      `;
      els.alarmList.appendChild(row);
    }
  }

  function renderTodayAlarmSummary() {
    const today = localDateISO();
    let fire = 0;
    let smoke = 0;
    for (const item of state.alarmLogCache) {
      const day = typeof item?.date === 'string' ? item.date : today;
      if (day !== today) continue;
      if (item?.level === 'fire') fire += 1;
      else if (item?.level === 'smoke') smoke += 1;
    }
    if (els.metricAlarmFire) els.metricAlarmFire.textContent = String(fire);
    if (els.metricAlarmSmoke) els.metricAlarmSmoke.textContent = String(smoke);
    if (els.metricAlarmTotal) els.metricAlarmTotal.textContent = String(fire + smoke);
  }

  async function postAlarmLog({ ts, cameraId, level }) {
    try {
      await fetch('/demo/alarm_logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ts, date: localDateISO(), cameraId, level }),
      });
    } catch (e) {}
  }

  function maybeAppendAlarmLog(payload, cameras) {
    const ts = safeText(payload?.ts, '--:--:--');
    for (const cam of cameras) {
      const cameraId = normalizeCameraId(cam?.camera_id);
      if (!cameraId) continue;
      const level = deriveCameraAlarmLevel(cam);
      const camState = getCameraState(cameraId);
      if (!camState) continue;
      if (level !== camState.alarmState && (level === 'fire' || level === 'smoke')) {
        camState.alarmState = level;
        state.alarmLogCache.unshift({ date: localDateISO(), ts, cameraId: shortCameraId(cameraId), level });
        state.alarmLogCache = state.alarmLogCache.slice(0, ALARM_LOG_CFG.maxItems);
        renderAlarmList();
        renderTodayAlarmSummary();
        postAlarmLog({ ts, cameraId: shortCameraId(cameraId), level });
      } else {
        camState.alarmState = level;
      }
      camState.finalReason = safeText(cam?.last_detection?.final_reason, '-');
    }
  }

  async function fetchTrendPoints() {
    try {
      const r = await fetch('/demo/alarm_trend', { cache: 'no-store' });
      const j = await r.json();
      const arr = Array.isArray(j?.points) ? j.points : [];
      state.trendPoints = arr.map((x) => Number(x)).filter((x) => Number.isFinite(x));
      if (state.trendPoints.length > TREND_CFG.windowSize) state.trendPoints = state.trendPoints.slice(-TREND_CFG.windowSize);
      if (state.trendPoints.length > 0) {
        state.trendEwma = Number(state.trendPoints[state.trendPoints.length - 1]) || 0;
      }
      renderTrendSparkline();
    } catch (e) {}
  }

  async function persistTrendPoints() {
    try {
      const windowPoints = state.trendPoints.slice(-TREND_CFG.visiblePoints);
      await fetch('/demo/alarm_trend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'replace', points: windowPoints }),
      });
    } catch (e) {}
  }

  function updateTrendPoints() {
    let fire = 0;
    let warning = 0;
    for (const building of buildingList()) {
      const status = aggregateBuildingStatus(building).status;
      if (status === 'alarm') fire += 1;
      else if (status === 'warning') warning += 1;
    }
    const severityRaw = Math.min(TREND_CFG.yMax, fire * 2 + warning);
    state.trendEwma = TREND_CFG.ewmaAlpha * severityRaw + (1 - TREND_CFG.ewmaAlpha) * state.trendEwma;
    state.trendPoints.push(state.trendEwma);
    if (state.trendPoints.length > TREND_CFG.windowSize) state.trendPoints = state.trendPoints.slice(-TREND_CFG.windowSize);
    const now = Date.now();
    if ((now - state.lastTrendPersistMs) >= 5000) {
      state.lastTrendPersistMs = now;
      persistTrendPoints();
    }
  }

  function renderTrendSparkline() {
    const line = dom.sparkLine;
    const fill = dom.sparkFill;
    const hint = dom.sparkHint;
    if (!line || !fill || !hint) return;

    const dotsG = dom.sparkSvg ? dom.sparkSvg.querySelector('.dt-sparkline__dots') : null;
    const series = state.trendPoints.slice(-TREND_CFG.visiblePoints);
    if (series.length < 2) {
      hint.textContent = '暂无数据';
      if (dotsG) dotsG.innerHTML = '';
      return;
    }
    hint.textContent = '';

    const x0 = 20;
    const x1 = 300;
    const yTop = 20;
    const yBot = 110;
    const spanX = x1 - x0;
    const spanY = yBot - yTop;

    const points = series.map((value, idx) => {
      const x = x0 + (spanX * idx) / Math.max(1, series.length - 1);
      const y = yBot - (clamp(value, 0, TREND_CFG.yMax) / TREND_CFG.yMax) * spanY;
      return [x, y];
    });

    const linePath = points.map((p, idx) => `${idx === 0 ? 'M' : 'L'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
    const fillPath = `${linePath} L${points[points.length - 1][0].toFixed(1)} ${yBot} L${points[0][0].toFixed(1)} ${yBot} Z`;
    line.setAttribute('d', linePath);
    fill.setAttribute('d', fillPath);

    if (dotsG) {
      const html = points.slice(-4).map((p) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.2" fill="rgba(var(--color-accent-rgb),0.82)" />`).join('');
      dotsG.innerHTML = html;
    }
  }

  function renderDetection(det) {
    state.lastDetection = det || null;
    const alarmName = det?.final_alarm === 'fire'
      ? '火焰告警'
      : det?.final_alarm === 'smoke'
        ? '烟雾告警'
        : '正常';

    if (els.finalStatus) {
      els.finalStatus.textContent = alarmName;
      els.finalStatus.className = `lstm-status ${det?.final_alarm || 'normal'}`;
    }
    if (els.finalReason) {
      const source = safeText(det?.final_source, '-');
      const reason = safeText(det?.final_reason, '暂无判定');
      els.finalReason.textContent = `${source} 路 ${reason}`;
    }

    drawBoxes(det);
    updateSiteIntro();
  }

  function drawBoxes(det) {
    if (!els.overlay || !els.videoImg) return;
    const canvas = els.overlay;
    const rect = els.videoImg.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width));
    canvas.height = Math.max(1, Math.round(rect.height));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!det || !Array.isArray(det?.yolo_detections)) return;

    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 480;
    ctx.font = '12px Inter';
    ctx.lineWidth = 2;

    for (const item of det.yolo_detections) {
      if (!Array.isArray(item?.bbox) || item.bbox.length !== 4) continue;
      const [x1, y1, x2, y2] = item.bbox;
      const cls = safeText(item.class_name, 'obj');
      const conf = Number(item.confidence || 0);
      const color = cls === 'fire' ? '#ff5252' : (cls === 'smoke' ? '#ffb300' : '#4fc3f7');
      const rx = x1 * scaleX;
      const ry = y1 * scaleY;
      const rw = Math.max(1, (x2 - x1) * scaleX);
      const rh = Math.max(1, (y2 - y1) * scaleY);

      ctx.strokeStyle = color;
      ctx.strokeRect(rx, ry, rw, rh);

      const text = `${cls} ${Math.round(conf * 100)}%`;
      const metrics = ctx.measureText(text);
      const labelW = metrics.width + 10;
      const labelH = 18;
      ctx.fillStyle = color;
      ctx.fillRect(rx, Math.max(0, ry - labelH), labelW, labelH);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, rx + 5, Math.max(12, ry - 5));
    }
  }

  function updateSiteIntro() {
    if (!els.siteIntroText) return;
    const building = getBuilding(state.selectedBuildingId);
    const currentCam = selectedCameraSnapshot();
    const level = safeText(currentCam?.last_detection?.final_alarm, 'normal');
    const label = level === 'fire' ? '火焰告警' : (level === 'smoke' ? '烟雾告警' : '正常');
    const inferMs = safeText(currentCam?.last_detection?.infer_ms, '--');
    const source = safeText(currentCam?.last_detection?.final_source, '-');
    els.siteIntroText.textContent = `${safeText(building?.name, '当前建筑')} · ${shortCameraId(state.selectedCameraId)} 当前处于 ${label} 状态，综合判定来源 ${source}，最近一次推理耗时 ${inferMs} ms。`;
  }

  function selectedCameraSnapshot() {
    const cameraId = normalizeCameraId(state.selectedCameraId);
    if (!cameraId) return null;
    return cameraSnapshotsMap().get(cameraId) || null;
  }

  function renderSelectedDetectionFromPayload() {
    const snap = selectedCameraSnapshot();
    renderDetection(snap?.last_detection || null);
  }

  function openVideoModal(cameraId) {
    state.selectedCameraId = normalizeCameraId(cameraId);
    if (els.videoModal) {
      els.videoModal.classList.remove('hidden');
      els.videoModal.setAttribute('aria-hidden', 'false');
    }
    if (els.videoModalTitle) {
      els.videoModalTitle.textContent = `${shortCameraId(cameraId)} 实时监控画面`;
    }
    refreshStream(true);
    renderSelectedDetectionFromPayload();
    setTimeout(() => drawBoxes(state.lastDetection), 80);
  }

  function closeVideoModal() {
    if (!els.videoModal) return;
    els.videoModal.classList.add('hidden');
    els.videoModal.setAttribute('aria-hidden', 'true');
    if (els.videoImg) {
      try { els.videoImg.src = ''; } catch (e) {}
    }
  }

  function refreshStream(immediate = false) {
    if (!els.videoImg || !state.selectedCameraId || els.videoModal?.classList.contains('hidden')) return;
    const apply = () => {
      const url = `/demo/stream?camera_id=${encodeURIComponent(state.selectedCameraId)}&_t=${Date.now()}`;
      try { els.videoImg.src = url; } catch (e) {}
    };
    if (immediate) apply();
    else setTimeout(apply, 60);
  }

  function scheduleStreamRetry() {
    if (state.streamRetryTimer) return;
    const delay = Math.min(6000, state.streamRetryMs);
    state.streamRetryTimer = setTimeout(() => {
      state.streamRetryTimer = null;
      refreshStream(true);
      state.streamRetryMs = Math.min(8000, Math.round(state.streamRetryMs * 1.6));
    }, delay);
  }

  function startStreamWatchdog() {
    if (state.streamRefreshTimer) clearInterval(state.streamRefreshTimer);
    state.streamRefreshTimer = setInterval(() => refreshStream(true), 60000);
  }

  function initStreamHandlers() {
    if (!els.videoImg) return;
    els.videoImg.addEventListener('error', () => scheduleStreamRetry());
    els.videoImg.addEventListener('load', () => {
      state.streamRetryMs = 400;
    });
  }

  function stopEvents() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
  }

  function startEvents() {
    stopEvents();
    const url = `/demo/events?_t=${Date.now()}`;
    state.eventSource = new EventSource(url);
    state.eventSource.onopen = () => {
      state.streamRetryMs = 400;
    };
    state.eventSource.onerror = () => {
      stopEvents();
      scheduleStreamRetry();
      setTimeout(startEvents, 1000);
    };
    state.eventSource.onmessage = (evt) => {
      if (!evt?.data) return;
      let payload = null;
      try {
        payload = JSON.parse(evt.data);
      } catch (e) {
        return;
      }
      state.payload = payload;
      const cameras = Array.isArray(payload?.cameras) ? payload.cameras : [];

      maybeAppendAlarmLog(payload, cameras);
      renderEnvGauges(sensorSnapshots());
      renderOverview();
      renderCurrentSelection();
      renderOverallKpis();
      renderSelectedDetectionFromPayload();
      updateTrendPoints();
      renderTrendSparkline();
    };
  }

  async function fetchOverviewConfig() {
    const r = await fetch('/demo/overview_config', { cache: 'no-store' });
    const j = await r.json();
    state.overviewConfig = j;
    if (!state.selectedBuildingId && Array.isArray(j?.buildings) && j.buildings.length) {
      state.selectedBuildingId = j.buildings[0].building_id;
    }
    renderOverview();
    renderCurrentSelection();
    renderViewState();
  }

  function focusAiPanel() {
    if (!els.aiRiskPanel) return;
    els.aiRiskPanel.classList.add('is-focused');
    setTimeout(() => els.aiRiskPanel?.classList.remove('is-focused'), 1400);
  }

  function bindUiEvents() {
    document.querySelectorAll('.dt-nav-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const nextView = btn.getAttribute('data-nav-view') || 'overview';
        setCurrentView(nextView);
        if (btn.getAttribute('data-nav-focus') === 'ai') {
          focusAiPanel();
        }
      });
    });

    if (els.btnBackToOverview) {
      els.btnBackToOverview.addEventListener('click', () => {
        setCurrentView('overview');
      });
    }

    if (els.overviewLayer) {
      els.overviewLayer.addEventListener('click', (e) => {
        const btn = e.target?.closest?.('.dt-overview-building');
        if (!btn) return;
        const buildingId = btn.getAttribute('data-building-id');
        if (!buildingId) return;
        selectBuilding(buildingId, 'building_detail');
      });
    }

    document.addEventListener('pointerdown', (e) => {
      if (state.currentView !== 'building_detail') return;
      if (!els.detailFloatPanel || els.detailFloatPanel.classList.contains('hidden')) return;
      const target = e.target;
      if (!(target instanceof Element)) return;
      if (target.closest('#detailFloatPanel')) return;
      if (target.closest('#videoModalContent')) return;
      if (target.closest('.simple-building')) return;
      if (target.closest('.dt-overview-building')) return;
      setCurrentView('overview');
    });

    if (els.camLayer) {
      els.camLayer.addEventListener('click', (e) => {
        const btn = e.target?.closest?.('.dt-cam-point');
        if (!btn) return;
        const cameraId = btn.getAttribute('data-camera-id');
        if (!cameraId) return;
        openVideoModal(cameraId);
      });
    }

    if (els.videoModalClose) {
      els.videoModalClose.addEventListener('click', closeVideoModal);
    }

    if (els.videoModal) {
      els.videoModal.addEventListener('click', (e) => {
        if (e.target === els.videoModal) closeVideoModal();
      });
    }

    if (els.videoModalDragHandle && els.videoModalContent) {
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
        const rect = els.videoModalContent.getBoundingClientRect();
        const minLeft = 8;
        const minTop = 8;
        const maxLeft = Math.max(minLeft, window.innerWidth - rect.width - 8);
        const maxTop = Math.max(minTop, window.innerHeight - rect.height - 8);
        els.videoModalContent.style.left = `${Math.min(maxLeft, Math.max(minLeft, nextLeft))}px`;
        els.videoModalContent.style.top = `${Math.min(maxTop, Math.max(minTop, nextTop))}px`;
      };

      const stopDrag = () => {
        dragging = false;
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', stopDrag);
      };

      els.videoModalDragHandle.addEventListener('mousedown', (ev) => {
        if (ev.target?.closest?.('#videoModalClose')) return;
        ev.preventDefault();
        const rect = els.videoModalContent.getBoundingClientRect();
        dragging = true;
        startX = ev.clientX;
        startY = ev.clientY;
        startLeft = rect.left;
        startTop = rect.top;
        els.videoModalContent.style.transform = 'none';
        els.videoModalContent.style.left = `${startLeft}px`;
        els.videoModalContent.style.top = `${startTop}px`;
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', stopDrag);
      });
    }
  }

  
  function bindSimpleBuildings() {
    document.querySelectorAll('.simple-building[data-building-id]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const buildingId = btn.getAttribute('data-building-id');
        if (buildingId) selectBuilding(buildingId, 'building_detail');
      });
    });
  }
  function updateClock() {
    if (!els.headerTime) return;
    const now = new Date();
    const yyyy = now.getFullYear();
    const MM = String(now.getMonth() + 1).padStart(2, '0');
    const dd = String(now.getDate()).padStart(2, '0');
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    const weekdays = ['\u661f\u671f\u65e5', '\u661f\u671f\u4e00', '\u661f\u671f\u4e8c', '\u661f\u671f\u4e09', '\u661f\u671f\u56db', '\u661f\u671f\u4e94', '\u661f\u671f\u516d'];
    const weekday = weekdays[now.getDay()];
    els.headerTime.textContent = `${yyyy}-${MM}-${dd}  ${hh}:${mm}:${ss}  ${weekday}`;
  }

  window.addEventListener('resize', () => { drawBoxes(state.lastDetection); });
  window.clearTrendPoints = async () => {
    try {
      await fetch('/demo/alarm_trend', { method: 'DELETE' });
    } catch (e) {}
    state.trendPoints = [];
    state.trendEwma = 0;
    renderTrendSparkline();
  };
  window.clearAlarmLogs = async () => {
    try {
      await fetch('/demo/alarm_logs', { method: 'DELETE' });
    } catch (e) {}
    state.alarmLogCache = [];
    renderAlarmList();
    renderTodayAlarmSummary();
  };

  async function init() {
    bindUiEvents();
    bindSimpleBuildings();
    bindMapViewportDrag();
    initStreamHandlers();
    startStreamWatchdog();
    updateClock();
    setInterval(updateClock, 1000);
    updateMapViewport(true);
    state.overviewConfig = STATIC_OVERVIEW;
    await Promise.all([fetchAlarmLogs(), fetchTrendPoints()]);
    startEvents();
  }

  init().catch((error) => {
    console.error('Failed to initialize demo platform:', error);
  });
})();






