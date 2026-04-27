import * as THREE from '/static/vendor/three/build/three.module.js?v=20260426b';
import { OrbitControls } from '/static/vendor/three/examples/jsm/controls/OrbitControls.js?v=20260426b';
import { GLTFLoader } from '/static/vendor/three/examples/jsm/loaders/GLTFLoader.js?v=20260426b';

const viewport = document.getElementById('twinModelViewport');
const loadingEl = document.getElementById('twinModelLoading');
const statusEl = document.getElementById('twinModelStatus');
const detailLayer = document.getElementById('detailLayer');

const CAMERA_ANCHOR_LAYOUT = {
  'CAM-01': { x: 0.28, z: 0.42, y: 0.72 },
  'CAM-02': { x: 0.48, z: 0.52, y: 0.78 },
  'CAM-03': { x: 0.72, z: 0.44, y: 0.70 },
};

const CAMERA_GLTF_NODE_NAME = {
  'CAM-01': '',
  'CAM-02': '',
  'CAM-03': '',
};

const TwinModelViewer = {
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  loader: null,
  floor: null,
  ring: null,
  resizeObserver: null,
  modelRoot: null,
  cameraAnchors: [],
  rafId: 0,
  clock: null,
  activeKey: '',
  loadingKey: '',
  loadingPromise: null,
  preparedKeys: new Set(),
  renderFrame: () => {},
};

const MODEL_ASSET_CACHE = new Map();
const FALLBACK_MODEL_MATERIAL = new THREE.MeshStandardMaterial({
  color: 0xd7d0c8,
  roughness: 0.82,
  metalness: 0.02,
});

window.TwinModelViewer = {
  load: async (config = {}) => {
    if (!viewport) return;
    await ensureViewer();
    await loadModel(config);
  },
  release: () => {
    releaseModelAssets();
  },
  setStatus,
  debugState: () => ({
    activeKey: TwinModelViewer.activeKey,
    loadingKey: TwinModelViewer.loadingKey,
    hasModel: Boolean(TwinModelViewer.modelRoot),
    anchorCount: TwinModelViewer.cameraAnchors.length,
    camera: TwinModelViewer.camera ? TwinModelViewer.camera.position.toArray() : null,
    target: TwinModelViewer.controls ? TwinModelViewer.controls.target.toArray() : null,
  }),
};

if (viewport) {
  const initialModelUrl = viewport.dataset.modelUrl || '';
  ensureViewer()
    .then(async () => {
      if (initialModelUrl) preloadModelAsset(initialModelUrl);
      await loadModel({
        modelUrl: viewport.dataset.modelUrl,
        anchorsUrl: viewport.dataset.anchorsUrl || '',
      });
    })
    .catch((error) => {
      console.error('Failed to initialize twin model:', error);
      setStatus('三维模型加载失败，请检查模型文件或网络依赖。');
      if (loadingEl) loadingEl.textContent = '三维模型加载失败';
    });

  window.addEventListener('beforeunload', () => {
    releaseModelAssets();
  });
}

async function ensureViewer() {
  if (TwinModelViewer.renderer) return;

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(Math.max(viewport.clientWidth, 1), Math.max(viewport.clientHeight, 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = null;
  scene.fog = new THREE.Fog(0xf8ece5, 12, 28);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
  camera.position.set(6.5, 4.2, 8.5);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.enablePan = true;
  controls.panSpeed = 0.65;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.45;
  controls.minDistance = 3;
  controls.maxDistance = 16;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.target.set(0, 1.2, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 1.8));

  const keyLight = new THREE.DirectionalLight(0xfff3e8, 2.6);
  keyLight.position.set(8, 12, 10);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(1024, 1024);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0xffc7a0, 1.3);
  rimLight.position.set(-8, 6, -5);
  scene.add(rimLight);

  const fillLight = new THREE.PointLight(0xffffff, 0.8, 30, 2);
  fillLight.position.set(0, 5, 0);
  scene.add(fillLight);

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(7.5, 64),
    new THREE.MeshStandardMaterial({
      color: 0xf1d7cb,
      transparent: true,
      opacity: 0.65,
      roughness: 0.92,
      metalness: 0.02,
    })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.02;
  floor.receiveShadow = true;
  scene.add(floor);

  const ring = new THREE.Mesh(
    new THREE.RingGeometry(5.8, 6.2, 96),
    new THREE.MeshBasicMaterial({
      color: 0xe53935,
      transparent: true,
      opacity: 0.18,
      side: THREE.DoubleSide,
    })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.01;
  scene.add(ring);

  const loader = new GLTFLoader();
  const resizeObserver = new ResizeObserver(() => {
    const width = Math.max(viewport.clientWidth, 1);
    const height = Math.max(viewport.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    TwinModelViewer.renderFrame();
  });
  resizeObserver.observe(viewport);

  controls.addEventListener('start', () => {
    controls.autoRotate = false;
  });
  controls.addEventListener('change', () => {
    TwinModelViewer.renderFrame();
  });

  TwinModelViewer.renderer = renderer;
  TwinModelViewer.scene = scene;
  TwinModelViewer.camera = camera;
  TwinModelViewer.controls = controls;
  TwinModelViewer.loader = loader;
  TwinModelViewer.floor = floor;
  TwinModelViewer.ring = ring;
  TwinModelViewer.resizeObserver = resizeObserver;
  TwinModelViewer.clock = new THREE.Clock();

  const renderFrame = () => {
    if (!TwinModelViewer.renderer || !TwinModelViewer.scene || !TwinModelViewer.camera) return;
    syncDomCameraPoints(TwinModelViewer.cameraAnchors, camera, viewport);
    renderer.render(scene, camera);
  };

  const animate = () => {
    const t = TwinModelViewer.clock.getElapsedTime();
    ring.material.opacity = 0.12 + (Math.sin(t * 1.35) + 1) * 0.04;
    floor.material.opacity = 0.58 + (Math.sin(t * 0.6) + 1) * 0.03;

    controls.update();
    renderFrame();
    TwinModelViewer.rafId = requestAnimationFrame(animate);
  };
  TwinModelViewer.renderFrame = renderFrame;
  animate();
}

async function loadModel(config = {}) {
  const modelUrl = config.modelUrl || viewport.dataset.modelUrl;
  const anchorsUrl = config.anchorsUrl || viewport.dataset.anchorsUrl || '';
  const cacheKey = `${modelUrl}|${anchorsUrl}`;
  if (!modelUrl || !TwinModelViewer.loader) {
    setStatus('未配置三维模型路径。');
    return;
  }
  viewport.dataset.modelUrl = modelUrl;
  viewport.dataset.anchorsUrl = anchorsUrl;
  if (TwinModelViewer.activeKey === cacheKey && TwinModelViewer.modelRoot) {
    refreshCameraAnchors(anchorsUrl);
    if (loadingEl) loadingEl.classList.add('hidden');
    setModelReady(true);
    setStatus('建筑模型已加载：拖动旋转，滚轮缩放，右键拖曳平移。');
    return;
  }
  if (TwinModelViewer.loadingKey === cacheKey && TwinModelViewer.loadingPromise) {
    return TwinModelViewer.loadingPromise;
  }

  setModelReady(false);

  if (loadingEl) {
    loadingEl.textContent = '正在加载建筑三维模型...';
    loadingEl.classList.remove('hidden');
  }

  TwinModelViewer.loadingKey = cacheKey;
  TwinModelViewer.loadingPromise = (async () => {
    try {
      clearCurrentModel();

      const [asset, anchorConfig] = await Promise.all([
        preloadModelAsset(modelUrl),
        fetchAnchorConfig(modelUrl, anchorsUrl),
      ]);

      const modelRoot = asset?.scene || asset?.scenes?.[0] || null;
      if (!modelRoot) throw new Error('GLB scene is empty');

      const meshCount = prepareModelRoot(modelRoot, cacheKey);

      if (meshCount === 0) throw new Error('Model contains no mesh');

      const modelBox = modelRoot.userData.__dtwModelBox || computeTightMeshBox3(modelRoot) || new THREE.Box3().setFromObject(modelRoot);
      fitModelToView(modelRoot, TwinModelViewer.camera, TwinModelViewer.controls, modelBox);
      TwinModelViewer.scene.add(modelRoot);
      TwinModelViewer.modelRoot = modelRoot;
      TwinModelViewer.activeKey = cacheKey;

      TwinModelViewer.cameraAnchors = createCameraAnchors(
        modelRoot,
        Array.from(document.querySelectorAll('#camLayer .dt-cam-point')),
        anchorConfig,
        modelBox,
        meshCount
      );

      TwinModelViewer.renderFrame();
      if (loadingEl) loadingEl.classList.add('hidden');
      setModelReady(true);
      setStatus('建筑模型已加载：拖动旋转，滚轮缩放，右键拖曳平移。');
    } catch (error) {
      console.error('Failed to load building model:', error);
      if (loadingEl) loadingEl.textContent = '建筑模型加载失败';
      setModelReady(false);
      setStatus('建筑模型加载失败，请检查模型文件或锚点配置。');
    } finally {
      TwinModelViewer.loadingKey = '';
      TwinModelViewer.loadingPromise = null;
    }
  })();

  return TwinModelViewer.loadingPromise;
}

async function preloadModelAsset(modelUrl) {
  if (!modelUrl || !TwinModelViewer.loader) return null;
  if (!MODEL_ASSET_CACHE.has(modelUrl)) {
    MODEL_ASSET_CACHE.set(modelUrl, TwinModelViewer.loader.loadAsync(modelUrl));
  }
  return MODEL_ASSET_CACHE.get(modelUrl);
}

function refreshCameraAnchors(anchorsUrl = '') {
  if (!TwinModelViewer.modelRoot) return;
  fetchAnchorConfig(viewport.dataset.modelUrl || '', anchorsUrl).then((anchorConfig) => {
    TwinModelViewer.cameraAnchors = createCameraAnchors(
      TwinModelViewer.modelRoot,
      Array.from(document.querySelectorAll('#camLayer .dt-cam-point')),
      anchorConfig
    );
  }).catch(() => {
    TwinModelViewer.cameraAnchors = createCameraAnchors(
      TwinModelViewer.modelRoot,
      Array.from(document.querySelectorAll('#camLayer .dt-cam-point')),
      null
    );
  });
}

function clearCurrentModel() {
  if (!TwinModelViewer.scene || !TwinModelViewer.modelRoot) return;
  removeExistingAnchorNodes(TwinModelViewer.modelRoot);
  TwinModelViewer.scene.remove(TwinModelViewer.modelRoot);
  TwinModelViewer.modelRoot = null;
  TwinModelViewer.cameraAnchors = [];
}

function prepareModelRoot(modelRoot, cacheKey) {
  if (TwinModelViewer.preparedKeys.has(cacheKey)) {
    return Number(modelRoot.userData.__dtwMeshCount || 0);
  }

  let meshCount = 0;
  modelRoot.traverse((child) => {
    if (!child.isMesh) return;
    meshCount += 1;
    child.castShadow = true;
    child.receiveShadow = true;
    child.frustumCulled = false;
    if (!child.material || (Array.isArray(child.material) && child.material.length === 0)) {
      child.material = FALLBACK_MODEL_MATERIAL.clone();
    }
    if (Array.isArray(child.material)) {
      child.material = child.material.map((material) => {
        const nextMaterial = material || FALLBACK_MODEL_MATERIAL.clone();
        nextMaterial.side = THREE.DoubleSide;
        nextMaterial.needsUpdate = true;
        return nextMaterial;
      });
    } else {
      child.material.side = THREE.DoubleSide;
      child.material.needsUpdate = true;
    }
  });

  modelRoot.userData.__dtwMeshCount = meshCount;
  modelRoot.userData.__dtwModelBox = computeTightMeshBox3(modelRoot) || new THREE.Box3().setFromObject(modelRoot);
  TwinModelViewer.preparedKeys.add(cacheKey);
  return meshCount;
}

function releaseModelAssets() {
  clearCurrentModel();
  TwinModelViewer.activeKey = '';
  for (const cachedAsset of MODEL_ASSET_CACHE.values()) {
    Promise.resolve(cachedAsset).then((asset) => {
      const sceneRoot = asset?.scene || asset?.scenes?.[0];
      if (sceneRoot) disposeObjectTree(sceneRoot);
    }).catch(() => {});
  }
  MODEL_ASSET_CACHE.clear();
}

function disposeObjectTree(root) {
  root.traverse((child) => {
    if (child.geometry) child.geometry.dispose?.();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach((material) => material?.dispose?.());
      else child.material.dispose?.();
    }
  });
}

function fitModelToView(modelRoot, camera, controls, box = null) {
  box = box || computeTightMeshBox3(modelRoot) || new THREE.Box3().setFromObject(modelRoot);
  const rawSize = box.getSize(new THREE.Vector3());
  const rawCenter = box.getCenter(new THREE.Vector3());
  const maxAxis = Math.max(rawSize.x, rawSize.y, rawSize.z) || 1;

  modelRoot.position.set(-rawCenter.x, -rawCenter.y, -rawCenter.z);
  modelRoot.scale.setScalar(12 / maxAxis);
  modelRoot.updateMatrixWorld(true);

  let fittedBox = new THREE.Box3().setFromObject(modelRoot);
  const fittedCenter = fittedBox.getCenter(new THREE.Vector3());
  modelRoot.position.x -= fittedCenter.x;
  modelRoot.position.z -= fittedCenter.z;
  modelRoot.position.y -= fittedBox.min.y;
  modelRoot.updateMatrixWorld(true);
  fittedBox = new THREE.Box3().setFromObject(modelRoot);

  const sphere = fittedBox.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, 1);
  const height = fittedBox.max.y - fittedBox.min.y;
  const targetY = fittedBox.min.y + height * 0.34;

  camera.near = Math.max(0.01, radius / 200);
  camera.far = Math.max(200, radius * 40);
  camera.position.set(radius * 0.32, radius * 0.42, radius * 0.74);
  camera.lookAt(0, targetY, 0);
  camera.updateProjectionMatrix();

  controls.target.set(0, targetY, 0);
  controls.minDistance = Math.max(0.8, radius * 0.22);
  controls.maxDistance = Math.max(8, radius * 3.4);
  controls.autoRotate = false;
  controls.update();
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

function setModelReady(ready) {
  if (!detailLayer) return;
  detailLayer.classList.toggle('is-model-ready', Boolean(ready));
}

async function fetchAnchorConfig(modelUrl, explicitAnchorsUrl = '') {
  try {
    let jsonUrl = explicitAnchorsUrl || '';
    if (!jsonUrl) {
      const path = modelUrl.split('?')[0];
      jsonUrl = path.replace(/\.(glb|gltf)$/i, '.anchors.json');
      if (jsonUrl === path) return null;
    }
    const res = await fetch(jsonUrl, { cache: 'no-cache' });
    if (!res.ok) return null;
    const data = await res.json();
    return data && typeof data === 'object' ? data : null;
  } catch {
    return null;
  }
}

function tryPlaceAnchorFromSidecar(modelRoot, anchor, cfg) {
  if (!cfg || typeof cfg !== 'object') return false;
  const bind = String(cfg.bind || '').toLowerCase();
  const ox = Number(cfg.offset?.x) || 0;
  const oy = Number(cfg.offset?.y) || 0;
  const oz = Number(cfg.offset?.z) || 0;

  if (bind === 'node' && cfg.name) {
    const target = modelRoot.getObjectByName(String(cfg.name).trim());
    if (!target) return false;
    target.add(anchor);
    anchor.position.set(ox, oy, oz);
    return true;
  }
  if (bind === 'parent' && cfg.parent) {
    const parent = modelRoot.getObjectByName(String(cfg.parent).trim());
    if (!parent) return false;
    parent.add(anchor);
    anchor.userData.__dtwCameraAnchor = true;
    anchor.position.set(Number(cfg.x) || 0, Number(cfg.y) || 0, Number(cfg.z) || 0);
    return true;
  }
  if (bind === 'modellocal' && cfg.x !== undefined && cfg.x !== null) {
    modelRoot.add(anchor);
    anchor.userData.__dtwCameraAnchor = true;
    anchor.position.set(Number(cfg.x), Number(cfg.y) || 0, Number(cfg.z) || 0);
    return true;
  }
  return false;
}

function tryPlaceAnchorOnNamedNode(modelRoot, anchor, nodeName, offset) {
  if (!nodeName || typeof nodeName !== 'string') return false;
  const target = modelRoot.getObjectByName(nodeName.trim());
  if (!target) return false;
  target.add(anchor);
  anchor.userData.__dtwCameraAnchor = true;
  anchor.position.set(offset?.x || 0, offset?.y || 0, offset?.z || 0);
  return true;
}

function removeExistingAnchorNodes(root) {
  if (!root) return;
  const anchors = [];
  root.traverse((child) => {
    if (child?.userData?.__dtwCameraAnchor) anchors.push(child);
  });
  anchors.forEach((anchor) => {
    anchor.parent?.remove(anchor);
  });
}

function computeTightMeshBox3(root) {
  const box = new THREE.Box3();
  let empty = true;
  root.updateMatrixWorld(true);
  root.traverse((child) => {
    if (!child.isMesh || !child.geometry) return;
    const geom = child.geometry;
    if (!geom.boundingBox) geom.computeBoundingBox();
    const ob = geom.boundingBox.clone();
    ob.applyMatrix4(child.matrixWorld);
    if (empty) {
      box.copy(ob);
      empty = false;
    } else {
      box.union(ob);
    }
  });
  return empty ? null : box;
}

function collectModelMeshes(root) {
  const meshes = [];
  root.traverse((child) => {
    if (child.isMesh && child.geometry) meshes.push(child);
  });
  return meshes;
}

function computeAnchorHorizonBox3(root) {
  const loose = new THREE.Box3().setFromObject(root);
  const looseH = loose.max.y - loose.min.y;
  if (looseH < 1e-6) return null;
  const yLine = loose.min.y + looseH * 0.26;
  const box = new THREE.Box3();
  let empty = true;
  root.updateMatrixWorld(true);
  root.traverse((child) => {
    if (!child.isMesh || !child.geometry) return;
    const geom = child.geometry;
    if (!geom.boundingBox) geom.computeBoundingBox();
    const ob = geom.boundingBox.clone();
    ob.applyMatrix4(child.matrixWorld);
    if (ob.max.y < yLine) return;
    if (empty) {
      box.copy(ob);
      empty = false;
    } else {
      box.union(ob);
    }
  });
  return empty ? null : box;
}

function worldPointOnBuildingSurface(meshes, modelRoot, boxTight, layout) {
  modelRoot.updateMatrixWorld(true);
  const loose = new THREE.Box3().setFromObject(modelRoot);
  const horizon = computeAnchorHorizonBox3(modelRoot) || boxTight;
  const min = horizon.min;
  const max = horizon.max;
  const spanX = max.x - min.x;
  const spanZ = max.z - min.z;
  const spanY = boxTight.max.y - boxTight.min.y;
  const wx = min.x + spanX * layout.x;
  const wz = min.z + spanZ * layout.z;
  const looseH = loose.max.y - loose.min.y;
  const raycaster = new THREE.Raycaster();
  const down = new THREE.Vector3(0, -1, 0);
  const top = loose.max.y + Math.max(loose.max.x - loose.min.x, looseH, loose.max.z - loose.min.z) * 0.08 + 0.2;
  const origin = new THREE.Vector3(wx, top, wz);
  raycaster.set(origin, down);
  const hits = raycaster.intersectObjects(meshes, false);
  const up = new THREE.Vector3(0, 1, 0);
  const roofCut = loose.min.y + looseH * 0.22;
  const roofMax = loose.min.y + looseH * 0.58;
  const targetY = loose.min.y + looseH * 0.4;
  const candidates = [];

  for (const h of hits) {
    const nMat = new THREE.Matrix3().getNormalMatrix(h.object.matrixWorld);
    const n = h.face.normal.clone().applyMatrix3(nMat).normalize();
    if (n.dot(up) < 0.15) continue;
    if (h.point.y < roofCut || h.point.y > roofMax) continue;
    candidates.push(h);
  }

  let hit = null;
  if (candidates.length > 0) {
    hit = candidates.reduce((bestHit, h) => {
      const d = Math.abs(h.point.y - targetY);
      const bd = Math.abs(bestHit.point.y - targetY);
      return d < bd ? h : bestHit;
    });
  }
  const highHits = hits.filter((h) => h.point.y >= roofCut && h.point.y <= roofMax);
  if (!hit && highHits.length > 0) {
    hit = highHits.reduce((hi, h) => (!hi || h.point.y > hi.point.y ? h : hi));
  }
  if (!hit && hits.length > 0) {
    hit = hits.reduce((hi, h) => (!hi || h.point.y > hi.point.y ? h : hi));
  }
  if (hit) {
    const p = hit.point.clone();
    const nMat = new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld);
    const n = hit.face.normal.clone().applyMatrix3(nMat).normalize();
    const lift = Math.max(0.02, looseH * 0.008);
    p.addScaledVector(n, lift);
    return p;
  }
  return new THREE.Vector3(wx, boxTight.min.y + spanY * layout.y, wz);
}

function createCameraAnchors(modelRoot, cameraPointButtons, anchorConfig, box = null, meshCount = 0) {
  box = box || computeTightMeshBox3(modelRoot) || new THREE.Box3().setFromObject(modelRoot);
  removeExistingAnchorNodes(modelRoot);
  const anchors = [];
  const fileAnchors = anchorConfig && anchorConfig.anchors && typeof anchorConfig.anchors === 'object'
    ? anchorConfig.anchors
    : null;

  if (fileAnchors) {
    for (const button of cameraPointButtons) {
      const cameraId = button.getAttribute('data-camera-id') || '';
      const shortCameraId = button.getAttribute('data-short-camera-id') || cameraId;
      const anchorKey = button.getAttribute('data-anchor-key') || shortCameraId;
      const anchor = new THREE.Object3D();
      anchor.userData.__dtwCameraAnchor = true;
      const cfg = fileAnchors[anchorKey] || fileAnchors[shortCameraId];
      if (cfg) {
        tryPlaceAnchorFromSidecar(modelRoot, anchor, cfg);
      } else {
        const layout = CAMERA_ANCHOR_LAYOUT[anchorKey] || CAMERA_ANCHOR_LAYOUT[shortCameraId] || CAMERA_ANCHOR_LAYOUT['CAM-02'];
        anchor.position.copy(modelRoot.worldToLocal(simpleWorldPointFromLayout(box, layout).clone()));
        modelRoot.add(anchor);
      }
      anchors.push({ cameraId, button, anchor });
    }
    return anchors;
  }

  let meshes = null;
  let requiresSurfaceFallback = false;

  for (const button of cameraPointButtons) {
    const cameraId = button.getAttribute('data-camera-id') || '';
    const shortCameraId = button.getAttribute('data-short-camera-id') || cameraId;
    const anchorKey = button.getAttribute('data-anchor-key') || shortCameraId;
    const layout = CAMERA_ANCHOR_LAYOUT[anchorKey] || CAMERA_ANCHOR_LAYOUT[shortCameraId] || CAMERA_ANCHOR_LAYOUT['CAM-02'];
    const anchor = new THREE.Object3D();
    anchor.userData.__dtwCameraAnchor = true;
    let placed = false;

    if (fileAnchors && (fileAnchors[anchorKey] || fileAnchors[shortCameraId])) {
      placed = tryPlaceAnchorFromSidecar(modelRoot, anchor, fileAnchors[anchorKey] || fileAnchors[shortCameraId]);
    }
    if (!placed) {
      const nodeName = CAMERA_GLTF_NODE_NAME[anchorKey] || CAMERA_GLTF_NODE_NAME[shortCameraId];
      placed = tryPlaceAnchorOnNamedNode(modelRoot, anchor, nodeName, { x: 0, y: 0, z: 0 });
    }
    if (!placed) {
      requiresSurfaceFallback = true;
      const worldPoint = simpleWorldPointFromLayout(box, layout);
      anchor.position.copy(modelRoot.worldToLocal(worldPoint.clone()));
      modelRoot.add(anchor);
    }

    anchors.push({ cameraId, button, anchor });
  }

  if (requiresSurfaceFallback && meshCount > 0 && meshCount <= 1500) {
    meshes = collectModelMeshes(modelRoot);
    for (const [index, button] of cameraPointButtons.entries()) {
      const anchorKey = button.getAttribute('data-anchor-key') || '';
      const shortCameraId = button.getAttribute('data-short-camera-id') || button.getAttribute('data-camera-id') || '';
      if (fileAnchors && (fileAnchors[anchorKey] || fileAnchors[shortCameraId])) continue;
      const nodeName = CAMERA_GLTF_NODE_NAME[anchorKey] || CAMERA_GLTF_NODE_NAME[shortCameraId];
      if (nodeName) continue;
      const layout = CAMERA_ANCHOR_LAYOUT[anchorKey] || CAMERA_ANCHOR_LAYOUT[shortCameraId] || CAMERA_ANCHOR_LAYOUT['CAM-02'];
      const worldPoint = worldPointOnBuildingSurface(meshes, modelRoot, box, layout);
      const anchor = anchors[index]?.anchor;
      if (!anchor) continue;
      anchor.position.copy(modelRoot.worldToLocal(worldPoint.clone()));
    }
  }

  return anchors;
}

function simpleWorldPointFromLayout(box, layout) {
  const min = box.min;
  const max = box.max;
  const spanX = max.x - min.x;
  const spanY = max.y - min.y;
  const spanZ = max.z - min.z;
  return new THREE.Vector3(
    min.x + spanX * layout.x,
    min.y + spanY * layout.y,
    min.z + spanZ * layout.z
  );
}

function syncDomCameraPoints(anchors, camera, container) {
  const layer = document.getElementById('camLayer');
  if (!layer) return;

  const viewRect = container.getBoundingClientRect();
  const layerRect = layer.getBoundingClientRect();
  const width = Math.max(viewRect.width, 1);
  const height = Math.max(viewRect.height, 1);
  const offsetX = viewRect.left - layerRect.left;
  const offsetY = viewRect.top - layerRect.top;

  const worldPosition = new THREE.Vector3();
  const projected = new THREE.Vector3();
  const viewDirection = new THREE.Vector3();
  camera.getWorldDirection(viewDirection);

  for (const item of anchors) {
    item.anchor.getWorldPosition(worldPosition);
    projected.copy(worldPosition).project(camera);

    const cameraToPoint = worldPosition.clone().sub(camera.position).normalize();
    const facingCamera = cameraToPoint.dot(viewDirection) > 0;
    const inClipSpace =
      projected.z >= -1 &&
      projected.z <= 1 &&
      projected.x >= -1.15 &&
      projected.x <= 1.15 &&
      projected.y >= -1.15 &&
      projected.y <= 1.15;

    if (!facingCamera || !inClipSpace) {
      item.button.style.opacity = '0';
      item.button.style.pointerEvents = 'none';
      continue;
    }

    const localX = (projected.x * 0.5 + 0.5) * width + offsetX;
    const localY = (-projected.y * 0.5 + 0.5) * height + offsetY;
    const depthFade = THREE.MathUtils.clamp(1 - (projected.z + 1) * 0.25, 0.35, 1);

    item.button.style.left = `${localX}px`;
    item.button.style.top = `${localY}px`;
    item.button.style.opacity = `${depthFade}`;
    item.button.style.pointerEvents = 'auto';
  }
}
