import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const viewport = document.getElementById('twinModelViewport');
const loadingEl = document.getElementById('twinModelLoading');
const statusEl = document.getElementById('twinModelStatus');

// 水平：相对「主体抬高区域」包围盒的 xz（见 computeAnchorHorizonBox3），避免广场/道路把点拉到地面。
// x 左–中–右；z 较小偏一侧檐廊，反了可调大 z（如 0.55–0.75）。
const CAMERA_ANCHOR_LAYOUT = {
  'CAM-01': { x: 0.28, z: 0.42, y: 0.72 },
  'CAM-02': { x: 0.48, z: 0.52, y: 0.78 },
  'CAM-03': { x: 0.72, z: 0.44, y: 0.70 },
};

// 可选：与 GLB 同名的侧车 JSON（如 qinghe-building.anchors.json）优先；此处仅作无 JSON 时的备用节点名。
const CAMERA_GLTF_NODE_NAME = {
  'CAM-01': '',
  'CAM-02': '',
  'CAM-03': '',
};

if (viewport) {
  initTwinModel().catch((error) => {
    console.error('Failed to initialize twin model:', error);
    setStatus('三维模型加载失败，请检查模型文件或网络依赖。');
    if (loadingEl) loadingEl.textContent = '三维模型加载失败';
  });
}

async function initTwinModel() {
  const modelUrl = viewport.dataset.modelUrl;
  if (!modelUrl) {
    setStatus('未配置三维模型路径。');
    return;
  }

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
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
  const [gltf, anchorConfig] = await Promise.all([
    loader.loadAsync(modelUrl),
    fetchAnchorConfig(modelUrl),
  ]);
  const modelRoot = gltf.scene || gltf.scenes?.[0];
  if (!modelRoot) throw new Error('GLB scene is empty');

  modelRoot.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
      if (child.material) {
        child.material.needsUpdate = true;
      }
    }
  });

  fitModelToView(modelRoot, camera, controls);
  scene.add(modelRoot);

  const cameraPointButtons = Array.from(
    document.querySelectorAll('#camLayer .dt-cam-point')
  );
  const cameraAnchors = createCameraAnchors(
    modelRoot,
    cameraPointButtons,
    anchorConfig
  );

  if (loadingEl) loadingEl.classList.add('hidden');
  setStatus('清河楼模型已加载：拖动旋转，滚轮缩放，右键拖曳平移（左右移动视点）。');

  const resizeObserver = new ResizeObserver(() => {
    const width = Math.max(viewport.clientWidth, 1);
    const height = Math.max(viewport.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  });
  resizeObserver.observe(viewport);

  controls.addEventListener('start', () => {
    controls.autoRotate = false;
  });

  const clock = new THREE.Clock();
  const animate = () => {
    const t = clock.getElapsedTime();
    ring.material.opacity = 0.12 + (Math.sin(t * 1.35) + 1) * 0.04;
    floor.material.opacity = 0.58 + (Math.sin(t * 0.6) + 1) * 0.03;

    controls.update();
    syncDomCameraPoints(cameraAnchors, camera, viewport);
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  };
  animate();
}

function fitModelToView(modelRoot, camera, controls) {
  const box = new THREE.Box3().setFromObject(modelRoot);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  modelRoot.position.sub(center);

  const maxAxis = Math.max(size.x, size.y, size.z) || 1;
  const scale = 4.8 / maxAxis;
  modelRoot.scale.setScalar(scale);

  const scaledBox = new THREE.Box3().setFromObject(modelRoot);
  const scaledSize = scaledBox.getSize(new THREE.Vector3());
  const scaledCenter = scaledBox.getCenter(new THREE.Vector3());

  modelRoot.position.x -= scaledCenter.x;
  modelRoot.position.y -= scaledCenter.y;
  modelRoot.position.z -= scaledCenter.z;
  modelRoot.position.y += scaledSize.y * 0.5;

  const distance = Math.max(scaledSize.x, scaledSize.y, scaledSize.z) * 2.1;
  camera.position.set(distance * 0.8, distance * 0.55, distance);
  camera.lookAt(0, scaledSize.y * 0.35, 0);

  controls.target.set(0, scaledSize.y * 0.35, 0);
  controls.minDistance = Math.max(2.5, distance * 0.35);
  controls.maxDistance = Math.max(8, distance * 2.2);
  controls.update();
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

/** 与 .glb 同路径、扩展名为 .anchors.json 的侧车配置（放在 static/models 下即可） */
async function fetchAnchorConfig(modelUrl) {
  try {
    const path = modelUrl.split('?')[0];
    const jsonUrl = path.replace(/\.(glb|gltf)$/i, '.anchors.json');
    if (jsonUrl === path) return null;
    const res = await fetch(jsonUrl, { cache: 'no-cache' });
    if (!res.ok) return null;
    const data = await res.json();
    return data && typeof data === 'object' ? data : null;
  } catch {
    return null;
  }
}

/**
 * 侧车 anchors[camId]：
 * - { "bind": "node", "name": "GLB内节点名", "offset": { "x", "y", "z" } } 锚点挂到该节点下（推荐，不漂移）
 * - { "bind": "parent", "parent": "父节点名", "x", "y", "z" } 在父节点局部坐标系下的位置
 * - { "bind": "modelLocal", "x", "y", "z" } 在 GLB 根（modelRoot）局部坐标系下的位置
 */
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
    anchor.position.set(
      Number(cfg.x) || 0,
      Number(cfg.y) || 0,
      Number(cfg.z) || 0
    );
    return true;
  }
  if (bind === 'modellocal' && cfg.x !== undefined && cfg.x !== null) {
    modelRoot.add(anchor);
    anchor.position.set(
      Number(cfg.x),
      Number(cfg.y) || 0,
      Number(cfg.z) || 0
    );
    return true;
  }
  return false;
}

function tryPlaceAnchorOnNamedNode(modelRoot, anchor, nodeName, offset) {
  if (!nodeName || typeof nodeName !== 'string') return false;
  const target = modelRoot.getObjectByName(nodeName.trim());
  if (!target) return false;
  target.add(anchor);
  anchor.position.set(offset?.x || 0, offset?.y || 0, offset?.z || 0);
  return true;
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

/** 仅用「伸到一定高度」的网格并盒，缩小 xz 到楼体附近，排除大平地/路面 */
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

/** 从模型上方垂直打射线，把点落到可见三角面上（屋顶/檐口等），仍随 modelRoot 变换 */
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
  const highHits = hits.filter(
    (h) => h.point.y >= roofCut && h.point.y <= roofMax
  );
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
  return new THREE.Vector3(
    wx,
    boxTight.min.y + spanY * layout.y,
    wz
  );
}

function createCameraAnchors(modelRoot, cameraPointButtons, anchorConfig) {
  modelRoot.updateMatrixWorld(true);
  const box =
    computeTightMeshBox3(modelRoot) || new THREE.Box3().setFromObject(modelRoot);
  const meshes = collectModelMeshes(modelRoot);
  const anchors = [];
  const fileAnchors =
    anchorConfig && anchorConfig.anchors && typeof anchorConfig.anchors === 'object'
      ? anchorConfig.anchors
      : null;

  for (const button of cameraPointButtons) {
    const cameraId = button.getAttribute('data-camera-id') || '';
    const layout = CAMERA_ANCHOR_LAYOUT[cameraId];
    if (!layout) continue;

    const anchor = new THREE.Object3D();
    let placed = false;

    if (fileAnchors && fileAnchors[cameraId]) {
      placed = tryPlaceAnchorFromSidecar(modelRoot, anchor, fileAnchors[cameraId]);
    }
    if (!placed) {
      const nodeName = CAMERA_GLTF_NODE_NAME[cameraId];
      placed = tryPlaceAnchorOnNamedNode(modelRoot, anchor, nodeName, {
        x: 0,
        y: 0,
        z: 0,
      });
    }
    if (!placed) {
      const worldPoint = worldPointOnBuildingSurface(meshes, modelRoot, box, layout);
      anchor.position.copy(modelRoot.worldToLocal(worldPoint.clone()));
      modelRoot.add(anchor);
    }

    anchors.push({ cameraId, button, anchor });
  }

  return anchors;
}

function syncDomCameraPoints(anchors, camera, container) {
  const layer = document.getElementById('camLayer');
  const viewRect = container.getBoundingClientRect();
  const layerRect = layer ? layer.getBoundingClientRect() : viewRect;
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
