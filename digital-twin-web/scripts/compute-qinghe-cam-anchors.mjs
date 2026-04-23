/**
 * 用与 twin-model.js 相同的 fit + 射线逻辑计算 CAM 锚点在 modelRoot 下的局部坐标，
 * 写入 qinghe-building.anchors.json（modelLocal，与页面 fit 后一致、不漂移）。
 *
 * 用法：node compute-qinghe-cam-anchors.mjs [path/to/qinghe-building.glb]
 */
import { readFileSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

globalThis.self = globalThis;

const CAMERA_ANCHOR_LAYOUT = {
  'CAM-01': { x: 0.28, z: 0.42, y: 0.72 },
  'CAM-02': { x: 0.48, z: 0.52, y: 0.78 },
  'CAM-03': { x: 0.72, z: 0.44, y: 0.7 },
};

function fitModelToView(modelRoot, camera) {
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
  return new THREE.Vector3(wx, boxTight.min.y + spanY * layout.y, wz);
}

function loadGlb(path) {
  const buf = readFileSync(path);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.parse(ab, '', resolve, reject);
  });
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const glbArg = process.argv[2];
const glbPath = glbArg
  ? join(process.cwd(), glbArg)
  : join(__dirname, '../static/models/qinghe-building.glb');

const gltf = await loadGlb(glbPath);
const modelRoot = gltf.scene || gltf.scenes?.[0];
if (!modelRoot) throw new Error('Empty GLB scene');

modelRoot.traverse((child) => {
  if (child.isMesh && child.material) child.material.needsUpdate = true;
});

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
fitModelToView(modelRoot, camera);

modelRoot.updateMatrixWorld(true);
const box =
  computeTightMeshBox3(modelRoot) || new THREE.Box3().setFromObject(modelRoot);
const meshes = collectModelMeshes(modelRoot);

const anchors = {};
for (const camId of ['CAM-01', 'CAM-02', 'CAM-03']) {
  const layout = CAMERA_ANCHOR_LAYOUT[camId];
  const worldPoint = worldPointOnBuildingSurface(meshes, modelRoot, box, layout);
  modelRoot.updateMatrixWorld(true);
  const local = modelRoot.worldToLocal(worldPoint.clone());
  anchors[camId] = {
    bind: 'modelLocal',
    x: Number(local.x.toFixed(5)),
    y: Number(local.y.toFixed(5)),
    z: Number(local.z.toFixed(5)),
  };
}

const outPath = glbPath.replace(/\.(glb|gltf)$/i, '.anchors.json');
const payload = {
  version: 1,
  _generatedBy: 'scripts/compute-qinghe-cam-anchors.mjs',
  anchors,
};

writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
console.log('Wrote', outPath);
console.log(JSON.stringify(anchors, null, 2));
