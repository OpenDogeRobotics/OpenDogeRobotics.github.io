import "./style.css";
import loadMujoco from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import * as ort from "onnxruntime-web";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import ortMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

// Keep ONNX Runtime's sidecar binary out of Vite's dependency-cache URL too.
// The simulation is intentionally single-threaded so the page stays usable
// without cross-origin-isolation response headers on GitHub Pages.
ort.env.wasm.wasmPaths = {
  mjs: ortMjsUrl,
  wasm: ortWasmUrl,
};
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

// Policies trained by OpenDoge_train.  Both use the mjlab velocity observation
// layout (48 fields) and the same 12-joint position action space:
//   [lin_vel(3), ang_vel(3), projected_gravity(3), joint_pos(12),
//    joint_vel(12), last_action(12), command(3)]
const skillNames = ["walk", "handstand", "getup", "jump"];
const policies = {
  flat: {
    name: "平地速度跟踪",
    file: "/policies/opendoge-flat.onnx",
    inputSize: 48,
    mode: "velocity",
    init: "stand",
    commandRanges: { vx: [-0.8, 0.8], vy: [-0.5, 0.5], yaw: [-0.8, 0.8] },
    note: "LainLab-OpenDoge-Flat 基线策略：平地速度跟踪，可用摇杆或 WASD/QE 给出 vx、vy、ω 指令。",
  },
  rough: {
    name: "粗地形行走",
    file: "/policies/opendoge-rough.onnx",
    inputSize: 213,
    mode: "rough",
    init: "stand",
    commandRanges: { vx: [-0.6, 0.8], vy: [-0.4, 0.4], yaw: [-0.7, 0.7] },
    note: "LainLab-OpenDoge-Rough 基线策略：48 维本体观测 + 165 维地形高度扫描，可在“地形”面板里加台阶或坡道试。",
  },
  skillsWalk: {
    name: "多技能 · 行走",
    file: "/policies/opendoge-skills.onnx",
    inputSize: 57,
    mode: "skills",
    skill: "walk",
    init: "stand",
    commandRanges: { vx: [-0.8, 0.8], vy: [-0.5, 0.5], yaw: [-0.8, 0.8] },
    note: "LainLab-OpenDoge-Skills-Flat：一个网络承载 walk / handstand / getup / jump 四种技能，当前运行 walk。",
  },
  skillsHandstand: {
    name: "多技能 · 倒立",
    file: "/policies/opendoge-skills.onnx",
    inputSize: 57,
    mode: "skills",
    skill: "handstand",
    init: "stand",
    initJointScale: 1.05,
    commandRanges: { vx: [0, 0], vy: [0, 0], yaw: [0, 0] },
    note: "同一网络的 handstand 技能：前腿支撑倒立，机身高度约 0.22 m。",
  },
  skillsGetup: {
    name: "多技能 · 起身",
    file: "/policies/opendoge-skills.onnx",
    inputSize: 57,
    mode: "skills",
    skill: "getup",
    init: "fallen",
    commandRanges: { vx: [0, 0], vy: [0, 0], yaw: [0, 0] },
    note: "同一网络的 getup 技能：从仰卧姿态翻身站起。",
  },
  getup: {
    name: "跌倒起身",
    file: "/policies/opendoge-getup.onnx",
    inputSize: 48,
    init: "fallen",
    commandRanges: { vx: [0, 0], vy: [0, 0], yaw: [0, 0] },
    note: "LainLab-OpenDoge-Getup 基线策略：从仰卧姿态自主翻身站起。",
  },
};

const jointNames = [
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
];
// Matches OPENDOGE_INIT_STATE in the training repository.
const defaultPose = [0, 0.8, -1.6, 0, -0.8, 1.6, 0, 0.8, -1.6, 0, -0.8, 1.6];
const effortLimits = [6, 6, 9, 6, 6, 9, 6, 6, 9, 6, 6, 9];
const ACTUATOR_KP = 12.0;
const ACTUATOR_KD = 0.5;
const ACTUATOR_ARMATURE = 0.005;
const ACTION_SCALE = 0.25;

// MuJoCo settings used for training: 2 ms physics, 2 physics steps per policy
// step (250 Hz control).
const MUJOCO_TIMESTEP = 0.002;
const POLICY_PHYSICS_STEPS = 2;
const POLICY_DT = MUJOCO_TIMESTEP * POLICY_PHYSICS_STEPS;

// Terrain height scan for the rough policy: mjlab's GridPatternCfg with
// size=(0.7, 0.5) and resolution=0.05, ordered y-outer / x-inner as
// torch.meshgrid(indexing="xy") flattens it. Misses report max_distance and the
// observation term scales everything by 1 / max_distance.
const SCAN_COLUMNS = 15;
const SCAN_ROWS = 11;
const SCAN_STEP = 0.05;
const SCAN_MISS = 5.0;
const SCAN_SCALE = 1 / SCAN_MISS;

const TERRAIN_AREA = { x: 3.5, y: 2.5 };
const TERRAIN_SLOT_COUNT = 96;

const meshColor = (name) => {
  if (name.startsWith("base_link")) return 0xdcd8d0;
  if (name.includes("hip")) return 0x4a4944;
  if (name.includes("thigh")) return 0x95928b;
  return 0xc3c0b8;
};

const $ = (id) => document.getElementById(id);
const publicAsset = (path) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, "")}`;
const engineState = $("engineState");
const simulation = {
  running: false,
  elapsed: 0,
  realTimeAccumulator: 0,
  action: new Float32Array(12),
  actionHistory: [],
  command: [0, 0, 0],
};
const terrainState = { kind: "flat", seed: 7, height: 0.08, tool: "platform", elements: [], selected: null };
let mujoco, model, data, policySession, activePolicy, jointAddresses, actuatorAddresses, baseRobotXml, robotAssets, robotVfs, terrainSlotIds, terrainSlotSet;
let physicsStepsPerPolicy = POLICY_PHYSICS_STEPS;
let actualPolicyDt = POLICY_DT;
const controllerState = {
  commandSpeed: [1, 1, 1],
  keyboardKeys: new Set(),
  joystick: { left: { x: 0, y: 0 }, right: { x: 0, y: 0 } },
};

function setStatus(text, kind = "") {
  engineState.textContent = text;
  engineState.className = `pill ${kind}`;
}

function setNotice(text, error = false) {
  if (error) console.warn(text);
}

function setBootStage(text, progress) {
  $("bootStage").textContent = text;
  $("bootProgress").style.width = `${progress}%`;
  $("bootPercent").textContent = `${progress}%`;
}

function revealScene() {
  $("bootScreen").classList.add("reveal");
}

function createScene() {
  const mount = $("scene");
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#faf5e9");
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const canvasMount = document.createElement("div");
  canvasMount.className = "scene-canvas";
  canvasMount.append(renderer.domElement);
  mount.prepend(canvasMount);
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 0, 0.095);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.18;
  orbit.rotateSpeed = 0.48;
  orbit.panSpeed = 0.55;
  orbit.zoomSpeed = 0.72;
  orbit.minDistance = 0.25;
  orbit.maxDistance = 6;
  scene.add(new THREE.HemisphereLight(0xffffff, 0xd6cab4, 2.0));
  const key = new THREE.DirectionalLight(0xfffaf0, 2.2);
  key.position.set(1.2, -1.8, 2.6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -2;
  key.shadow.camera.right = 2;
  key.shadow.camera.top = 2;
  key.shadow.camera.bottom = -2;
  key.shadow.camera.near = 0.1;
  key.shadow.camera.far = 8;
  key.shadow.bias = -0.0005;
  scene.add(key);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(TERRAIN_AREA.x * 3.4, TERRAIN_AREA.y * 3.4),
    new THREE.MeshStandardMaterial({ color: 0xf3ebd9, roughness: 0.92 }),
  );
  floor.receiveShadow = true;
  floor.position.z = -0.002;
  scene.add(floor);
  const grid = new THREE.GridHelper(TERRAIN_AREA.x * 3.4, 68, 0xd9c9a6, 0xeee3cb);
  // GridHelper is horizontal in Three.js' Y-up world. This scene and MuJoCo
  // model use Z-up, so rotate it onto the physical floor plane.
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.001;
  scene.add(grid);
  // Keep the robot framed on both landscape desktops and portrait phones:
  // portrait viewports have a much narrower horizontal field of view.
  const cameraDirection = new THREE.Vector3(0.38, -0.7, 0.215).normalize();
  const frameCamera = () => {
    const distance = 0.86 * Math.max(1, 1.05 / camera.aspect);
    camera.position.copy(orbit.target).addScaledVector(cameraDirection, distance);
    orbit.update();
  };
  const resize = () => {
    const width = mount.clientWidth;
    const height = mount.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    frameCamera();
  };
  new ResizeObserver(resize).observe(mount);
  resize();
  return { scene, camera, renderer, orbit, floor, grid, key };
}

const view = createScene();
const terrainVisual = new THREE.Group();
terrainVisual.name = "simulator-terrain";
view.scene.add(terrainVisual);
const robot = new THREE.Group();
view.scene.add(robot);
const bodyNodes = new Map();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const dragArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0, 0xb57b06, 0.05, 0.025);
dragArrow.visible = false;
view.scene.add(dragArrow);
let robotDrag = null;

function attribute(element, name, fallback = "") {
  return element.getAttribute(name) ?? fallback;
}

function vector(text, size, fallback) {
  const values = text.trim().split(/\s+/).map(Number);
  return values.length === size && values.every(Number.isFinite) ? values : fallback;
}

function localTransform(node, element) {
  const pos = vector(attribute(element, "pos"), 3, [0, 0, 0]);
  const quat = vector(attribute(element, "quat"), 4, [1, 0, 0, 0]);
  node.position.fromArray(pos);
  node.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
}

function parseMeshAssets(xml) {
  const document = new DOMParser().parseFromString(xml, "text/xml");
  return [...document.querySelectorAll("asset > mesh")].map((mesh) => ({
    name: attribute(mesh, "name"),
    file: attribute(mesh, "file"),
  }));
}

async function buildRobot(xml, meshes) {
  const loader = new STLLoader();
  const geometries = new Map(await Promise.all(meshes.map(async ({ name, file }) => [name, await loader.loadAsync(publicAsset(`robot/assets/${file}`))])));
  const document = new DOMParser().parseFromString(xml, "text/xml");
  robot.clear();
  bodyNodes.clear();
  function visit(body) {
    const name = attribute(body, "name");
    if (!name) return;
    const group = new THREE.Group();
    bodyNodes.set(name, group);
    robot.add(group);
    for (const geom of body.children) {
      if (geom.tagName !== "geom" || !geom.hasAttribute("mesh")) continue;
      const geometry = geometries.get(attribute(geom, "mesh"));
      if (!geometry) continue;
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          color: meshColor(attribute(geom, "mesh")),
          roughness: 0.52,
          metalness: 0.22,
        }),
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      localTransform(mesh, geom);
      group.add(mesh);
    }
    for (const child of body.children) if (child.tagName === "body") visit(child);
  }
  for (const body of document.querySelectorAll("worldbody > body")) visit(body);
}

async function initializeMujoco() {
  // Vite pre-bundles JS dependencies during development, but the Emscripten
  // loader resolves its sidecar WASM relative to that transient cache path.
  // Giving it Vite's emitted URL makes dev and production use the same binary.
  mujoco = await loadMujoco({
    locateFile: (file) => (file === "mujoco.wasm" ? mujocoWasmUrl : file),
  });
  baseRobotXml = await fetch(publicAsset("robot/opendoge.xml")).then((response) => response.text());
  const meshes = parseMeshAssets(baseRobotXml);
  const binaries = await Promise.all(meshes.map(async ({ file }) => [file, new Uint8Array(await (await fetch(publicAsset(`robot/assets/${file}`))).arrayBuffer())]));
  robotAssets = binaries;
  robotVfs = new mujoco.MjVFS();
  for (const [file, buffer] of robotAssets) robotVfs.addBuffer(`assets/${file}`, buffer);
  createMujocoModel(terrainXml(), terrainBoxes().length);
}

function createMujocoModel(xml, terrainCount) {
  data?.delete?.();
  model?.delete?.();
  model = mujoco.MjModel.from_xml_string(xml, robotVfs);
  data = new mujoco.MjData(model);
  const policyStepRatio = POLICY_DT / model.opt.timestep;
  if (Math.abs(model.opt.timestep - MUJOCO_TIMESTEP) > 1e-9 || !Number.isInteger(policyStepRatio) || policyStepRatio !== POLICY_PHYSICS_STEPS) {
    throw new Error(`MuJoCo timestep 必须为 ${MUJOCO_TIMESTEP}s，且策略周期必须由 ${POLICY_PHYSICS_STEPS} 个物理步组成。`);
  }
  physicsStepsPerPolicy = policyStepRatio;
  actualPolicyDt = physicsStepsPerPolicy * model.opt.timestep;
  // The JS binding exposes these model fields as plain integers (the Python
  // binding returns length-1 arrays), so accept either shape.
  const scalar = (value) => (typeof value === "number" ? value : value[0]);
  jointAddresses = jointNames.map((name) => ({
    qpos: scalar(model.jnt(name).qposadr),
    dof: scalar(model.jnt(name).dofadr),
  }));
  actuatorAddresses = jointNames.map((name) => scalar(model.actuator(name).id));
  terrainSlotIds = Array.from({ length: terrainCount }, (_, index) => model.geom(`simulator_terrain_${index}`).id);
  terrainSlotSet = new Set(terrainSlotIds);
}

function seededRandom(seed) {
  let value = (Number(seed) || 1) >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function box(x, y, z, sx, sy, sz, yaw = 0, label = "障碍") {
  return { type: "box", x, y, z, sx, sy, sz, yaw, label };
}

// A ramp is a right-triangular prism: flat at -x, full height at +x. mjlab-style
// slopes are expressed in percent, so height = length * slope / 100.
function ramp(x, y, height, length, width, yaw = 0, label = "斜坡") {
  return { type: "ramp", x, y, z: height / 2, sx: length / 2, sy: width / 2, sz: height / 2, yaw, label };
}

const RAMP_FACES = [0, 1, 2, 3, 5, 4, 0, 3, 4, 0, 4, 1, 1, 4, 5, 1, 5, 2, 0, 2, 5, 0, 5, 3];
const rampVertices = ({ sx, sy, sz }) => [-sx, -sy, -sz, sx, -sy, -sz, sx, -sy, sz, -sx, sy, -sz, sx, sy, -sz, sx, sy, sz];

// Terrain dimensions are sized for OpenDoge's ~0.2 m body rather than the
// 0.4 m quadruped the original playground was built for.
function profileBoxes() {
  const height = terrainState.height;
  if (terrainState.kind === "slope") return [ramp(1.15, 0, height * 1.5, 1.3, 1.0, 0, "坡道")];
  if (terrainState.kind === "stairs") return Array.from({ length: 7 }, (_, index) => {
    const step = index + 1;
    return box(0.55 + index * 0.16, 0, height * step / 14, 0.08, 0.45, height * step / 14, 0, "阶梯");
  });
  if (terrainState.kind === "obstacle_mix") {
    const random = seededRandom(terrainState.seed);
    const boxes = [];
    while (boxes.length < 10) {
      const x = -TERRAIN_AREA.x + .35 + random() * (TERRAIN_AREA.x * 2 - .7);
      const y = -TERRAIN_AREA.y + .35 + random() * (TERRAIN_AREA.y * 2 - .7);
      if (Math.hypot(x, y) < 0.45) continue;
      const scale = 0.35 + random() * 0.45;
      boxes.push(box(x, y, height * scale / 2, 0.05 + random() * 0.06, 0.05 + random() * 0.06, height * scale / 2, 0, "随机障碍"));
    }
    return boxes;
  }
  return [];
}

const TERRAIN_ELEMENT_DEFAULTS = {
  platform: { length: 0.32, width: 0.32, height: 0.08 },
  wall: { length: 0.5, width: 0.06, height: 0.1 },
  ramp: { length: 0.55, width: 0.4, height: 0.09 },
  stairs: { stepCount: 4, stepDepth: 0.12, stepWidth: 0.4, stepHeight: 0.03 },
  stones: { length: 0.08, width: 0.08, height: 0.06, columns: 3, rows: 2, gapX: 0.16, gapY: 0.16, layout: "aligned" },
};

function createTerrainElement(kind, x, y) {
  const element = { kind, x, y, yaw: 0, ...TERRAIN_ELEMENT_DEFAULTS[kind] };
  if (kind === "ramp") element.slope = element.height / element.length * 100;
  return element;
}

function elementFootprint(element) {
  const length = element.length ?? 0.3;
  const width = element.width ?? 0.3;
  if (element.kind === "stairs") {
    const count = Math.round(element.stepCount ?? 4);
    return { length: count * (element.stepDepth ?? 0.12), width: element.stepWidth ?? 0.4, yaw: element.yaw || 0 };
  }
  if (element.kind === "stones") {
    const columns = Math.round(element.columns ?? 3);
    const rows = Math.round(element.rows ?? 2);
    const pitchX = length + (element.gapX ?? 0.16);
    const pitchY = width + (element.gapY ?? 0.16);
    const shift = element.layout === "staggered" ? pitchX / 2 : 0;
    return { length: (columns - 1) * pitchX + length + shift, width: (rows - 1) * pitchY + width, yaw: element.yaw || 0 };
  }
  return { length, width, yaw: element.yaw || 0 };
}

function elementBoxes(element) {
  const { x, y, kind } = element;
  const height = Number.isFinite(element.height) ? element.height : terrainState.height;
  const length = Number.isFinite(element.length) ? element.length : 0.3;
  const width = Number.isFinite(element.width) ? element.width : 0.3;
  const yaw = element.yaw || 0;
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const place = (localX, localY, z, sx, sy, sz, label) => box(
    x + localX * cos - localY * sin,
    y + localX * sin + localY * cos,
    z, sx, sy, sz, yaw, label,
  );
  if (kind === "stairs") {
    const count = Math.max(1, Math.round(element.stepCount ?? 4));
    const depth = element.stepDepth ?? 0.12;
    const stepWidth = element.stepWidth ?? 0.4;
    const stepHeight = element.stepHeight ?? 0.03;
    return Array.from({ length: count }, (_, index) => place(
      (index - (count - 1) / 2) * depth, 0,
      stepHeight * (index + 1) / 2, depth / 2, stepWidth / 2, stepHeight * (index + 1) / 2, "台阶",
    ));
  }
  if (kind === "ramp") return [ramp(x, y, height, length, width, yaw, "斜坡")];
  if (kind === "stones") {
    const columns = Math.max(1, Math.round(element.columns ?? 3));
    const rows = Math.max(1, Math.round(element.rows ?? 2));
    const pitchX = length + (element.gapX ?? 0.16);
    const pitchY = width + (element.gapY ?? 0.16);
    return Array.from({ length: columns * rows }, (_, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const offset = element.layout === "staggered" && row % 2 ? pitchX / 2
        : element.layout === "diagonal" ? (row - (rows - 1) / 2) * pitchX / 2 : 0;
      return place(
        (column - (columns - 1) / 2) * pitchX + offset,
        (row - (rows - 1) / 2) * pitchY,
        height / 2, length / 2, width / 2, height / 2, "梅花桩",
      );
    });
  }
  return [place(0, 0, height / 2, length / 2, width / 2, height / 2, kind === "wall" ? "矮墙" : "高台")];
}

function terrainBoxes() {
  // The Three.js floor is visual only. This explicit MuJoCo box carries the
  // physics floor so the robot cannot fall through gaps between obstacles.
  return [box(0, 0, -0.05, 5, 5, .05, 0, "场地地面"), ...profileBoxes(), ...terrainState.elements.flatMap(elementBoxes)];
}

function clearTerrainVisual() {
  terrainVisual.clear();
}

function addVisualBox(definition, accent = 0x9c8a63) {
  const geometry = definition.type === "ramp"
    ? (() => {
      const shape = new THREE.BufferGeometry();
      shape.setAttribute("position", new THREE.Float32BufferAttribute(rampVertices(definition), 3));
      shape.setIndex(RAMP_FACES);
      const geometry = shape.toNonIndexed();
      shape.dispose();
      geometry.computeVertexNormals();
      return geometry;
    })()
    : new THREE.BoxGeometry(definition.sx * 2, definition.sy * 2, definition.sz * 2);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: accent, roughness: .76, metalness: .03 }));
  mesh.position.set(definition.x, definition.y, definition.z);
  mesh.rotation.z = definition.yaw || 0;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  terrainVisual.add(mesh);
}

function renderTerrain() {
  clearTerrainVisual();
  view.floor.material.color.set(terrainState.kind === "flat" ? 0xf3ebd9 : 0xeee4cd);
  for (const definition of profileBoxes()) addVisualBox(definition, 0xc3ae83);
  for (const element of terrainState.elements) for (const definition of elementBoxes(element)) addVisualBox(definition, 0xa8873f);
}

function terrainXml() {
  const document = new DOMParser().parseFromString(baseRobotXml, "text/xml");
  const worldbody = document.querySelector("worldbody");
  const asset = document.querySelector("asset");
  // Joint armature and actuator gains mirror OPENDOGE_ACTUATOR_* in training.
  for (const joint of document.querySelectorAll("worldbody joint")) {
    joint.setAttribute("armature", String(ACTUATOR_ARMATURE));
  }
  for (const geom of document.querySelectorAll("worldbody geom")) {
    if (attribute(geom, "class") !== "collision") continue;
    // mjlab's FULL_COLLISION keeps robot-terrain contacts but disables robot
    // self-collisions (contype=1, conaffinity=0).
    geom.setAttribute("conaffinity", "0");
    if (!geom.getAttribute("name")?.endsWith("_foot_collision")) {
      geom.setAttribute("condim", "1");
      geom.setAttribute("priority", "0");
      continue;
    }
    geom.setAttribute("condim", "3");
    geom.setAttribute("priority", "1");
    geom.setAttribute("friction", "0.8 0.005 0.0001");
    geom.setAttribute("solimp", "0.9 0.95 0.023");
  }
  const definitions = terrainBoxes();
  if (definitions.length > TERRAIN_SLOT_COUNT) throw new Error(`地形最多支持 ${TERRAIN_SLOT_COUNT} 个碰撞组件。`);
  definitions.forEach((definition, index) => {
    const yaw = definition.yaw || 0;
    if (definition.type === "ramp") {
      const mesh = document.createElement("mesh");
      mesh.setAttribute("name", `simulator_ramp_${index}`);
      mesh.setAttribute("vertex", rampVertices(definition).map((value) => Number(value.toFixed(6))).join(" "));
      mesh.setAttribute("face", RAMP_FACES.join(" "));
      asset.append(mesh);
    }
    const geom = document.createElement("geom");
    geom.setAttribute("name", `simulator_terrain_${index}`);
    geom.setAttribute("type", definition.type === "ramp" ? "mesh" : "box");
    if (definition.type === "ramp") geom.setAttribute("mesh", `simulator_ramp_${index}`);
    else geom.setAttribute("size", `${definition.sx} ${definition.sy} ${definition.sz}`);
    geom.setAttribute("pos", `${definition.x} ${definition.y} ${definition.z}`);
    geom.setAttribute("euler", `0 0 ${yaw}`);
    geom.setAttribute("rgba", "0 0 0 0");
    geom.setAttribute("friction", "0.9 0.1 0.1");
    geom.setAttribute("solref", "0.02 1");
    geom.setAttribute("solimp", "0.9 0.95 0.001");
    geom.setAttribute("contype", "1");
    geom.setAttribute("conaffinity", "1");
    worldbody.append(geom);
  });
  const actuator = document.createElement("actuator");
  jointNames.forEach((name, index) => {
    const position = document.createElement("position");
    position.setAttribute("name", name);
    position.setAttribute("joint", name);
    position.setAttribute("kp", String(ACTUATOR_KP));
    position.setAttribute("kv", String(ACTUATOR_KD));
    position.setAttribute("forcerange", `${-effortLimits[index]} ${effortLimits[index]}`);
    position.setAttribute("ctrllimited", "false");
    actuator.append(position);
  });
  const sensor = document.querySelector("sensor");
  if (sensor) document.documentElement.insertBefore(actuator, sensor);
  else document.documentElement.append(actuator);
  return new XMLSerializer().serializeToString(document);
}

async function applyTerrain() {
  if (!mujoco || !baseRobotXml) return;
  simulation.running = false;
  $("start").textContent = "开始";
  const definitions = terrainBoxes();
  if (definitions.length > TERRAIN_SLOT_COUNT) throw new Error(`地形最多支持 ${TERRAIN_SLOT_COUNT} 个碰撞组件。`);
  $("terrainApply").disabled = true;
  setStatus("正在同步地形碰撞…");
  setNotice("正在同步地形碰撞…");
  try {
    // Geometry topology is part of MuJoCo's collision broad-phase. Compile
    // only active boxes, while keeping robot mesh assets cached in memory.
    await new Promise((resolve) => requestAnimationFrame(resolve));
    createMujocoModel(terrainXml(), definitions.length);
    renderTerrain();
    reset();
    setStatus("浏览器物理引擎已就绪", "ready");
    setNotice(`已应用${terrainState.kind === "flat" ? "平地" : "自定义"}地形；物理碰撞已同步。`);
  } finally {
    $("terrainApply").disabled = false;
  }
}

async function loadPolicy(key) {
  activePolicy = policies[key];
  simulation.running = false;
  $("start").textContent = "开始";
  $("policyName").textContent = activePolicy.name;
  const detail = activePolicy.mode === "rough" ? "本体 + 地形扫描" : activePolicy.mode === "skills" ? `多技能 · ${activePolicy.skill}` : "本体观测";
  $("policyMeta").textContent = `${activePolicy.inputSize} 维观测 · ${detail}`;
  simulation.command.fill(0);
  configureCommandControls();
  recomputeCommand();
  updateControls();
  setNotice(`正在加载「${activePolicy.name}」ONNX 策略…`);
  policySession = await ort.InferenceSession.create(publicAsset(activePolicy.file), { executionProviders: ["wasm"] });
  reset();
  setNotice(`${activePolicy.note} 点击“开始”后，物理和策略均在此浏览器内运行。`);
}

function reset() {
  if (!model || !data || !activePolicy) return;
  mujoco.mj_resetData(model, data);
  const fallen = activePolicy.init === "fallen";
  data.qpos[0] = 0;
  data.qpos[1] = 0;
  data.qpos[2] = fallen ? 0.06 : 0.158;
  data.qpos[3] = fallen ? 0 : 1;
  data.qpos[4] = fallen ? 1 : 0;
  data.qpos[5] = 0;
  data.qpos[6] = 0;
  const jointScale = activePolicy.initJointScale ?? 1;
  defaultPose.forEach((value, index) => { data.qpos[jointAddresses[index].qpos] = value * jointScale; });
  data.qvel.fill(0);
  data.ctrl.fill(0);
  simulation.action.fill(0);
  simulation.actionHistory = [];
  simulation.elapsed = 0;
  simulation.realTimeAccumulator = 0;
  mujoco.mj_forward(model, data);
  updateRobot();
}

function bodyQuaternion() {
  return new THREE.Quaternion(data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]);
}

function observation() {
  const inverse = bodyQuaternion().invert();
  // Free-joint linear velocity is global, angular velocity is already body
  // frame. mjlab reads the velocimeter mounted at the imu site, so include the
  // small lever-arm term for that 3 cm offset.
  const angular = new THREE.Vector3(data.qvel[3], data.qvel[4], data.qvel[5]);
  const linear = new THREE.Vector3(data.qvel[0], data.qvel[1], data.qvel[2])
    .applyQuaternion(inverse)
    .add(new THREE.Vector3().crossVectors(angular, new THREE.Vector3(0, 0, 0.03)));
  const gravity = new THREE.Vector3(0, 0, -1).applyQuaternion(inverse);
  const values = [
    linear.x, linear.y, linear.z,
    angular.x, angular.y, angular.z,
    gravity.x, gravity.y, gravity.z,
  ];
  jointAddresses.forEach((address, index) => values.push(data.qpos[address.qpos] - defaultPose[index]));
  jointAddresses.forEach((address) => values.push(data.qvel[address.dof]));
  values.push(...simulation.action);
  values.push(simulation.command[0], simulation.command[1], simulation.command[2]);
  if (activePolicy.mode === "rough") {
    values.push(...heightScan());
  } else if (activePolicy.mode === "skills") {
    // [jump twist(3)] [one_hot(skill)(4)] [phase_sin, phase_cos] — the phase
    // clock only advances for the jump skill, so stationary skills see (0, 1).
    values.push(0, 0, 0);
    const token = skillNames.indexOf(activePolicy.skill);
    for (let index = 0; index < skillNames.length; index += 1) values.push(index === token ? 1 : 0);
    values.push(0, 1);
  }
  return new Float32Array(values);
}

// Vertical ray against the simulator's analytic terrain boxes (all MuJoCo
// terrain geoms are boxes; the robot is excluded by construction). Returns the
// distance to the first hit, or -1, matching mj_ray's convention.
// Vertical ray against the simulator's analytic terrain: boxes are yaw-rotated
// cuboids, ramps are the same triangular prism used for rendering and physics.
// Returns the distance to the first hit, or -1, matching mj_ray's convention.
function rayDown(x, y, z, definitions) {
  let nearest = -1;
  for (const definition of definitions) {
    const yaw = definition.yaw || 0;
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const dx = x - definition.x;
    const dy = y - definition.y;
    const localX = cos * dx + sin * dy;
    const localY = -sin * dx + cos * dy;
    if (Math.abs(localX) > definition.sx || Math.abs(localY) > definition.sy) continue;
    const hitZ = definition.type === "ramp"
      ? definition.z + definition.sz * (localX / definition.sx)
      : definition.z + definition.sz;
    if (hitZ > z) continue;
    const distance = z - hitZ;
    if (nearest < 0 || distance < nearest) nearest = distance;
  }
  return nearest;
}

function heightScan() {
  const bodyId = model.body("base_link").id;
  const px = data.xpos[bodyId * 3];
  const py = data.xpos[bodyId * 3 + 1];
  const pz = data.xpos[bodyId * 3 + 2];
  const matrix = data.xmat;
  const yaw = Math.atan2(matrix[bodyId * 9 + 3], matrix[bodyId * 9]);
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const boxes = terrainBoxes();
  const scan = new Float32Array(SCAN_COLUMNS * SCAN_ROWS);
  let index = 0;
  for (let row = 0; row < SCAN_ROWS; row += 1) {
    const gy = -0.25 + row * SCAN_STEP;
    for (let column = 0; column < SCAN_COLUMNS; column += 1) {
      const gx = -0.35 + column * SCAN_STEP;
      const x = px + cos * gx - sin * gy;
      const y = py + sin * gx + cos * gy;
      const distance = rayDown(x, y, pz, boxes);
      scan[index] = (distance < 0 ? SCAN_MISS : distance) * SCAN_SCALE;
      index += 1;
    }
  }
  return scan;
}

async function policyStep() {
  const obs = observation();
  if (obs.length !== activePolicy.inputSize) throw new Error(`策略观测维度不匹配：${obs.length} / ${activePolicy.inputSize}`);
  const inputName = policySession.inputNames[0];
  const outputName = policySession.outputNames[0];
  const output = await policySession.run({ [inputName]: new ort.Tensor("float32", obs, [1, obs.length]) });
  simulation.action.set(output[outputName].data);
  simulation.actionHistory.push(Array.from(simulation.action));
  if (simulation.actionHistory.length > 90) simulation.actionHistory.shift();
}

function applyControl() {
  jointAddresses.forEach((address, index) => {
    // Position target = default pose + scaled policy action. The PD gains and
    // torque limits live inside the MuJoCo <position> actuators.
    data.ctrl[actuatorAddresses[index]] = defaultPose[index] + ACTION_SCALE * simulation.action[index];
  });
}

function updateRobot() {
  if (!model || !data) return;
  for (const [name, node] of bodyNodes) {
    const id = model.body(name).id;
    node.position.fromArray(data.xpos, id * 3);
    const matrix = data.xmat;
    const offset = id * 9;
    const rotation = new THREE.Matrix4().set(
      matrix[offset], matrix[offset + 1], matrix[offset + 2], 0,
      matrix[offset + 3], matrix[offset + 4], matrix[offset + 5], 0,
      matrix[offset + 6], matrix[offset + 7], matrix[offset + 8], 0,
      0, 0, 0, 1,
    );
    node.quaternion.setFromRotationMatrix(rotation);
  }
}

function updateControls() {
  const [vx, vy, yaw] = simulation.command;
  $("command").textContent = `vx ${vx.toFixed(2)} · vy ${vy.toFixed(2)} · ω ${yaw.toFixed(2)}`;
  $("commandVxValue").textContent = `${vx.toFixed(2)} m/s`;
  $("commandVyValue").textContent = `${vy.toFixed(2)} m/s`;
  $("commandYawValue").textContent = `${yaw.toFixed(2)} rad/s`;
  ["commandSpeedVxValue", "commandSpeedVyValue", "commandSpeedYawValue"].forEach((id, index) => {
    const unit = index === 2 ? "rad/s" : "m/s";
    $(id).textContent = `${controllerState.commandSpeed[index].toFixed(2)} ${unit}`;
  });
  const keyboardActive = controllerState.keyboardKeys.size > 0;
  const joystickActive = Math.abs(controllerState.joystick.left.x) > .001 || Math.abs(controllerState.joystick.left.y) > .001 || Math.abs(controllerState.joystick.right.x) > .001;
  $("controlSource").textContent = keyboardActive ? "键盘 · 满幅" : joystickActive ? "摇杆 · 线性" : "待命";
  syncJoystickVisuals();
  $("leftJoystickValue").textContent = `vx ${vx.toFixed(2)} · vy ${vy.toFixed(2)}`;
  $("rightJoystickValue").textContent = `ω ${yaw.toFixed(2)}`;
}

function configureCommandControls() {
  if (!activePolicy) return;
  const controls = [
    ["commandSpeedVx", activePolicy.commandRanges.vx],
    ["commandSpeedVy", activePolicy.commandRanges.vy],
    ["commandSpeedYaw", activePolicy.commandRanges.yaw],
  ];
  controls.forEach(([id, [min, max]], index) => {
    const input = $(id);
    const maxSpeed = Math.max(Math.abs(min), Math.abs(max));
    input.min = "0";
    input.max = String(maxSpeed || 1);
    input.value = String(maxSpeed);
    input.disabled = maxSpeed === 0;
    input.setAttribute("aria-valuetext", `0 到 ${maxSpeed}`);
    controllerState.commandSpeed[index] = maxSpeed;
  });
}

function keyboardAxis(index) {
  const keys = controllerState.keyboardKeys;
  if (index === 0) {
    if (keys.has("w")) return 1;
    if (keys.has("s")) return -1;
  } else if (index === 1) {
    if (keys.has("a")) return 1;
    if (keys.has("d")) return -1;
  } else {
    if (keys.has("q")) return 1;
    if (keys.has("e")) return -1;
  }
  return null;
}

function inputAxis(index) {
  const keyboard = keyboardAxis(index);
  if (keyboard !== null) return keyboard;
  if (index === 0) return -controllerState.joystick.left.y;
  if (index === 1) return -controllerState.joystick.left.x;
  return -controllerState.joystick.right.x;
}

function recomputeCommand() {
  if (!activePolicy) return;
  ["vx", "vy", "yaw"].forEach((key, index) => {
    const [min, max] = activePolicy.commandRanges[key];
    const value = inputAxis(index) * controllerState.commandSpeed[index];
    simulation.command[index] = THREE.MathUtils.clamp(value, min, max);
  });
  updateControls();
}

function keyboardJoystickPosition(stickKey) {
  const state = controllerState.joystick[stickKey];
  if (stickKey === "left") {
    const vx = keyboardAxis(0);
    const vy = keyboardAxis(1);
    return {
      x: vy === null ? state.x : -vy,
      y: vx === null ? state.y : -vx,
    };
  }
  const yaw = keyboardAxis(2);
  return { x: yaw === null ? state.x : -yaw, y: state.y };
}

function renderJoystickPosition(stickKey) {
  const mount = $(stickKey === "left" ? "leftJoystick" : "rightJoystick");
  const base = mount.querySelector(".joystick-base");
  const knob = mount.querySelector(".joystick-knob");
  const { x, y } = keyboardJoystickPosition(stickKey);
  const radius = Math.max(1, base.getBoundingClientRect().width / 2 - knob.offsetWidth / 2 - 3);
  knob.style.left = `calc(50% + ${x * radius}px)`;
  knob.style.top = `calc(50% + ${y * radius}px)`;
}

function syncJoystickVisuals() {
  renderJoystickPosition("left");
  renderJoystickPosition("right");
}

function drawActionChart() {
  const canvas = $("actionChart");
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fdf8ea";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#ecdfc3";
  context.lineWidth = 1;
  for (let row = 1; row < 4; row += 1) { const y = height * row / 4; context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  context.strokeStyle = "#cdbb95";
  context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
  const samples = simulation.actionHistory;
  const palette = ["#a8760a", "#17140d", "#d19a1e", "#6f5a2d", "#b4551e", "#4a463c"];
  if (samples.length > 1) {
    for (let joint = 0; joint < 12; joint += 1) {
      context.beginPath();
      context.strokeStyle = palette[joint % palette.length];
      context.globalAlpha = joint < 6 ? .78 : .43;
      samples.forEach((sample, index) => {
        const x = index * width / (samples.length - 1);
        const y = height / 2 - THREE.MathUtils.clamp(sample[joint], -1.2, 1.2) * height / 3;
        if (index) context.lineTo(x, y); else context.moveTo(x, y);
      });
      context.stroke();
    }
  }
  context.globalAlpha = 1;
  $("actionValues").replaceChildren(...Array.from(simulation.action, (value, index) => {
    const item = document.createElement("span");
    item.innerHTML = `a${index + 1} <b>${value.toFixed(2)}</b>`;
    return item;
  }));
}

let lastTelemetryDraw = 0;
// Keep the robot centred while it walks or rears up: move the orbit target and
// the camera together so the user's orbit angle and zoom are preserved.
function followRobot(frameDelta) {
  if (!data || robotDrag) return;
  const target = view.orbit.target;
  const desired = new THREE.Vector3(
    data.qpos[0],
    data.qpos[1],
    THREE.MathUtils.clamp(data.qpos[2], 0.08, 0.24),
  );
  // Frame-rate independent smoothing: the target trails the robot by roughly
  // speed * tau, so keep tau short enough that it stays framed while walking.
  const gain = 1 - Math.exp(-Math.max(frameDelta, 0.001) / 0.12);
  const delta = desired.sub(target).multiplyScalar(gain);
  if (delta.lengthSq() < 1e-10) return;
  target.add(delta);
  view.camera.position.add(delta);
}

function updateTelemetry() {
  if (!data) return;
  $("bodyPosition").textContent = `x ${data.qpos[0].toFixed(2)} · y ${data.qpos[1].toFixed(2)} · z ${data.qpos[2].toFixed(2)}`;
  $("contactCount").textContent = String(data.ncon ?? 0);
  let terrainContacts = 0;
  for (let index = 0; index < data.ncon; index += 1) {
    const contact = data.contact.get(index);
    if (contact && (terrainSlotSet.has(contact.geom1) || terrainSlotSet.has(contact.geom2))) terrainContacts += 1;
  }
  $("terrainContactCount").textContent = String(terrainContacts);
  if (performance.now() - lastTelemetryDraw > 100) {
    drawActionChart();
    lastTelemetryDraw = performance.now();
  }
}

function updatePointer(event) {
  const bounds = view.renderer.domElement.getBoundingClientRect();
  pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
  raycaster.setFromCamera(pointer, view.camera);
}

function robotHit(event) {
  updatePointer(event);
  return raycaster.intersectObject(robot, true)[0] ?? null;
}

function installRobotDrag() {
  const canvas = view.renderer.domElement;
  const start = (event) => {
    if (event.button !== 0) return;
    const hit = robotHit(event);
    if (!hit) return;
    const point = hit.point.clone();
    const planePoint = new THREE.Vector3(point.x, point.y, 0);
    view.orbit.enabled = false;
    robotDrag = {
      start: planePoint,
      vector: new THREE.Vector3(),
      pointerId: event.pointerId,
    };
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  };
  const move = (event) => {
    if (!robotDrag || robotDrag.pointerId !== event.pointerId) return;
    updatePointer(event);
    const target = new THREE.Vector3();
    if (!raycaster.ray.intersectPlane(dragPlane, target)) return;
    robotDrag.vector.set(target.x - robotDrag.start.x, target.y - robotDrag.start.y, 0);
    const length = robotDrag.vector.length();
    dragArrow.position.copy(robotDrag.start);
    dragArrow.setDirection(length > .001 ? robotDrag.vector.clone().normalize() : new THREE.Vector3(1, 0, 0));
    dragArrow.setLength(length, 0.04, 0.018);
    dragArrow.visible = true;
    event.preventDefault();
    event.stopImmediatePropagation();
  };
  const release = (event) => {
    if (!robotDrag || !data) return;
    const impulse = robotDrag.vector.clone().multiplyScalar(1.4);
    data.qvel[0] += impulse.x;
    data.qvel[1] += impulse.y;
    dragArrow.visible = false;
    robotDrag = null;
    view.orbit.enabled = true;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    canvas.style.cursor = "";
    if (impulse.lengthSq() > .0005) setNotice("已向机器人施加拖拽向量。", false);
    event.preventDefault();
    event.stopImmediatePropagation();
  };
  canvas.addEventListener("pointerdown", start, true);
  canvas.addEventListener("pointermove", move, true);
  canvas.addEventListener("pointerup", release, true);
  canvas.addEventListener("pointercancel", release, true);
}

let pendingStep = false;
let lastFrameTime = null;
async function tick(timestamp) {
  const now = Number.isFinite(timestamp) ? timestamp : performance.now();
  if (lastFrameTime === null) lastFrameTime = now;
  const frameDelta = Math.max(0, (now - lastFrameTime) / 1000);
  lastFrameTime = now;

  if (simulation.running) {
    // Clamp the backlog so a background tab or a slow frame does not try to
    // replay seconds of physics in one animation frame.
    simulation.realTimeAccumulator = Math.min(simulation.realTimeAccumulator + frameDelta, 0.1);
  } else {
    simulation.realTimeAccumulator = 0;
  }

  if (simulation.running && !pendingStep) {
    pendingStep = true;
    try {
      while (simulation.realTimeAccumulator >= actualPolicyDt) {
        await policyStep();
        for (let index = 0; index < physicsStepsPerPolicy; index += 1) {
          applyControl();
          mujoco.mj_step(model, data);
        }
        simulation.elapsed = data.time;
        simulation.realTimeAccumulator -= actualPolicyDt;

        if (data.qpos[2] < -0.2) {
          reset();
          setNotice("机器人离开场地，已自动重置。");
          break;
        }
      }
    } catch (error) {
      simulation.running = false;
      $("start").textContent = "开始";
      setNotice(error.message, true);
    } finally { pendingStep = false; }
  }
  updateRobot();
  $("clock").textContent = `${simulation.elapsed.toFixed(2)} s`;
  updateTelemetry();
  followRobot(frameDelta);
  view.orbit.update();
  view.renderer.render(view.scene, view.camera);
  requestAnimationFrame(tick);
}

const terrainLabels = { platform: "高台", stairs: "台阶", ramp: "斜坡", stones: "梅花桩", wall: "矮墙" };

function terrainCanvasPosition(event) {
  const canvas = $("terrainMap");
  const bounds = canvas.getBoundingClientRect();
  return {
    x: THREE.MathUtils.clamp(((event.clientX - bounds.left) / bounds.width - .5) * TERRAIN_AREA.x * 2, -TERRAIN_AREA.x, TERRAIN_AREA.x),
    y: THREE.MathUtils.clamp((.5 - (event.clientY - bounds.top) / bounds.height) * TERRAIN_AREA.y * 2, -TERRAIN_AREA.y, TERRAIN_AREA.y),
  };
}

function repaintTerrainMap() {
  const canvas = $("terrainMap");
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  context.fillStyle = terrainState.kind === "flat" ? "#f7efdc" : "#f2e7cd";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#ddcfae";
  context.lineWidth = 1;
  for (let x = 0; x <= width; x += width / 10) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  for (let y = 0; y <= height; y += height / 8) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  const scaleX = width / (TERRAIN_AREA.x * 2);
  const scaleY = height / (TERRAIN_AREA.y * 2);
  const drawFootprint = (x, y, length, breadth, yaw, fill, label, textColor) => {
    context.save();
    context.translate((x / (TERRAIN_AREA.x * 2) + .5) * width, (.5 - y / (TERRAIN_AREA.y * 2)) * height);
    context.rotate(-yaw);
    context.fillStyle = fill;
    context.fillRect(-length / 2 * scaleX, -breadth / 2 * scaleY, length * scaleX, breadth * scaleY);
    context.restore();
    if (label) {
      context.fillStyle = textColor;
      context.font = "10px system-ui";
      context.textAlign = "center";
      context.fillText(label, (x / (TERRAIN_AREA.x * 2) + .5) * width, (.5 - y / (TERRAIN_AREA.y * 2)) * height + 3);
    }
  };
  if (terrainState.kind === "slope") drawFootprint(1.15, 0, 1.3, 1.0, 0, "#e0cda4", "", "#5a4a26");
  if (terrainState.kind === "stairs") drawFootprint(1.03, 0, 7 * 0.16, 0.9, 0, "#e0cda4", "", "#5a4a26");
  terrainState.elements.forEach((element, index) => {
    const footprint = elementFootprint(element);
    const selected = index === terrainState.selected;
    drawFootprint(element.x, element.y, footprint.length, footprint.width, footprint.yaw,
      selected ? "#c9992a" : "#b9a173", String(index + 1), selected ? "#fffdf7" : "#3b3220");
  });
}

function renderTerrainElements() {
  const mount = $("terrainElements");
  mount.replaceChildren(...terrainState.elements.map((element, index) => {
    const row = document.createElement("div");
    row.className = `terrain-element${index === terrainState.selected ? " selected" : ""}`;
    const label = terrainLabels[element.kind] ?? "导入障碍";
    const footprint = elementFootprint(element);
    row.innerHTML = `<span>${index + 1}. ${label} · ${footprint.length.toFixed(2)}×${footprint.width.toFixed(2)} m${element.kind === "ramp" ? ` · ${element.slope.toFixed(0)}%` : ""}</span>`;
    row.onclick = () => { terrainState.selected = index; repaintTerrainEditor(); };
    const remove = document.createElement("button");
    remove.textContent = "删除";
    remove.onclick = (event) => {
      event.stopPropagation();
      terrainState.elements.splice(index, 1);
      terrainState.selected = null;
      repaintTerrainEditor();
    };
    row.append(remove);
    return row;
  }));
}

function repaintTerrainEditor() {
  repaintTerrainMap();
  renderTerrainElements();
  updateTerrainElementEditor();
  renderTerrain();
}

function updateTerrainElementEditor() {
  const element = terrainState.elements[terrainState.selected];
  $("terrainElementEditor").hidden = !element;
  if (!element) return;
  $("elementEditorTitle").textContent = `${terrainLabels[element.kind] ?? "障碍"}参数`;
  document.querySelectorAll("[data-element-for]").forEach((label) => {
    label.hidden = !label.dataset.elementFor.split(" ").includes(element.kind);
  });
  $("elementX").value = element.x.toFixed(2);
  $("elementY").value = element.y.toFixed(2);
  $("elementYaw").value = THREE.MathUtils.radToDeg(element.yaw || 0).toFixed(0);
  const values = {
    elementLength: element.length,
    elementWidth: element.width,
    elementHeight: element.height,
    stepDepth: element.stepDepth,
    stepWidth: element.stepWidth,
    stepHeight: element.stepHeight,
    stepCount: element.stepCount,
    elementSlope: element.slope,
    stoneColumns: element.columns,
    stoneRows: element.rows,
    stoneGapX: element.gapX,
    stoneGapY: element.gapY,
  };
  for (const [id, value] of Object.entries(values)) {
    if (value !== undefined) $(id).value = typeof value === "number" ? Number(value.toFixed(2)) : value;
  }
  if (element.kind === "stones") $("stoneLayout").value = element.layout ?? "aligned";
}

function elementAt(position) {
  for (let index = terrainState.elements.length - 1; index >= 0; index -= 1) {
    const hit = elementBoxes(terrainState.elements[index]).some((definition) => {
      const yaw = definition.yaw || 0;
      const dx = position.x - definition.x;
      const dy = position.y - definition.y;
      const localX = Math.cos(yaw) * dx + Math.sin(yaw) * dy;
      const localY = -Math.sin(yaw) * dx + Math.cos(yaw) * dy;
      return Math.abs(localX) <= definition.sx && Math.abs(localY) <= definition.sy;
    });
    if (hit) return index;
  }
  return terrainState.elements.findIndex((element) => Math.hypot(element.x - position.x, element.y - position.y) < 0.12);
}


function installTerrainMap() {
  const canvas = $("terrainMap");
  let drag = null;
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const position = terrainCanvasPosition(event);
    const index = elementAt(position);
    if (index >= 0) {
      terrainState.selected = index;
      const element = terrainState.elements[index];
      drag = { index, offsetX: element.x - position.x, offsetY: element.y - position.y, pointerId: event.pointerId };
      canvas.setPointerCapture(event.pointerId);
      repaintTerrainEditor();
    } else {
      terrainState.elements.push(createTerrainElement(terrainState.tool, position.x, position.y));
      terrainState.selected = terrainState.elements.length - 1;
      repaintTerrainEditor();
    }
    event.preventDefault();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const position = terrainCanvasPosition(event);
    const element = terrainState.elements[drag.index];
    element.x = THREE.MathUtils.clamp(position.x + drag.offsetX, -TERRAIN_AREA.x, TERRAIN_AREA.x);
    element.y = THREE.MathUtils.clamp(position.y + drag.offsetY, -TERRAIN_AREA.y, TERRAIN_AREA.y);
    repaintTerrainEditor();
    $("terrainHint").textContent = "障碍位置已更新预览；点击“应用到场景”同步物理碰撞。";
    event.preventDefault();
  });
  const end = (event) => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    drag = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
}

function installPanelDrag() {
  document.querySelectorAll(".cmd-panel, .monitor-panel, .terrain-panel").forEach((panel) => {
    const handle = panel.querySelector(".panel-heading");
    if (!handle) return;
    let drag = null;
    handle.addEventListener("pointerdown", (event) => {
      if (event.target.closest("button")) return;
      if (event.button !== 0) return;
      const rect = panel.getBoundingClientRect();
      panel.style.left = `${rect.left}px`;
      panel.style.top = `${rect.top}px`;
      panel.style.right = "auto";
      panel.style.bottom = "auto";
      if (panel.classList.contains("terrain-panel")) panel.style.height = "auto";
      drag = { pointerId: event.pointerId, offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top };
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    handle.addEventListener("pointermove", (event) => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const rect = panel.getBoundingClientRect();
      const maxLeft = window.innerWidth - rect.width - 8;
      const maxTop = window.innerHeight - 80;
      panel.style.left = `${THREE.MathUtils.clamp(event.clientX - drag.offsetX, 8, Math.max(8, maxLeft))}px`;
      panel.style.top = `${THREE.MathUtils.clamp(event.clientY - drag.offsetY, 48, Math.max(48, maxTop))}px`;
      if (panel.classList.contains("terrain-panel")) panel.style.maxHeight = `${window.innerHeight - parseFloat(panel.style.top) - 12}px`;
      event.preventDefault();
    });
    const end = (event) => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      drag = null;
      if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
    };
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  });
}

function importTerrainScene(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const content = String(reader.result);
      const normalize = (raw) => {
        const aliases = { stepping_stones: "stones", high_wall: "wall" };
        const kind = aliases[raw.kind] ?? (["platform", "stairs", "ramp", "stones", "wall"].includes(raw.kind) ? raw.kind : "platform");
        const element = createTerrainElement(kind, Number(raw.x) || 0, Number(raw.y) || 0);
        element.yaw = Number(raw.yaw) || 0;
        const number = (value) => (Number.isFinite(Number(value)) ? Number(value) : undefined);
        const scaleX = number(raw.scaleX ?? raw.scale_x);
        const scaleY = number(raw.scaleY ?? raw.scale_y);
        element.length = number(raw.length) ?? (scaleX !== undefined ? (TERRAIN_ELEMENT_DEFAULTS[kind].length ?? 0.3) * scaleX : element.length);
        element.width = number(raw.width) ?? (scaleY !== undefined ? (TERRAIN_ELEMENT_DEFAULTS[kind].width ?? 0.3) * scaleY : element.width);
        element.height = number(raw.height ?? raw.size?.[2]) ?? element.height;
        for (const key of ["stepDepth", "stepWidth", "stepHeight", "stepCount", "columns", "rows", "gapX", "gapY"]) {
          element[key] = number(raw[key]) ?? element[key];
        }
        if (["aligned", "staggered", "diagonal"].includes(raw.layout)) element.layout = raw.layout;
        element.slope = kind === "ramp"
          ? (number(raw.slope) ?? element.height / element.length * 100)
          : element.slope;
        return element;
      };
      if (file.name.toLowerCase().endsWith(".xml")) {
        const document = new DOMParser().parseFromString(content, "text/xml");
        const imported = [...document.querySelectorAll('geom[type="box"]')].map((geom) => {
          const position = vector(geom.getAttribute("pos") ?? "", 3, [0, 0, 0]);
          const size = vector(geom.getAttribute("size") ?? "", 3, [.3, .3, .1]);
          const euler = vector(geom.getAttribute("euler") ?? "", 3, [0, 0, 0]);
          return normalize({
            kind: size[0] > 0.4 && size[1] < .12 ? "wall" : "platform",
            x: position[0], y: position[1], yaw: euler[2] || 0,
            length: size[0] * 2, width: size[1] * 2, height: size[2] * 2,
          });
        });
        if (!imported.length) throw new Error("未找到可导入的 box 地形；高度场和外部 mesh 需先导出为场景 JSON。");
        terrainState.elements = imported;
      } else {
        const scene = JSON.parse(content);
        const terrain = scene.terrain ?? scene;
        if (terrain.kind && ["flat", "slope", "stairs", "obstacle_mix"].includes(terrain.kind)) terrainState.kind = terrain.kind;
        terrainState.seed = Number(terrain.seed ?? terrainState.seed);
        terrainState.height = Number(terrain.height ?? terrain.obstacle_height ?? terrainState.height);
        terrainState.elements = (scene.elements ?? []).map(normalize);
      }
      terrainState.selected = null;
      $("terrainKind").value = terrainState.kind;
      $("terrainSeed").value = terrainState.seed;
      repaintTerrainEditor();
      setNotice("地形已导入预览；点击“应用到场景”后写入浏览器内 MuJoCo 物理模型。");
    } catch (error) { setNotice(error.message || String(error), true); }
  };
  reader.readAsText(file);
}

function exportTerrainScene() {
  const payload = {
    version: 2,
    name: "opendoge-terrain",
    terrain: { kind: terrainState.kind, seed: terrainState.seed, height: terrainState.height },
    elements: terrainState.elements,
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "opendoge-terrain.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function installJoystick(id, stickKey) {
  const mount = $(id);
  const base = mount.querySelector(".joystick-base");
  const knob = mount.querySelector(".joystick-knob");
  let pointerId = null;

  const setPosition = (event) => {
    const bounds = base.getBoundingClientRect();
    const radius = Math.max(1, bounds.width / 2 - knob.offsetWidth / 2 - 3);
    let x = event.clientX - (bounds.left + bounds.width / 2);
    let y = event.clientY - (bounds.top + bounds.height / 2);
    const length = Math.hypot(x, y);
    if (length > radius) {
      x = x / length * radius;
      y = y / length * radius;
    }
    const state = controllerState.joystick[stickKey];
    state.x = x / radius;
    state.y = y / radius;
    renderJoystickPosition(stickKey);
    recomputeCommand();
  };

  const release = (event) => {
    if (pointerId !== event.pointerId) return;
    pointerId = null;
    controllerState.joystick[stickKey].x = 0;
    controllerState.joystick[stickKey].y = 0;
    renderJoystickPosition(stickKey);
    if (mount.hasPointerCapture(event.pointerId)) mount.releasePointerCapture(event.pointerId);
    recomputeCommand();
  };

  mount.addEventListener("pointerdown", (event) => {
    if (pointerId !== null) return;
    pointerId = event.pointerId;
    mount.setPointerCapture(pointerId);
    setPosition(event);
    event.preventDefault();
  });
  mount.addEventListener("pointermove", (event) => {
    if (pointerId !== event.pointerId) return;
    setPosition(event);
    event.preventDefault();
  });
  mount.addEventListener("pointerup", release);
  mount.addEventListener("pointercancel", release);
}

function resetJoysticks() {
  for (const stick of Object.values(controllerState.joystick)) {
    stick.x = 0;
    stick.y = 0;
  }
  syncJoystickVisuals();
}

function installKeyboardControl() {
  const supportedKeys = new Set(["q", "w", "e", "a", "s", "d"]);
  const isTyping = (target) => ["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName) || target?.isContentEditable;
  window.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (!supportedKeys.has(key) || isTyping(event.target)) return;
    controllerState.keyboardKeys.add(key);
    event.preventDefault();
    recomputeCommand();
  });
  window.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (!supportedKeys.has(key)) return;
    controllerState.keyboardKeys.delete(key);
    event.preventDefault();
    recomputeCommand();
  });
  window.addEventListener("blur", () => {
    if (!controllerState.keyboardKeys.size) return;
    controllerState.keyboardKeys.clear();
    recomputeCommand();
  });
}

for (const [key, config] of Object.entries(policies)) $("policy").add(new Option(config.name, key));
$("policy").onchange = () => loadPolicy($("policy").value).catch((error) => setNotice(error.message, true));
$("start").onclick = () => {
  simulation.running = !simulation.running;
  $("start").textContent = simulation.running ? "暂停" : "继续";
};
$("reset").onclick = () => reset();
$("cmdToggle").onclick = () => {
  const panel = $("cmdPanel");
  panel.hidden = !panel.hidden;
};
$("cmdClose").onclick = () => { $("cmdPanel").hidden = true; };
$("monitorToggle").onclick = () => {
  const panel = $("monitorPanel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) drawActionChart();
};
$("monitorClose").onclick = () => { $("monitorPanel").hidden = true; };
$("terrainToggle").onclick = () => {
  const panel = $("terrainPanel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) repaintTerrainEditor();
};
$("terrainClose").onclick = () => { $("terrainPanel").hidden = true; };
$("terrainKind").onchange = (event) => { terrainState.kind = event.target.value; repaintTerrainEditor(); };
$("terrainSeed").oninput = (event) => { terrainState.seed = Number(event.target.value) || 0; repaintTerrainEditor(); };
function editSelectedTerrain(key, value) {
  const element = terrainState.elements[terrainState.selected];
  if (!element || !Number.isFinite(value)) { updateTerrainElementEditor(); return; }
  if (key === "x") value = THREE.MathUtils.clamp(value, -TERRAIN_AREA.x, TERRAIN_AREA.x);
  if (key === "y") value = THREE.MathUtils.clamp(value, -TERRAIN_AREA.y, TERRAIN_AREA.y);
  if (key === "yaw") value = THREE.MathUtils.degToRad(value);
  if (["length", "width", "height", "stepDepth", "stepWidth", "stepHeight", "gapX", "gapY"].includes(key)) {
    value = THREE.MathUtils.clamp(value, 0.02, 10);
  }
  if (key === "stepCount") value = THREE.MathUtils.clamp(Math.round(value), 1, 20);
  if (key === "columns" || key === "rows") value = THREE.MathUtils.clamp(Math.round(value), 1, 8);
  element[key] = value;
  if (element.kind === "ramp") {
    if (key === "slope" || key === "length") element.height = element.length * element.slope / 100;
    if (key === "height") element.slope = element.height / element.length * 100;
  }
  repaintTerrainEditor();
}

const terrainFieldMap = {
  elementX: "x", elementY: "y", elementYaw: "yaw",
  elementLength: "length", elementWidth: "width", elementHeight: "height",
  stepDepth: "stepDepth", stepWidth: "stepWidth", stepHeight: "stepHeight", stepCount: "stepCount",
  elementSlope: "slope", stoneColumns: "columns", stoneRows: "rows",
  stoneGapX: "gapX", stoneGapY: "gapY",
};
for (const [id, key] of Object.entries(terrainFieldMap)) {
  $(id).onchange = (event) => editSelectedTerrain(key, Number(event.target.value));
}
$("stoneLayout").onchange = (event) => {
  const element = terrainState.elements[terrainState.selected];
  if (!element) return;
  element.layout = event.target.value;
  repaintTerrainEditor();
};
document.querySelectorAll("[data-terrain-tool]").forEach((button) => {
  button.onclick = () => {
    terrainState.tool = button.dataset.terrainTool;
    document.querySelectorAll("[data-terrain-tool]").forEach((item) => item.classList.toggle("active", item === button));
    $("terrainHint").textContent = `当前工具：${terrainLabels[terrainState.tool]}。在场地中点击放置。`;
  };
});
$("terrainClear").onclick = () => { terrainState.elements = []; terrainState.selected = null; repaintTerrainEditor(); };
$("terrainApply").onclick = () => {
  applyTerrain().catch((error) => { setStatus("地形同步失败", "error"); setNotice(error.message || String(error), true); });
};
$("terrainExport").onclick = exportTerrainScene;
$("terrainImport").onchange = (event) => {
  const [file] = event.target.files;
  if (file) importTerrainScene(file);
  event.target.value = "";
};
[["commandSpeedVx", 0], ["commandSpeedVy", 1], ["commandSpeedYaw", 2]].forEach(([id, index]) => {
  $(id).oninput = (event) => {
    controllerState.commandSpeed[index] = Number(event.target.value);
    recomputeCommand();
  };
});
$("commandStop").onclick = () => {
  controllerState.keyboardKeys.clear();
  resetJoysticks();
  recomputeCommand();
};
installJoystick("leftJoystick", "left");
installTerrainMap();
installPanelDrag();
installJoystick("rightJoystick", "right");
installKeyboardControl();

setBootStage("加载机器人模型…", 8);
const modelText = fetch(publicAsset("robot/opendoge.xml")).then((response) => response.text());
const robotTask = modelText
  .then(async (xml) => {
    await buildRobot(xml, parseMeshAssets(xml));
    setBootStage("机器人模型已就绪…", 38);
  });
const engineTask = initializeMujoco().then(() => setBootStage("MuJoCo 物理引擎已就绪…", 68));

Promise.all([robotTask, engineTask])
  .then(() => {
    setBootStage("加载策略网络…", 82);
    return loadPolicy("flat");
  })
  .then(() => {
    $("start").disabled = false;
    $("reset").disabled = false;
    $("cmdToggle").disabled = false;
    $("monitorToggle").disabled = false;
    $("terrainToggle").disabled = false;
    installRobotDrag();
    setStatus("浏览器物理引擎已就绪", "ready");
    setBootStage("场景已就绪", 100);
    tick();
    requestAnimationFrame(revealScene);
  })
  .catch((error) => {
    console.error(error);
    setStatus("仿真器初始化失败", "error");
    setNotice(error.message || String(error), true);
    setBootStage("初始化失败，请检查浏览器控制台", 100);
  });
