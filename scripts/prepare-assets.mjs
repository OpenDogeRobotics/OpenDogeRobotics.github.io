// Refresh the committed simulator assets from a local OpenDoge_train checkout.
//
// The generated files (public/robot, public/policies) are committed to this
// repository so GitHub Pages can build without the training repository.  When
// OpenDoge_train is not present, the committed copies are used as-is.

import { access, copyFile, cp, mkdir, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const site = path.resolve(here, "..");
const train = process.env.OPENDOGE_TRAIN
  ? path.resolve(process.env.OPENDOGE_TRAIN)
  : path.resolve(site, "../OpenDoge_train");

const robotSource = path.join(train, "src/assets/robots/opendoge/xmls");
const policySource = path.join(train, "src/baseline/opendoge");

const exists = async (target) => {
  try {
    await access(target);
    return true;
  } catch {
    return false;
  }
};

if (!(await exists(robotSource))) {
  console.log("[prepare-assets] OpenDoge_train not found; using committed assets in public/.");
  process.exit(0);
}

const robotTarget = path.join(site, "public/robot");
await rm(robotTarget, { recursive: true, force: true });
await mkdir(robotTarget, { recursive: true });
await cp(robotSource, robotTarget, { recursive: true });

const policies = [
  ["flat/policy.onnx", "opendoge-flat.onnx"],
  ["rough/policy.onnx", "opendoge-rough.onnx"],
  ["skills/policy.onnx", "opendoge-skills.onnx"],
  ["getup/policy.onnx", "opendoge-getup.onnx"],
];
await mkdir(path.join(site, "public/policies"), { recursive: true });
for (const [source, target] of policies) {
  await copyFile(path.join(policySource, source), path.join(site, "public/policies", target));
}

console.log("[prepare-assets] synced robot model and policies from", train);
