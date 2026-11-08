import { defineConfig } from "vite";

// Served from the repository root of OpenDogeRobotics.github.io, but relative
// output keeps the build portable if the site ever moves to a sub-path.
export default defineConfig({ base: "./" });
