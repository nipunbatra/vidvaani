import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const read = (path) => readFile(new URL(path, `file://${root}`), "utf8");
const [html, main, gemini, pkg, vite] = await Promise.all([
  read("index.html"),
  read("src/main.js"),
  read("src/gemini.js"),
  read("package.json"),
  read("vite.config.js"),
]);

const combined = [html, main, gemini].join("\n");

assert.match(html, /Content-Security-Policy/);
assert.match(html, /default-src 'self'/);
assert.match(html, /object-src 'none'/);
assert.match(html, /base-uri 'self'/);
assert.doesNotMatch(html, /'unsafe-inline'/);
assert.doesNotMatch(html, /'unsafe-eval'/);
assert.doesNotMatch(combined, /localStorage|sessionStorage|indexedDB/);
assert.doesNotMatch(combined, /innerHTML|outerHTML|insertAdjacentHTML/);
assert.match(gemini, /"x-goog-api-key": apiKey/);
assert.match(gemini, /credentials: "omit"/);
assert.match(gemini, /cache: "no-store"/);
assert.match(gemini, /referrerPolicy: "no-referrer"/);
assert.match(main, /if \(usedGemini\) forgetKey\(\)/);
assert.match(vite, /sourcemap: false/);
assert.match(vite, /base: "\.\/"/);

const manifest = JSON.parse(pkg);
for (const [name, version] of Object.entries({ ...manifest.dependencies, ...manifest.devDependencies })) {
  assert.match(version, /^\d/, `${name} must use an exact version, received ${version}`);
}

console.log("Browser lab security invariants: passed");
