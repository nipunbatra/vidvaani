# Browser Lab security model

## Protected data

- selected lecture media and extracted audio;
- English transcript and Hindi translation;
- generated Hindi speech and assembled media;
- a user-entered Gemini API key;
- model usage and cost metadata.

## Trust boundaries

The GitHub Pages origin serves all executable code and WebAssembly. Hugging Face hosts model weights. Google receives only the text for phases explicitly configured to use Gemini and the API key needed for those calls. GitHub Pages, VidVaani, and Hugging Face do not receive the selected media or generated project data from application code.

## Controls in this build

- A restrictive content-security-policy meta tag limits scripts, workers, media, and network destinations.
- There are no remotely loaded scripts, fonts, iframes, analytics, cookies, or service workers.
- All model/user output is inserted with DOM text APIs; no model text is evaluated or assigned through `innerHTML`.
- API keys are password inputs held only in memory, are not included in errors, and are erased after each run and on page unload.
- Gemini calls omit credentials and referrers and disable HTTP caching.
- Dependencies are pinned exactly, the production build omits source maps, and `npm audit` is part of release verification.
- The 120-second input limit reduces denial-of-service risk from memory exhaustion, although a maliciously malformed media file can still stress FFmpeg.wasm.

## GitHub Pages limitation

GitHub Pages does not provide repository-level control over every HTTP response security header. This build therefore expresses CSP and referrer policy in HTML metadata. A meta CSP cannot set all header-only directives, and it starts applying only after the browser parses the tag. The CSP is intentionally the first meaningful content in `<head>`, but a controlled host with response headers is stronger for a production deployment.

Cross-origin isolation is also unavailable as a repository setting on GitHub Pages. The current single-threaded FFmpeg configuration avoids depending on `SharedArrayBuffer`; it may be slower than a cross-origin-isolated deployment.

## Accepted expert-mode risk

Google advises against exposing long-lived production API keys in client-side applications. Direct BYOK is retained here at the project owner's request for a static research demonstration. The interface communicates the risk and minimizes persistence, but it cannot protect a key from a compromised browser, malicious extension, screen capture, developer tools, or code served by a compromised hosting account.

Use a disposable key restricted to the required Gemini APIs, monitor its quota, and revoke it after the demo. Do not use an institutional or production key.

## Out of scope

- protection from a compromised browser, operating system, extension, GitHub account, or dependency publisher;
- DRM or prevention of local copying;
- guarantees that a remote model provider will not retain submitted text under its own service policy;
- suitability of the MMS-TTS model licence for commercial use;
- speaker voice cloning. This browser lab uses only stock voices.

## Reporting

Do not file a public issue containing an API key, private lecture, transcript, or generated media. Revoke an exposed key first, then report the minimal reproducible technical details without sensitive data.
