/* Sémio : active l'isolation cross-origin sur GitHub Pages (COOP/COEP) pour que le modèle
   puisse utiliser plusieurs cœurs du processeur. Inspiré de coi-serviceworker (MIT). */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (e) => {
    const r = e.request;
    if (r.mode !== "navigate") return;
    e.respondWith(fetch(r).then((res) => {
      if (res.status === 0) return res;
      const h = new Headers(res.headers);
      h.set("Cross-Origin-Embedder-Policy", "credentialless");
      h.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
    }));
  });
} else {
  (() => {
    if (window.crossOriginIsolated || !window.isSecureContext || !navigator.serviceWorker) return;
    let reloaded = false;
    try { reloaded = sessionStorage.getItem("coi-reloaded") === "1"; } catch (e) {}
    navigator.serviceWorker.register(document.currentScript.src).then((reg) => {
      const reload = () => { if (reloaded) return; try { sessionStorage.setItem("coi-reloaded", "1"); } catch (e) {} location.reload(); };
      if (reg.active && !navigator.serviceWorker.controller) reload();
      else if (reg.installing) reg.installing.addEventListener("statechange", function () { if (this.state === "activated") reload(); });
    }).catch(() => {});
  })();
}
