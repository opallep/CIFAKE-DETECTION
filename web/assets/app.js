// app.js - CekCitra: unggah gambar, tampilkan versi 32x32 yang dilihat model, kirim ke /api/predict.
// Tanpa server atau model (mis. index.html dibuka langsung), hasil berupa simulasi yang diberi label jelas.
(() => {
  "use strict";

  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const MAX_BYTES = 10 * 1024 * 1024;
  const INPUT = 32;
  const CELLS = 40;
  const SAMPLES = [
    ["real_1", "REAL"], ["fake_1", "FAKE"], ["fake_4", "FAKE"], ["real_4", "REAL"],
    ["real_2", "REAL"], ["fake_2", "FAKE"], ["real_5", "REAL"], ["fake_5", "FAKE"],
    ["fake_3", "FAKE"], ["real_3", "REAL"], ["fake_6", "FAKE"], ["real_6", "REAL"],
  ].map(([f, truth], i) => ({ src: `assets/samples/${f}.jpg`, name: `Contoh ${String(i + 1).padStart(2, "0")}`, truth }));

  const $ = (id) => document.getElementById(id);
  const el = {
    status: $("status"), statusText: $("status-text"),
    input: $("file-input"), drop: $("drop"), dropEmpty: $("drop-empty"), view: $("view"), pxBadge: $("px-badge"),
    viewMode: $("view-mode"), reset: $("btn-reset"), fileLine: $("file-line"), error: $("error"), check: $("btn-check"),
    sim: $("sim-flag"), verdict: $("verdict"), vWord: $("v-word"), vSub: $("v-sub"), meter: $("meter"),
    rReal: $("r-real"), rFake: $("r-fake"), rLevel: $("r-level"), rModel: $("r-model"), rTime: $("r-time"),
    truth: $("truth"), sheet: $("sheet"), logBlock: $("log-block"), log: $("log"),
  };

  let mode = "offline";          // live | sim | offline
  let modelName = null;
  let current = null;            // { img, blob, name, src, truth, bytes, seed, thumb }
  let viewMode = "full";         // full | model
  let paintedSize = null;        // ukuran piksel yang sedang dilukis (null = asli)
  let busy = false;
  const revealed = new Set();
  const log = [];

  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const pct = (x) => `${(x * 100).toFixed(1).replace(".", ",")}%`;
  const fmtBytes = (b) => (b == null ? null : b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`);

  // ---------------- Status server ----------------
  async function detectMode() {
    const forceSim = new URLSearchParams(location.search).has("demo");
    if (location.protocol !== "file:") {
      try {
        const data = await (await fetch("/api/status", { cache: "no-store" })).json();
        modelName = data.model || null;
        mode = data.model_ready && !forceSim ? "live" : "sim";
      } catch { mode = "offline"; }
    }
    el.status.dataset.mode = mode;
    el.statusText.textContent =
      mode === "live" ? `Model aktif: ${modelName}` :
      mode === "sim" ? "Model belum terhubung, hasil disimulasikan" : "Tanpa server, hasil disimulasikan";
  }

  // ---------------- Kanvas ----------------
  function paint(size) {
    if (!current) return;
    const c = el.view;
    const box = c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(box.width * dpr);
    c.height = Math.round(box.height * dpr);
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const { img } = current;
    const pad = 0.86;

    if (size == null) {
      const s = Math.min(c.width / img.naturalWidth, c.height / img.naturalHeight) * pad;
      const w = img.naturalWidth * s, h = img.naturalHeight * s;
      ctx.imageSmoothingEnabled = img.naturalWidth > 128;
      ctx.drawImage(img, (c.width - w) / 2, (c.height - h) / 2, w, h);
      el.pxBadge.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
    } else {
      // Sama seperti Resize((32, 32)) di pipeline: gambar dipaksa persegi.
      const off = document.createElement("canvas");
      off.width = off.height = size;
      const octx = off.getContext("2d");
      octx.imageSmoothingQuality = "high";
      octx.drawImage(img, 0, 0, size, size);
      const side = Math.min(c.width, c.height) * pad;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, (c.width - side) / 2, (c.height - side) / 2, side, side);
      el.pxBadge.textContent = `${size} × ${size} px`;
    }
    paintedSize = size;
  }

  function setViewMode(m) {
    viewMode = m;
    el.viewMode.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === m)));
    paint(m === "model" ? INPUT : null);
  }

  async function pixelate() {
    viewMode = "model";
    el.viewMode.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === "model")));
    if (reduceMotion) return paint(INPUT);
    for (const s of [160, 96, 64, 48, INPUT]) { paint(s); await sleep(130); }
  }

  function modelPixels(img) {
    const off = document.createElement("canvas");
    off.width = off.height = INPUT;
    const ctx = off.getContext("2d");
    ctx.drawImage(img, 0, 0, INPUT, INPUT);
    try {
      const px = ctx.getImageData(0, 0, INPUT, INPUT).data;
      let seed = 0;
      for (let i = 0; i < px.length; i += 5) seed = (Math.imul(seed, 31) + px[i]) >>> 0;
      return { seed, thumb: off.toDataURL("image/png") };
    } catch {
      return { seed: 0, thumb: null };      // kanvas tercemar saat dibuka via file://
    }
  }

  // ---------------- Memuat gambar ----------------
  function loadImg(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("File ini tidak bisa dibaca sebagai gambar. Coba file lain."));
      img.src = src;
    });
  }

  async function useImage(item, keepResult = false) {
    showError("");
    let img;
    try { img = await loadImg(item.src); } catch (e) { return showError(e.message); }
    const { seed, thumb } = modelPixels(img);
    current = { ...item, img, seed: seed || hashString(item.name), thumb: thumb || item.src };

    el.dropEmpty.hidden = true;
    el.view.hidden = false;
    el.pxBadge.hidden = false;
    el.drop.classList.add("has-image");
    el.viewMode.hidden = false;
    el.reset.hidden = false;
    el.check.disabled = false;

    const parts = [item.name, `${img.naturalWidth}×${img.naturalHeight}`, fmtBytes(item.bytes)].filter(Boolean);
    el.fileLine.textContent = parts.join("  ·  ");
    el.fileLine.hidden = false;

    setViewMode("full");
    if (!keepResult) idleReadout();
    markFrame(item.src);
  }

  function useFile(file) {
    if (!file) return;
    if (!ALLOWED_TYPES.includes(file.type)) return showError("Format ini tidak didukung. Gunakan gambar JPG, PNG, atau WEBP.");
    if (file.size > MAX_BYTES) return showError(`Ukuran file ${fmtBytes(file.size)}, melebihi batas 10 MB. Perkecil gambarnya lalu coba lagi.`);
    useImage({ blob: file, name: file.name || "gambar-tempel.png", src: URL.createObjectURL(file), bytes: file.size, truth: null });
  }

  async function useSample(s) {
    let blob = null;
    if (location.protocol !== "file:") {
      try { blob = await (await fetch(s.src)).blob(); } catch { /* tetap bisa ditampilkan */ }
    }
    await useImage({ blob, name: s.name, src: s.src, bytes: blob ? blob.size : null, truth: s.truth });
    el.drop.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
  }

  function resetAll() {
    current = null;
    el.input.value = "";
    el.dropEmpty.hidden = false;
    el.view.hidden = true;
    el.pxBadge.hidden = true;
    el.drop.classList.remove("has-image");
    el.viewMode.hidden = true;
    el.reset.hidden = true;
    el.fileLine.hidden = true;
    el.check.disabled = true;
    showError("");
    idleReadout();
    markFrame(null);
  }

  function showError(msg) {
    el.error.textContent = msg || "";
    el.error.hidden = !msg;
  }

  // ---------------- Prediksi ----------------
  function hashString(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619) >>> 0;
    return h;
  }

  function simulate(seed) {
    const r = ((Math.imul(seed ^ 0x9e3779b9, 2654435761) >>> 0) % 10000) / 10000;
    const fake = Math.min(0.995, Math.max(0.005, r));
    return {
      label: fake >= 0.5 ? "FAKE" : "REAL", confidence: Math.max(fake, 1 - fake),
      probabilities: { FAKE: fake, REAL: 1 - fake }, model: "Simulasi", inference_ms: 6 + (seed % 900) / 100, simulated: true,
    };
  }

  async function predict() {
    if (mode !== "live") { await sleep(500); return simulate(current.seed); }
    if (!current.blob) throw new Error("Gambar ini tidak bisa dikirim ke server. Unggah gambar dari perangkat Anda.");
    const body = new FormData();
    body.append("image", current.blob, current.name);
    const res = await fetch("/api/predict", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Server menolak permintaan (HTTP ${res.status}).`);
    return data;
  }

  async function runCheck() {
    if (!current || busy) return;
    busy = true;
    showError("");
    el.check.disabled = true;
    el.check.classList.add("is-busy");
    busyReadout();
    try {
      const [result] = await Promise.all([predict(), pixelate()]);
      const item = current;
      renderReadout(result, item);
      if (item.truth) { revealed.add(item.src); renderSheet(); markFrame(item.src); }
      addLog(result, item);
    } catch (e) {
      idleReadout();
      showError(e.message || "Server tidak bisa dihubungi. Pastikan python web/app.py sedang berjalan.");
    } finally {
      busy = false;
      el.check.disabled = !current;
      el.check.classList.remove("is-busy");
    }
  }

  // ---------------- Readout ----------------
  function buildMeter() {
    el.meter.innerHTML = "<i></i>".repeat(CELLS);
  }

  function setMeter(pReal) {
    const n = pReal == null ? -1 : Math.round(pReal * CELLS);
    [...el.meter.children].forEach((c, i) => { c.className = n < 0 ? "" : i < n ? "r" : "f"; });
  }

  function idleReadout() {
    el.verdict.dataset.state = "idle";
    el.vWord.textContent = "?";
    el.vSub.textContent = current ? "Tekan Periksa gambar untuk memulai." : "Belum ada gambar yang diperiksa.";
    setMeter(null);
    [el.rReal, el.rFake, el.rLevel, el.rModel, el.rTime].forEach((d) => (d.textContent = "—"));
    el.truth.hidden = true;
    el.sim.hidden = true;
  }

  function busyReadout() {
    el.verdict.dataset.state = "busy";
    el.vWord.textContent = "···";
    el.vSub.textContent = "Memperkecil ke 32×32 lalu menganalisis…";
    setMeter(null);
  }

  function levelOf(c) {
    if (c >= 0.9) return ["Tinggi", "Model sangat yakin."];
    if (c >= 0.7) return ["Sedang", "Cukup meyakinkan, tapi masih bisa keliru."];
    return ["Rendah", "Kedua kemungkinan hampir seimbang, jadi anggap belum pasti."];
  }

  function renderReadout(r, item) {
    const real = r.label === "REAL";
    const [lv, note] = levelOf(r.confidence);
    el.verdict.dataset.state = real ? "real" : "fake";
    el.vWord.textContent = real ? "ASLI" : "AI";
    el.vSub.textContent = `${real ? "Kemungkinan foto asli" : "Kemungkinan dibuat AI"}, ${pct(r.confidence)}. ${note}`;
    setMeter(r.probabilities.REAL);
    el.rReal.textContent = pct(r.probabilities.REAL);
    el.rFake.textContent = pct(r.probabilities.FAKE);
    el.rLevel.textContent = lv;
    el.rModel.textContent = r.model;
    el.rTime.textContent = `${Number(r.inference_ms).toFixed(1).replace(".", ",")} ms`;
    el.sim.hidden = !r.simulated;

    if (item.truth) {
      const ok = item.truth === r.label;
      el.truth.innerHTML = `Label sebenarnya: <b>${item.truth === "REAL" ? "Asli" : "AI"}</b>. ` +
        `Jawaban model <b class="${ok ? "ok" : "bad"}">${ok ? "benar" : "keliru"}</b>.`;
      el.truth.hidden = false;
    } else {
      el.truth.hidden = true;
    }
  }

  // ---------------- Lembar contoh & riwayat ----------------
  function renderSheet() {
    el.sheet.innerHTML = "";
    SAMPLES.forEach((s) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "frame";
      b.dataset.src = s.src;
      const shown = revealed.has(s.src);
      b.setAttribute("aria-label", `${s.name}${shown ? `, label ${s.truth === "REAL" ? "asli" : "AI"}` : ""}`);
      b.innerHTML = `<img alt="" src="${s.src}" width="32" height="32">` +
        `<span class="frame-meta"><span>${s.name.slice(-2)}</span>` +
        `<span class="frame-ans ${shown ? s.truth.toLowerCase() : ""}">${shown ? (s.truth === "REAL" ? "ASLI" : "AI") : "?"}</span></span>`;
      b.addEventListener("click", () => useSample(s));
      li.appendChild(b);
      el.sheet.appendChild(li);
    });
  }

  function markFrame(src) {
    el.sheet.querySelectorAll(".frame").forEach((f) => f.classList.toggle("is-active", f.dataset.src === src));
  }

  function addLog(r, item) {
    log.unshift({ r, item });
    if (log.length > 8) log.pop();
    el.log.innerHTML = "";
    log.forEach(({ r: lr, item: li }) => {
      const row = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "log-item";
      b.innerHTML = `<img alt="" src="${li.thumb}"><span class="log-name"></span>` +
        `<span class="log-res ${lr.label === "REAL" ? "real" : "fake"}">${lr.label === "REAL" ? "ASLI" : "AI"} ${pct(lr.confidence)}</span>`;
      b.querySelector(".log-name").textContent = li.name;
      b.addEventListener("click", async () => {
        await useImage(li, true);
        renderReadout(lr, li);
        setViewMode("model");
        el.drop.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
      });
      row.appendChild(b);
      el.log.appendChild(row);
    });
    el.logBlock.hidden = false;
  }

  // ---------------- Event ----------------
  el.input.addEventListener("change", () => useFile(el.input.files[0]));
  el.check.addEventListener("click", runCheck);
  el.reset.addEventListener("click", resetAll);
  el.viewMode.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (b && !busy) setViewMode(b.dataset.mode);
  });

  el.drop.addEventListener("click", (e) => { if (current) e.preventDefault(); });
  el.drop.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !current) { e.preventDefault(); el.input.click(); }
  });
  ["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add("is-drag"); }));
  ["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("is-drag"); }));
  el.drop.addEventListener("drop", (e) => useFile(e.dataTransfer.files[0]));

  document.addEventListener("paste", (e) => {
    const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (it) useFile(it.getAsFile());
  });
  document.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName || "";
    if (/INPUT|TEXTAREA|SELECT|BUTTON/.test(tag)) return;
    if (e.key === "Enter" && current) { e.preventDefault(); runCheck(); }
    if (e.key === "Escape" && current && !busy) resetAll();
  });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => current && paint(paintedSize), 100);
  });

  buildMeter();
  renderSheet();
  idleReadout();
  detectMode();
})();
