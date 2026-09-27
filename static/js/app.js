(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    lang: "en",
    movement: "vertical_jump",
    ws: null,
    stream: null,
    looping: false,
    lastReport: null,
    lastTs: 0,
    frames: 0,
    fps: 0,
    voice: true,
    lastCueAt: 0,
  };

  const POSE_EDGES = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [27, 31],
    [24, 26], [26, 28], [28, 32],
  ];

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $("tab-" + btn.dataset.tab).classList.add("active");
    });
  });

  $("movementSelect").addEventListener("change", (e) => {
    state.movement = e.target.value;
    if (state.ws && state.ws.readyState === 1) {
      state.ws.send(JSON.stringify({ type: "config", movement: state.movement, language: state.lang }));
    }
  });

  $("langToggle").addEventListener("click", () => {
    state.lang = state.lang === "en" ? "hi" : "en";
    $("langToggle").textContent = state.lang === "en" ? "EN | हिं" : "हिं | EN";
    renderReport(state.lastReport);
    if (state.ws && state.ws.readyState === 1) {
      state.ws.send(JSON.stringify({ type: "config", movement: state.movement, language: state.lang }));
    }
  });

  $("voiceToggle").addEventListener("change", (e) => { state.voice = e.target.checked; });

  const getApiBase = () => (localStorage.getItem("kheldrishti_backend") || "").replace(/\/$/, "");
  const getWsUrl = () => {
    const custom = getApiBase();
    if (custom) {
      try {
        const u = new URL(custom);
        const wsProto = u.protocol === "https:" ? "wss" : "ws";
        return `${wsProto}://${u.host}/ws/live-stream`;
      } catch (e) {
        return `ws://${custom.replace(/^https?:\/\//, "")}/ws/live-stream`;
      }
    }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}/ws/live-stream`;
  };

  function refreshHealth() {
    fetch(`${getApiBase()}/api/health`).then((r) => r.json()).then((h) => {
      const mode = h.environment === "vercel-preview" ? "Vercel Preview" : (h.mediapipe ? "MediaPipe" : "stub");
      $("healthPill").textContent = `${h.status} • ${mode} • ${h.device}`;
    }).catch(() => {
      $("healthPill").textContent = getApiBase() ? "backend offline" : "offline";
    });
  }
  refreshHealth();

  if ($("backendBtn")) {
    $("backendBtn").onclick = () => {
      const current = localStorage.getItem("kheldrishti_backend") || "";
      const val = prompt(
        "Enter your KhelDrishti Python Backend URL:\n(Leave empty to use Vercel Serverless / current host)\n\nExamples:\n- Render/Railway: https://your-backend.onrender.com\n- Local: http://localhost:8000",
        current
      );
      if (val !== null) {
        if (val.trim()) {
          localStorage.setItem("kheldrishti_backend", val.trim());
        } else {
          localStorage.removeItem("kheldrishti_backend");
        }
        refreshHealth();
        loadMovements();
        loadSamples();
      }
    };
  }

  function loadMovements() {
    fetch(`${getApiBase()}/api/movements`).then((r) => r.json()).then((data) => {
      const grid = $("benchGrid");
      grid.innerHTML = "";
      Object.entries(data.movements || {}).forEach(([key, m]) => {
        const art = document.createElement("article");
        art.className = "glass";
        art.innerHTML = `<h3>${m.label}</h3>` + Object.entries(m.benchmarks || {})
          .map(([k, v]) => `<p><strong>${k.replaceAll("_", " ")}</strong><br/>${v}</p>`).join("");
        grid.appendChild(art);
      });
    }).catch(console.error);
  }
  loadMovements();

  function loadSamples() {
    fetch(`${getApiBase()}/api/sample-videos`).then((r) => r.json()).then((data) => {
      const el = $("sampleCarousel");
      el.innerHTML = "";
      (data.samples || []).forEach((s) => {
        const b = document.createElement("button");
        b.className = "chip";
        b.textContent = s.name;
        b.onclick = () => {
          $("preview").src = s.url;
          state.sampleId = s.id;
        };
        el.appendChild(b);
      });
      if (data.samples && data.samples[0]) state.sampleId = data.samples[0].id;
    }).catch(console.error);
  }
  loadSamples();

  function speak(text) {
    if (!state.voice || !text || !window.speechSynthesis) return;
    const now = Date.now();
    if (now - state.lastCueAt < 2500) return;
    state.lastCueAt = now;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = state.lang === "hi" ? "hi-IN" : "en-IN";
    u.rate = 1.05;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }

  function drawSkeleton(ctx, landmarks, w, h, colour) {
    ctx.clearRect(0, 0, w, h);
    if (!landmarks || !landmarks.length) return;
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.lineWidth = 3;
    const pt = (i) => landmarks[i] && [landmarks[i].x * w, landmarks[i].y * h];
    POSE_EDGES.forEach(([a, b]) => {
      const pa = pt(a), pb = pt(b);
      if (!pa || !pb) return;
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.stroke();
    });
    landmarks.forEach((lm) => {
      if (!lm || lm.v < 0.25) return;
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function connectWs() {
    if (state.ws && state.ws.readyState <= 1) return state.ws;
    const wsUrl = getWsUrl();
    let ws;
    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      $("liveCue").textContent = "Could not connect to WebSocket at " + wsUrl + ". Click ⚙️ Backend to configure.";
      return null;
    }
    state.ws = ws;
    ws.onopen = () => ws.send(JSON.stringify({ type: "config", movement: state.movement, language: state.lang }));
    ws.onerror = () => {
      $("liveCue").textContent = "WebSocket connection to " + wsUrl + " disconnected or failed. For live streaming, ensure your Python backend is running and click ⚙️ Backend.";
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type !== "telemetry") return;
      const canvas = $("overlay");
      const video = $("cam");
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
      const colour = msg.risk_colour === "red" ? "#ff3366" : msg.risk_colour === "amber" ? "#ffb800" : "#00ff87";
      drawSkeleton(canvas.getContext("2d"), msg.landmarks, canvas.width, canvas.height, colour);
      $("lkLive").textContent = msg.left_knee != null ? msg.left_knee.toFixed(0) + "°" : "—";
      $("rkLive").textContent = msg.right_knee != null ? msg.right_knee.toFixed(0) + "°" : "—";
      $("lvLive").textContent = (msg.left_valgus || 0).toFixed(1) + "°";
      $("rvLive").textContent = (msg.right_valgus || 0).toFixed(1) + "°";
      $("trLive").textContent = (msg.torso_lean || 0).toFixed(1) + "°";
      const badge = $("valgusBadge");
      badge.textContent = "Valgus: " + (msg.risk || "Safe");
      badge.className = "badge " + (msg.risk_colour === "red" ? "danger" : msg.risk_colour === "amber" ? "warn" : "safe");
      if (msg.voice_cue) {
        $("liveCue").textContent = msg.voice_cue;
        speak(msg.voice_cue);
      }
      state.frames += 1;
      const t = performance.now();
      if (t - state.lastTs > 1000) {
        state.fps = state.frames;
        state.frames = 0;
        state.lastTs = t;
        $("fpsValue").textContent = state.fps;
      }
    };
    return ws;
  }

  async function loopSend() {
    const video = $("cam");
    const tmp = document.createElement("canvas");
    while (state.looping) {
      if (state.ws && state.ws.readyState === 1 && video.readyState >= 2) {
        tmp.width = 320;
        tmp.height = 240;
        tmp.getContext("2d").drawImage(video, 0, 0, 320, 240);
        const image = tmp.toDataURL("image/jpeg", 0.55);
        state.ws.send(JSON.stringify({ image }));
      }
      await new Promise((r) => setTimeout(r, 90));
    }
  }

  $("startCam").onclick = async () => {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      $("cam").srcObject = state.stream;
      await $("cam").play();
      connectWs();
      state.looping = true;
      loopSend();
    } catch (err) {
      alert("Camera access error: " + err.message);
    }
  };
  $("stopCam").onclick = () => {
    state.looping = false;
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    if (state.ws) state.ws.close();
  };

  const drop = $("dropzone");
  drop.addEventListener("dragover", (e) => e.preventDefault());
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
  });
  $("fileInput").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
  });

  async function uploadFile(file) {
    $("preview").src = URL.createObjectURL(file);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("movement", state.movement);
    fd.append("language", state.lang);
    $("progressBar").style.width = "15%";
    try {
      const res = await fetch(`${getApiBase()}/api/analyze-video`, { method: "POST", body: fd });
      $("progressBar").style.width = "100%";
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || data.error || `Upload error (${res.status})`);
      }
      applyReport(data);
    } catch (err) {
      $("progressBar").style.width = "0%";
      alert("Analysis error: " + err.message + "\n\nTip: Video analysis requires the Python backend. If running on Vercel, connect your backend URL via ⚙️ Backend.");
    }
  }

  $("runSample").onclick = async (ev) => {
    ev.preventDefault();
    try {
      $("progressBar").style.width = "20%";
      const fd = new FormData();
      fd.append("sample_id", state.sampleId || "vertical_jump_cmj");
      fd.append("movement", state.movement);
      fd.append("language", state.lang);
      const res = await fetch(`${getApiBase()}/api/analyze-sample`, { method: "POST", body: fd });
      $("progressBar").style.width = "100%";
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || errJson.detail || ("Analysis failed: " + res.status));
      }
      const data = await res.json();
      applyReport(data);
    } catch (err) {
      $("progressBar").style.width = "0%";
      $("keyframeList").textContent = String(err);
    }
  };

  function applyReport(data) {
    state.lastReport = data;
    if (data.annotated_video_url) $("preview").src = data.annotated_video_url;
    const kf = $("keyframeList");
    kf.innerHTML = "";
    Object.entries(data.keyframes || {}).forEach(([i, label]) => {
      const d = document.createElement("div");
      d.textContent = `Frame ${i}: ${label}`;
      kf.appendChild(d);
    });
    renderReport(data);
    document.querySelector('[data-tab="report"]').click();
  }

  function renderReport(data) {
    if (!data) return;
    const c = data.coaching || {};
    const score = c.form_score ?? data.analysis?.form_score ?? 0;
    $("scoreLabel").textContent = Math.round(score);
    const risk = c.risk || {};
    const rl = $("riskLabel");
    rl.textContent = (state.lang === "hi" ? "जोखिम: " : "Risk: ") + (risk.level || "—");
    rl.className = "badge " + (risk.colour === "red" ? "danger" : risk.colour === "amber" ? "warn" : "safe");
    $("summaryText").textContent = state.lang === "hi" ? (c.summary_hi || "") : (c.summary_en || "");
    const tips = $("tipsList");
    tips.innerHTML = "";
    (c.tips || []).forEach((t) => {
      const li = document.createElement("li");
      li.textContent = (state.lang === "hi" ? t.hi : t.en) + (t.drill_en ? " → " + (state.lang === "hi" ? t.drill_hi : t.drill_en) : "");
      tips.appendChild(li);
    });
    const a = data.analysis || {};
    const asym = Number(a.asymmetry_index ?? a.symmetry_index ?? a.landing_asymmetry ?? 0);
    $("symmetryText").textContent = `Asymmetry index: ${asym.toFixed(1)}%`;
    drawGauge($("scoreGauge"), score, risk.colour || "green");
    drawSeries($("angleChart"), a.series || []);
  }

  function drawGauge(canvas, score, colour) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#1f2937";
    ctx.lineWidth = 16;
    ctx.beginPath();
    ctx.arc(w / 2, h - 10, 90, Math.PI, 2 * Math.PI);
    ctx.stroke();
    ctx.strokeStyle = colour === "red" ? "#ff3366" : colour === "amber" ? "#ffb800" : "#00ff87";
    ctx.beginPath();
    ctx.arc(w / 2, h - 10, 90, Math.PI, Math.PI + Math.PI * (score / 100));
    ctx.stroke();
  }

  function drawSeries(canvas, series) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#0b0f19";
    ctx.fillRect(0, 0, w, h);
    if (!series.length) return;
    const ys = series.map((s) => s.lk ?? 90);
    const min = Math.min(...ys, 40), max = Math.max(...ys, 180);
    ctx.strokeStyle = "#00f5ff";
    ctx.beginPath();
    ys.forEach((y, i) => {
      const x = (i / (ys.length - 1)) * (w - 20) + 10;
      const py = h - 10 - ((y - min) / (max - min + 1e-6)) * (h - 20);
      i ? ctx.lineTo(x, py) : ctx.moveTo(x, py);
    });
    ctx.stroke();
    ctx.strokeStyle = "#00ff87";
    ctx.beginPath();
    series.forEach((s, i) => {
      const y = s.rk ?? s.lk ?? 90;
      const x = (i / (series.length - 1)) * (w - 20) + 10;
      const py = h - 10 - ((y - min) / (max - min + 1e-6)) * (h - 20);
      i ? ctx.lineTo(x, py) : ctx.moveTo(x, py);
    });
    ctx.stroke();
  }

  $("exportBtn").onclick = () => {
    const data = state.lastReport;
    if (!data) return;
    const blob = new Blob([JSON.stringify({
      athlete: "KhelDrishti athlete card",
      when: new Date().toISOString(),
      movement: data.movement,
      coaching: data.coaching,
      analysis: { ...data.analysis, series: undefined },
    }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "kheldrishti-athlete-card.json";
    a.click();
  };
})();
