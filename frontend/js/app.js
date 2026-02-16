// Shield AI - Content Protection Platform
"use strict";

const API = "";
let state = {
    videoId: null,
    jobId: null,
    analysis: null,
    ws: null,
    processStart: null,
};

// ── DOM ──
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const show = (el) => el.classList.remove("hidden");
const hide = (el) => el.classList.add("hidden");

function switchScreen(id) {
    $$(".screen").forEach(s => s.classList.remove("screen--active"));
    const target = $(`#screen-${id}`);
    target.classList.add("screen--active");
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// ══════════════════════════════════════════
//  SCREEN 1: UPLOAD
// ══════════════════════════════════════════
const dropZone = $("#drop-zone");
const fileInput = $("#file-input");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
    if (!file.type.startsWith("video/")) {
        alert("Please select a video file.");
        return;
    }

    // Show upload progress
    show($("#upload-progress"));
    const fillEl = $("#upload-fill");
    const textEl = $("#upload-text");
    fillEl.style.width = "0%";
    textEl.textContent = `Uploading ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)...`;

    // Simulate progress during upload
    let fakeProgress = 0;
    const progressInterval = setInterval(() => {
        fakeProgress = Math.min(fakeProgress + Math.random() * 15, 90);
        fillEl.style.width = fakeProgress + "%";
    }, 200);

    try {
        const form = new FormData();
        form.append("file", file);
        const resp = await fetch(`${API}/api/upload`, { method: "POST", body: form });
        const data = await resp.json();
        state.videoId = data.video_id;

        clearInterval(progressInterval);
        fillEl.style.width = "100%";
        textEl.textContent = "Upload complete. Analyzing...";

        // Auto-navigate to diagnosis
        setTimeout(() => startAnalysis(), 500);
    } catch (err) {
        clearInterval(progressInterval);
        textEl.textContent = "Upload failed. Please try again.";
        fillEl.style.width = "0%";
    }
}

// ══════════════════════════════════════════
//  SCREEN 2: DIAGNOSIS
// ══════════════════════════════════════════
async function startAnalysis() {
    switchScreen("diagnosis");
    show($("#scan-overlay"));
    hide($("#diagnosis-results"));

    try {
        const resp = await fetch(`${API}/api/analyze/${state.videoId}`, { method: "POST" });
        const data = await resp.json();
        state.analysis = data;

        hide($("#scan-overlay"));
        show($("#diagnosis-results"));
        renderDiagnosis(data);
    } catch (err) {
        hide($("#scan-overlay"));
        show($("#diagnosis-results"));
        $("#risk-value").textContent = "ANALYSIS FAILED";
    }
}

function renderDiagnosis(data) {
    const info = data.video_info;
    const duration = info.duration;
    const mins = Math.floor(duration / 60);
    const secs = Math.floor(duration % 60);
    $("#video-meta").textContent = `${info.width}x${info.height} \u00b7 ${info.fps.toFixed(0)}fps \u00b7 ${mins}:${secs.toString().padStart(2,"0")} \u00b7 ${info.frame_count} frames`;

    // Risk badge
    const badge = $("#risk-badge");
    badge.dataset.level = data.risk_level;
    $("#risk-value").textContent = data.risk_level.toUpperCase();

    // Score cards
    const dfScore = Math.round(data.max_deepfake_score * 100);
    const nsfwScore = Math.round(data.max_nsfw_score * 100);
    // Compute safe score from frame results
    let maxSafe = 0;
    if (data.frame_results) {
        data.frame_results.forEach(fr => {
            const s = fr.scores?.safe || 0;
            if (s > maxSafe) maxSafe = s;
        });
    }
    const safeScore = Math.round(maxSafe * 100);

    animateScore("score-deepfake", "fill-deepfake", dfScore);
    animateScore("score-nsfw", "fill-nsfw", nsfwScore);
    animateScore("score-safe", "fill-safe", safeScore);

    // Update colors based on scores
    const dfEl = $("#card-deepfake .score-card-value");
    const nsfwEl = $("#card-nsfw .score-card-value");
    dfEl.style.color = dfScore > 50 ? "var(--red)" : dfScore > 25 ? "var(--yellow)" : "var(--green)";
    nsfwEl.style.color = nsfwScore > 30 ? "var(--red)" : nsfwScore > 10 ? "var(--yellow)" : "var(--green)";

    // Flags
    const flagsList = $("#flags-list");
    flagsList.innerHTML = "";
    const FLAG_LABELS = {
        deepfake: "Deepfake Detected",
        sexual: "Sexual Content",
        nudity: "Nudity",
        sensual: "Sensual Content",
        lingerie: "Revealing Clothing",
        intimacy: "Physical Intimacy",
        violence: "Violence",
        weapons: "Weapons",
        drugs: "Drug Content",
        hate: "Hate Symbols",
        child_safety: "Child Safety Risk",
    };
    const severeFlags = ["deepfake", "nudity", "sexual", "violence", "weapons", "child_safety"];

    data.flags.forEach(flag => {
        const chip = document.createElement("span");
        chip.className = `flag-chip ${severeFlags.includes(flag) ? "flag-chip--danger" : "flag-chip--warning"}`;
        chip.innerHTML = `<span>${severeFlags.includes(flag) ? "\u26a0" : "\u25cb"}</span> ${FLAG_LABELS[flag] || flag}`;
        flagsList.appendChild(chip);
    });

    if (data.flags.length === 0) {
        flagsList.innerHTML = '<span class="flag-chip" style="background:var(--green-dim);color:var(--green);border:1px solid rgba(52,211,153,0.15)">\u2713 No issues detected</span>';
    }

    // Protection plan
    const planChips = $("#plan-chips");
    planChips.innerHTML = "";
    const PLAN_LABELS = {
        anti_deepfake: "Anti-Deepfake Shield",
        clip: "NSFW Classifier Bypass",
    };
    // Always add remux
    data.recommended_attacks.forEach(atk => {
        const chip = document.createElement("span");
        chip.className = "plan-chip";
        chip.textContent = PLAN_LABELS[atk] || atk;
        planChips.appendChild(chip);
    });
    const remuxChip = document.createElement("span");
    remuxChip.className = "plan-chip";
    remuxChip.textContent = "Hash Bypass (Remux)";
    planChips.appendChild(remuxChip);

    // Time estimate
    const estMins = Math.ceil(data.estimated_time_seconds / 60);
    $("#plan-time").textContent = `Est. ~${estMins} min`;

    // Auto-fill repel text
    if (data.repel_texts && data.repel_texts.length > 0) {
        $("#repel-text").value = data.repel_texts.join(", ");
    }
}

function animateScore(valueId, fillId, target) {
    const valueEl = $(`#${valueId}`);
    const fillEl = $(`#${fillId}`);
    let current = 0;
    const step = Math.max(1, Math.floor(target / 30));
    const timer = setInterval(() => {
        current = Math.min(current + step, target);
        valueEl.textContent = current + "%";
        fillEl.style.width = current + "%";
        if (current >= target) clearInterval(timer);
    }, 30);
}

// ── Epsilon slider ──
$("#epsilon-slider").addEventListener("input", () => {
    $("#eps-val").textContent = $("#epsilon-slider").value;
});

// ══════════════════════════════════════════
//  SCREEN 3: PROCESSING
// ══════════════════════════════════════════
$("#protect-btn").addEventListener("click", () => startProcessing());

async function startProcessing() {
    if (!state.videoId) return;
    $("#protect-btn").disabled = true;

    const method = $("#attack-method").value;
    const epsVal = parseInt($("#epsilon-slider").value);

    const body = {
        video_id: state.videoId,
        attack_method: method,
        preset: $("#preset").value,
        target_text: $("#target-text").value,
        repel_text: $("#repel-text").value,
        epsilon: method === "auto" ? 0 : epsVal / 255,
        alpha: 0,
        yolo_weight: 0.2,
        clip_weight: 0.8,
    };

    switchScreen("processing");
    state.processStart = Date.now();

    try {
        const resp = await fetch(`${API}/api/process`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        state.jobId = data.job_id;
        connectProgress(data.job_id);
    } catch (err) {
        $("#progress-pct").textContent = "Error starting job";
    }
}

function connectProgress(jobId) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/progress/${jobId}`);
    state.ws = ws;

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        updateProgress(msg);
    };
    ws.onerror = () => pollProgress(jobId);
    ws.onclose = () => { state.ws = null; };
}

function pollProgress(jobId) {
    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`${API}/api/status/${jobId}`);
            const msg = await resp.json();
            updateProgress(msg);
            if (msg.status === "completed" || msg.status === "failed") {
                clearInterval(interval);
            }
        } catch(e) {}
    }, 1500);
}

const STAGE_MAP = {
    extracting: "step-analyze",
    classifying_original: "step-analyze",
    analyzing_content: "step-analyze",
    computing_uap: "step-uap",
    anti_deepfake_pgd: "step-deepfake",
    perturbing: "step-stream",
    streaming: "step-stream",
    verifying: "step-stream",
    reconstructing: "step-remux",
};

const STAGE_ORDER = ["step-analyze", "step-uap", "step-deepfake", "step-stream", "step-remux"];

function updateProgress(msg) {
    const pct = msg.progress || 0;
    const stage = msg.stage || "";

    // Update progress bar
    $("#main-progress").style.width = pct + "%";
    $("#progress-pct").textContent = pct + "%";

    // ETA
    if (state.processStart && pct > 5) {
        const elapsed = (Date.now() - state.processStart) / 1000;
        const total = elapsed / (pct / 100);
        const remaining = Math.max(0, total - elapsed);
        const remMins = Math.floor(remaining / 60);
        const remSecs = Math.floor(remaining % 60);
        $("#progress-eta").textContent = `~${remMins}:${remSecs.toString().padStart(2,"0")} remaining`;
    }

    // Update pipeline steps
    const activeStep = STAGE_MAP[stage];
    if (activeStep) {
        const activeIdx = STAGE_ORDER.indexOf(activeStep);
        STAGE_ORDER.forEach((stepId, i) => {
            const el = $(`#${stepId}`);
            el.classList.remove("pipeline-step--active", "pipeline-step--done");
            if (i < activeIdx) el.classList.add("pipeline-step--done");
            else if (i === activeIdx) el.classList.add("pipeline-step--active");
        });
    }

    // Completed
    if (msg.status === "completed") {
        STAGE_ORDER.forEach(id => {
            $(`#${id}`).classList.remove("pipeline-step--active");
            $(`#${id}`).classList.add("pipeline-step--done");
        });
        $("#progress-eta").textContent = "Complete!";
        setTimeout(() => loadResults(), 800);
    }

    // Failed
    if (msg.status === "failed") {
        $("#progress-pct").textContent = "Failed";
        $("#progress-eta").textContent = msg.error || "Unknown error";
    }
}

// ══════════════════════════════════════════
//  SCREEN 4: RESULTS
// ══════════════════════════════════════════
async function loadResults() {
    try {
        const resp = await fetch(`${API}/api/results/${state.jobId}`);
        const result = await resp.json();
        switchScreen("results");
        renderResults(result);
    } catch (err) {
        switchScreen("results");
        $("#result-badge").innerHTML = '<span style="color:var(--red)">Failed to load results</span>';
    }
}

function renderResults(result) {
    // Set video sources
    const origVideo = $("#video-original");
    const protVideo = $("#video-protected");

    // Original video is the uploaded one
    if (state.videoId) {
        origVideo.src = `${API}/api/export-original/${state.videoId}`;
    }
    // Protected video
    if (state.jobId) {
        protVideo.src = `${API}/api/export/${state.jobId}`;
    }

    // Render comparison scores
    const origScores = result.original_classifications || [];
    const pertScores = result.perturbed_classifications || [];

    const origEl = $("#scores-original");
    const protEl = $("#scores-protected");

    if (origScores.length && pertScores.length) {
        const orig = origScores[0];
        const pert = pertScores[0];

        origEl.innerHTML = renderScoreRows(orig, false);
        protEl.innerHTML = renderScoreRows(pert, true, orig);

        // Delta summary
        renderDeltaSummary(origScores, pertScores);
    } else {
        origEl.innerHTML = '<p style="padding:8px;color:var(--text-dim);font-size:13px">Classification data not available</p>';
        protEl.innerHTML = origEl.innerHTML;
    }

    // Check if all passed
    let allPassed = true;
    if (pertScores.length) {
        const pert = pertScores[0];
        if (pert.clip_scores) {
            const nsfw = Object.entries(pert.clip_scores)
                .filter(([k]) => k.toLowerCase().includes("adult") || k.toLowerCase().includes("nudity") || k.toLowerCase().includes("explicit"))
                .some(([, v]) => v > 0.15);
            if (nsfw) allPassed = false;
        }
    }

    const badge = $("#result-badge");
    if (allPassed) {
        badge.style.background = "var(--green-dim)";
        badge.style.borderColor = "rgba(52,211,153,0.2)";
        badge.style.color = "var(--green)";
        badge.innerHTML = `
            <svg width="28" height="28" viewBox="0 0 32 32" fill="none"><path d="M16 3L5 9v8c0 7 4.5 13.5 11 15 6.5-1.5 11-8 11-15V9L16 3z" fill="currentColor" opacity="0.12" stroke="currentColor" stroke-width="1.5"/><path d="M11 16l3.5 3.5L21 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <span>Protection Successful - All Checks Passed</span>
        `;
    }
}

function renderScoreRows(cls, isProtected, origCls) {
    let html = "";

    // YOLO detections
    const dets = cls.yolo_detections ? cls.yolo_detections.length : 0;
    html += `<div class="score-row"><span class="score-label">Objects detected</span><span class="score-value">${dets}</span></div>`;

    // CLIP scores (top 5)
    if (cls.clip_scores) {
        const sorted = Object.entries(cls.clip_scores).sort((a, b) => b[1] - a[1]).slice(0, 5);
        for (const [label, score] of sorted) {
            const pct = (score * 100).toFixed(1);
            let cls2 = "score-value--neutral";

            if (isProtected && origCls?.clip_scores) {
                const origScore = origCls.clip_scores[label] || 0;
                const delta = score - origScore;
                const isNsfw = ["adult", "nudity", "explicit", "sexual", "violence"].some(w => label.toLowerCase().includes(w));
                if (isNsfw) cls2 = delta < -0.02 ? "score-value--good" : delta > 0.02 ? "score-value--bad" : "score-value--neutral";
                else cls2 = delta > 0.02 ? "score-value--good" : "score-value--neutral";
            }

            const shortLabel = label.length > 35 ? label.slice(0, 35) + "\u2026" : label;
            html += `<div class="score-row"><span class="score-label">${shortLabel}</span><span class="score-value ${cls2}">${pct}%</span></div>`;
        }
    }
    return html;
}

function renderDeltaSummary(origScores, pertScores) {
    const container = $("#delta-summary");
    if (!origScores.length || !pertScores.length) {
        container.innerHTML = "";
        return;
    }

    const orig = origScores[0];
    const pert = pertScores[0];
    let html = '<h3 style="font-size:13px;color:var(--text-secondary);margin-bottom:12px;letter-spacing:0.02em">Score Changes</h3>';

    if (orig.clip_scores && pert.clip_scores) {
        const allLabels = Object.keys(orig.clip_scores);
        const changes = allLabels.map(label => ({
            label,
            before: orig.clip_scores[label] || 0,
            after: pert.clip_scores[label] || 0,
            delta: (pert.clip_scores[label] || 0) - (orig.clip_scores[label] || 0),
        })).filter(c => Math.abs(c.delta) > 0.01).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

        changes.forEach(c => {
            const isNsfw = ["adult", "nudity", "explicit", "sexual", "violence"].some(w => c.label.toLowerCase().includes(w));
            const isGood = isNsfw ? c.delta < 0 : c.delta > 0;
            const color = isGood ? "var(--green)" : "var(--red)";
            const shortLabel = c.label.length > 40 ? c.label.slice(0, 40) + "\u2026" : c.label;

            html += `<div class="delta-row">
                <span class="delta-label">${shortLabel}</span>
                <span class="delta-before">${(c.before * 100).toFixed(1)}%</span>
                <span class="delta-arrow">\u2192</span>
                <span class="delta-after" style="color:${color}">${(c.after * 100).toFixed(1)}%</span>
                <span class="delta-change" style="color:${color}">${c.delta > 0 ? "+" : ""}${(c.delta * 100).toFixed(1)}%</span>
            </div>`;
        });
    }

    container.innerHTML = html;
}

// ── Download ──
$("#download-btn").addEventListener("click", () => {
    if (state.jobId) {
        window.location.href = `${API}/api/export/${state.jobId}`;
    }
});

// ── Restart ──
$("#restart-btn").addEventListener("click", () => {
    state = { videoId: null, jobId: null, analysis: null, ws: null, processStart: null };
    // Reset upload zone
    hide($("#upload-progress"));
    $("#upload-fill").style.width = "0%";
    // Reset pipeline steps
    STAGE_ORDER.forEach(id => {
        $(`#${id}`).classList.remove("pipeline-step--active", "pipeline-step--done");
    });
    $("#main-progress").style.width = "0%";
    $("#progress-pct").textContent = "0%";
    $("#progress-eta").textContent = "";
    $("#protect-btn").disabled = false;
    switchScreen("upload");
});

// ── Nav status updates ──
function updateNavStatus(text, color) {
    const dot = $(".status-dot");
    const label = $(".nav-status span:last-child");
    dot.style.background = color || "var(--green)";
    dot.style.boxShadow = `0 0 8px ${color || "var(--green)"}`;
    label.textContent = text;
}
