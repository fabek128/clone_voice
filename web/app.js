const cloneForm = document.getElementById("cloneForm");
const fxForm = document.getElementById("fxForm");
const modeSelect = document.getElementById("modeSelect");
const singleTextField = document.getElementById("singleTextField");
const batchTextField = document.getElementById("batchTextField");
const applyFxToggle = document.getElementById("applyFxToggle");
const fxOptions = document.getElementById("fxOptions");
const cloneStatus = document.getElementById("cloneStatus");
const fxStatus = document.getElementById("fxStatus");
const tasksList = document.getElementById("tasksList");
const taskSummary = document.getElementById("taskSummary");
const liveIndicator = document.getElementById("liveIndicator");

function setStatus(el, text, tone = "info") {
  el.textContent = text;
  el.style.color = tone === "error" ? "#8a1c1c" : "#5a5a5a";
}

function toggleMode() {
  const isBatch = modeSelect.value === "batch";
  singleTextField.classList.toggle("hidden", isBatch);
  batchTextField.classList.toggle("hidden", !isBatch);
}

function toggleFxOptions() {
  fxOptions.classList.toggle("hidden", !applyFxToggle.checked);
}

modeSelect.addEventListener("change", toggleMode);
applyFxToggle.addEventListener("change", toggleFxOptions);

function parseNumber(value) {
  if (value === "" || value === null || value === undefined) {
    return null;
  }
  const num = Number(value);
  return Number.isNaN(num) ? null : num;
}

function buildFxConfig(form) {
  const cfg = {};
  const preset = form.fx_preset?.value || "none";
  if (preset && preset !== "none") {
    cfg.preset = preset;
  }
  const mapping = [
    ["fx_pitch", "pitch"],
    ["fx_gain_db", "gain_db"],
    ["fx_reverb", "reverb"],
    ["fx_echo_ms", "echo_ms"],
    ["fx_echo_feedback", "echo_feedback"],
    ["fx_echo_mix", "echo_mix"],
    ["fx_distortion", "distortion"],
    ["fx_lowpass_hz", "lowpass_hz"],
    ["fx_highpass_hz", "highpass_hz"],
  ];
  mapping.forEach(([field, key]) => {
    const value = parseNumber(form[field]?.value);
    if (value !== null) {
      cfg[key] = value;
    }
  });
  if (form.fx_compress?.checked) cfg.compress = true;
  if (form.fx_limit?.checked) cfg.limit = true;
  if (form.fx_normalize?.checked) cfg.normalize = true;
  return cfg;
}

async function loadDefaults() {
  try {
    const res = await fetch("/api/config");
    const data = await res.json();
    const defaults = data.defaults || {};
    cloneForm.model.value = defaults.model || "";
    cloneForm.language.value = defaults.language || "Auto";
    cloneForm.device.value = defaults.device || "auto";
    cloneForm.dtype.value = defaults.dtype || "auto";
    cloneForm.attn.value = defaults.attn || "auto";
    cloneForm.chunk_size.value = defaults.batch_chunk_size || 8;
  } catch (err) {
    console.warn("Failed to load defaults", err);
  }
}

cloneForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(cloneStatus, "Submitting job...");

  const fd = new FormData(cloneForm);
  fd.set("mode", modeSelect.value);
  fd.set("x_vector_only", cloneForm.x_vector_only.checked ? "true" : "false");
  fd.set("apply_fx", applyFxToggle.checked ? "true" : "false");

  const extraKw = cloneForm.extra_kwargs.value.trim();
  if (extraKw) {
    fd.set("extra_kwargs", extraKw);
  }

  if (applyFxToggle.checked) {
    const fxCfg = buildFxConfig(cloneForm);
    fd.set("fx_config", JSON.stringify(fxCfg));
  }

  try {
    const res = await fetch("/api/clone", { method: "POST", body: fd });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || "Request failed");
    }
    setStatus(cloneStatus, "Task queued.");
    cloneForm.reset();
    toggleMode();
    toggleFxOptions();
  } catch (err) {
    setStatus(cloneStatus, `Error: ${err.message}`, "error");
  }
});

fxForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(fxStatus, "Submitting FX job...");

  const fd = new FormData();
  const file = fxForm.input_audio.files[0];
  if (!file) {
    setStatus(fxStatus, "Select an audio file first.", "error");
    return;
  }
  fd.append("input_audio", file);
  const fxCfg = buildFxConfig(fxForm);
  fd.append("fx_config", JSON.stringify(fxCfg));

  try {
    const res = await fetch("/api/fx", { method: "POST", body: fd });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || "Request failed");
    }
    setStatus(fxStatus, "FX task queued.");
    fxForm.reset();
  } catch (err) {
    setStatus(fxStatus, `Error: ${err.message}`, "error");
  }
});

function renderTasks(tasks) {
  if (!tasks || tasks.length === 0) {
    tasksList.innerHTML = "<div class='task-meta'>No tasks yet.</div>";
    taskSummary.textContent = "No tasks yet";
    return;
  }

  const counts = tasks.reduce(
    (acc, task) => {
      acc.total += 1;
      acc[task.status] = (acc[task.status] || 0) + 1;
      return acc;
    },
    { total: 0 }
  );

  taskSummary.textContent = `Total ${counts.total} | running ${counts.running || 0} | queued ${
    counts.queued || 0
  } | done ${counts.done || 0} | error ${counts.error || 0}`;

  tasksList.innerHTML = "";
  tasks.forEach((task) => {
    const card = document.createElement("div");
    card.className = "task";

    const header = document.createElement("div");
    header.className = "task-header";

    const title = document.createElement("div");
    title.className = "task-title";
    title.textContent = `${task.type} #${task.id}`;

    const badge = document.createElement("span");
    badge.className = `badge ${task.status}`;
    badge.textContent = task.status;

    header.appendChild(title);
    header.appendChild(badge);

    const meta = document.createElement("div");
    meta.className = "task-meta";
    meta.textContent = `${task.message || ""} | created ${task.created_at || ""}`;

    const files = document.createElement("div");
    files.className = "task-files";
    if (task.outputs && task.outputs.length) {
      task.outputs.forEach((output) => {
        const link = document.createElement("a");
        link.href = output.file;
        link.textContent = output.file.split("/").slice(-1)[0];
        link.target = "_blank";
        files.appendChild(link);
      });
    }

    if (task.error) {
      const error = document.createElement("div");
      error.className = "task-meta";
      error.style.color = "#8a1c1c";
      error.textContent = `Error: ${task.error}`;
      card.appendChild(error);
    }

    card.appendChild(header);
    card.appendChild(meta);
    if (files.childElementCount) {
      card.appendChild(files);
    }
    tasksList.appendChild(card);
  });
}

function connectEvents() {
  const events = new EventSource("/api/events");
  events.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    renderTasks(payload.tasks || []);
    liveIndicator.classList.remove("hidden");
  };
  events.onerror = () => {
    liveIndicator.classList.add("hidden");
  };
}

toggleMode();
toggleFxOptions();
loadDefaults();
connectEvents();
