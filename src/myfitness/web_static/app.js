const LAYOUT_KEY = "myfitness.layout";
const NARROW_INSPECTOR = 1180;
const NARROW_SIDEBAR = 760;

const state = {
  sessions: [],
  activeId: null,
  currentSession: null,
  sending: false,
  tasks: [],
  taskTypes: {},
  editingTaskId: null,
  models: [],
  providers: [],
  activeModelId: null,
  editingModelId: null,
  artifacts: [],
  openArtifactPath: null,
};

const el = (id) => document.getElementById(id);
const sessionList = el("sessionList");
const messages = el("messages");
const welcome = el("welcomeState");
const input = el("messageInput");
const sendButton = el("sendButton");

const icons = {
  chat: '<svg viewBox="0 0 24 24"><path d="M5 6h14v10H9l-4 3z"/></svg>',
  agent: '<svg viewBox="0 0 24 24"><path d="M7 12h10M9.5 8v8M14.5 8v8M4.5 10v4M19.5 10v4"/></svg>',
  clock: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/></svg>',
};

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[char]));
}

function renderMarkdown(source) {
  const codeBlocks = [];
  let text = escapeHtml(source).replace(/```([\w-]*)\n([\s\S]*?)```/g, (_, language, code) => {
    const lang = (language || "").toLowerCase();
    const block = lang === "mermaid"
      ? `<pre class="mermaid-block"><span class="mermaid-tag">Mermaid 图</span><code>${code.trim()}</code></pre>`
      : `<pre><code data-language="${escapeHtml(language)}">${code.trim()}</code></pre>`;
    return `@@CODEBLOCK${codeBlocks.push(block) - 1}@@`;
  });
  text = text
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^[-*] (.+)$/gm, "<li>$1</li>")
    .replace(/(?:<li>.*<\/li>\n?)+/g, (list) => `<ul>${list}</ul>`)
    .split(/\n{2,}/)
    .map((block) => /^(<h\d|<ul|<pre|@@CODEBLOCK)/.test(block) ? block : `<p>${block.replace(/\n/g, "<br>")}</p>`)
    .join("");
  return text.replace(/@@CODEBLOCK(\d+)@@/g, (_, index) => codeBlocks[Number(index)]);
}

function relativeTime(value) {
  const time = new Date(value).getTime();
  if (!Number.isFinite(time)) return "";
  const minutes = Math.max(0, Math.floor((Date.now() - time) / 60000));
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)} 小时前`;
  if (minutes < 10080) return `${Math.floor(minutes / 1440)} 天前`;
  return new Date(value).toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
}

function formatSize(bytes = 0) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/* ------------------------------------------------------------------ 布局 */

function panelNode(name) {
  return name === "sidebar" ? document.querySelector(".session-sidebar") : el("inspector");
}

function isNarrow(name) {
  return window.innerWidth <= (name === "sidebar" ? NARROW_SIDEBAR : NARROW_INSPECTOR);
}

function panelOpen(name) {
  // 窄屏：浮层用 .open 控制；宽屏：网格列宽用 body 上的 hide-* 控制
  return isNarrow(name)
    ? panelNode(name).classList.contains("open")
    : !document.body.classList.contains(`hide-${name}`);
}

function setPanel(name, open) {
  const node = panelNode(name);
  if (isNarrow(name)) {
    node.classList.toggle("open", open);
    document.body.classList.remove(`hide-${name}`);
  } else {
    node.classList.remove("open");
    document.body.classList.toggle(`hide-${name}`, !open);
  }
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify({
      sidebar: panelOpen("sidebar"),
      inspector: panelOpen("inspector"),
    }));
  } catch { /* 隐私模式下 localStorage 不可用，忽略 */ }
}

function togglePanel(name) {
  setPanel(name, !panelOpen(name));
}

function initLayout() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) || "{}") || {}; } catch { saved = {}; }
  const width = window.innerWidth;
  // 窄屏一律从收起开始，浮层不该一上来就盖住对话区
  setPanel("sidebar", Boolean(saved.sidebar) && width > NARROW_SIDEBAR);
  setPanel("inspector", Boolean(saved.inspector) && width > NARROW_INSPECTOR);

  let lastWidth = width;
  window.addEventListener("resize", () => {
    const current = window.innerWidth;
    if (current <= NARROW_INSPECTOR && lastWidth > NARROW_INSPECTOR) setPanel("inspector", false);
    if (current <= NARROW_SIDEBAR && lastWidth > NARROW_SIDEBAR) setPanel("sidebar", false);
    lastWidth = current;
  });
}

/* ------------------------------------------------------------------ 会话 */

function renderSessionList() {
  const query = el("sessionFilter").value.trim().toLowerCase();
  const visible = state.sessions.filter((item) => item.title.toLowerCase().includes(query));
  sessionList.innerHTML = visible.length ? visible.map((item) => `
    <button class="session-item ${item.session_id === state.activeId ? "active" : ""}" data-session="${item.session_id}">
      ${icons.chat}<strong>${escapeHtml(item.title)}</strong><small>${relativeTime(item.updated_at)}</small>
    </button>
  `).join("") : '<div class="empty-list">还没有对话记录</div>';
  sessionList.querySelectorAll("[data-session]").forEach((button) => {
    button.addEventListener("click", () => loadSession(button.dataset.session));
  });
}

async function refreshSessions() {
  const data = await api("/api/sessions");
  state.sessions = data.sessions;
  renderSessionList();
}

async function newSession() {
  if (state.sending) return;
  const session = await api("/api/sessions", { method: "POST", body: "{}" });
  state.activeId = session.session_id;
  await refreshSessions();
  renderConversation(session);
  input.focus();
  if (isNarrow("sidebar")) setPanel("sidebar", false);
}

async function loadSession(sessionId) {
  if (state.sending) return;
  try {
    const session = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
    state.activeId = session.session_id;
    renderSessionList();
    renderConversation(session);
    if (isNarrow("sidebar")) setPanel("sidebar", false);
  } catch (error) { showToast(error.message); }
}

function renderConversation(session) {
  state.currentSession = session;
  el("chatTitle").textContent = session.title || "新对话";
  const items = session.messages || [];
  welcome.classList.toggle("hidden", items.length > 0);
  messages.innerHTML = items.map(messageHtml).join("");
  bindArtifactCards();
  closeArtifactView();
  syncArtifacts(items);
  requestAnimationFrame(scrollToBottom);
}

function messageHtml(message) {
  if (message.role === "user") {
    return `<article class="message user"><div class="user-bubble">${escapeHtml(message.content)}</div></article>`;
  }
  return `<article class="message assistant"><div class="message-avatar">${icons.agent}</div><div class="message-content">${renderMarkdown(message.content)}${artifactCardsHtml(message.artifacts)}</div></article>`;
}

/* ------------------------------------------------------------------ 产物 */

function artifactCardsHtml(artifacts) {
  if (!artifacts || !artifacts.length) return "";
  const cards = artifacts.map((artifact) => `
    <button class="artifact-card" data-artifact-path="${escapeHtml(artifact.path)}">
      <span class="artifact-card-icon ${artifact.kind === "chart" ? "chart" : "report"}">${artifact.kind === "chart" ? "◫" : "▤"}</span>
      <span class="artifact-card-copy">
        <strong>${escapeHtml(artifact.title || "会话产物")}</strong>
        <small>${escapeHtml([artifact.subtitle, artifact.path].filter(Boolean).join(" · "))}</small>
      </span>
      <span class="artifact-card-action">查看 ›</span>
    </button>`).join("");
  return `<div class="artifact-cards">${cards}</div>`;
}

function bindArtifactCards() {
  messages.querySelectorAll("[data-artifact-path]").forEach((button) => {
    button.addEventListener("click", () => openArtifact(button.dataset.artifactPath));
  });
}

function syncArtifacts(items) {
  const seen = new Set();
  const list = [];
  for (const message of items || []) {
    for (const artifact of message.artifacts || []) {
      if (artifact && artifact.path && !seen.has(artifact.path)) {
        seen.add(artifact.path);
        list.push(artifact);
      }
    }
  }
  state.artifacts = list;
  if (state.openArtifactPath && !seen.has(state.openArtifactPath)) closeArtifactView();
  renderArtifactList();
}

function renderArtifactList() {
  const container = el("artifactList");
  el("artifactCount").textContent = String(state.artifacts.length);
  if (!state.artifacts.length) {
    container.innerHTML = `
      <div class="artifact-empty">
        <strong>还没有产物</strong>
        <p>让 Agent 生成日报、周期报表或统计图，结果会作为卡片出现在对话里。</p>
      </div>`;
    return;
  }
  container.innerHTML = [...state.artifacts].reverse().map((artifact) => `
    <button class="artifact-row ${state.openArtifactPath === artifact.path ? "active" : ""}" data-artifact-path="${escapeHtml(artifact.path)}">
      <span class="artifact-row-icon ${artifact.kind === "chart" ? "chart" : "report"}">${artifact.kind === "chart" ? "◫" : "▤"}</span>
      <span class="artifact-row-copy">
        <strong>${escapeHtml(artifact.title || "会话产物")}</strong>
        <small>${escapeHtml(artifact.subtitle || (artifact.kind === "chart" ? "统计图" : "报告"))}</small>
      </span>
    </button>`).join("");
  container.querySelectorAll("[data-artifact-path]").forEach((button) => {
    button.addEventListener("click", () => openArtifact(button.dataset.artifactPath));
  });
}

async function openArtifact(path) {
  if (!path) return;
  setPanel("inspector", true);
  state.openArtifactPath = path;
  renderArtifactList();
  el("artifactName").textContent = "加载中…";
  el("artifactMeta").textContent = "";
  el("artifactBody").innerHTML = '<div class="artifact-loading">正在读取产物…</div>';
  el("artifactView").classList.remove("hidden");
  try {
    const data = await api(`/api/artifact?path=${encodeURIComponent(path)}`);
    el("artifactName").textContent = data.name;
    const kind = data.kind === "chart" ? "统计图" : "报告";
    el("artifactMeta").textContent = `${kind} · ${formatSize(data.size)} · ${relativeTime(data.modified_at)}`;
    el("artifactBody").innerHTML = renderMarkdown(data.content);
  } catch (error) {
    el("artifactName").textContent = "读取失败";
    el("artifactBody").innerHTML = `<div class="artifact-loading">${escapeHtml(error.message)}</div>`;
    showToast(error.message);
  }
}

function closeArtifactView() {
  state.openArtifactPath = null;
  el("artifactView").classList.add("hidden");
  renderArtifactList();
}

async function copyArtifactPath() {
  const path = state.openArtifactPath;
  if (!path) return;
  try {
    await navigator.clipboard.writeText(path);
    showToast("已复制产物路径");
  } catch { showToast("复制失败，请手动选择路径"); }
}

/* ------------------------------------------------------------------ 模型 */

async function loadModels() {
  try {
    const data = await api("/api/models");
    state.models = data.models || [];
    state.providers = data.providers || [];
    state.activeModelId = data.active_id || null;
  } catch (error) {
    state.models = [];
    state.providers = [];
    state.activeModelId = null;
  }
  renderModelSelect();
  renderModelList();
  renderProviderChips();
  renderModelSummary();
}

function activeModel() {
  return state.models.find((item) => item.id === state.activeModelId) || null;
}

function renderModelSummary() {
  const model = activeModel();
  const text = model ? `${model.name} · ${model.model}` : "未配置模型";
  el("settingsModelSummary").textContent = model
    ? `${model.name} · ${model.model}${model.has_key ? "" : "（缺少 API Key）"}`
    : "未配置，点击添加模型";
  el("footerModelLabel").textContent = text;
}

function renderModelSelect() {
  const select = el("modelSelect");
  if (!state.models.length) {
    select.innerHTML = '<option value="">未配置模型</option>';
    select.disabled = true;
    select.title = "在「设置 → 模型」中添加模型";
    return;
  }
  select.disabled = false;
  select.innerHTML = [
    state.activeModelId ? "" : '<option value="">未选择模型</option>',
    ...state.models.map((item) =>
      `<option value="${escapeHtml(item.id)}" ${item.id === state.activeModelId ? "selected" : ""}>${escapeHtml(item.name)} · ${escapeHtml(item.model)}</option>`),
  ].join("");
  select.title = "切换本轮对话使用的模型";
}

function renderProviderChips() {
  const container = el("providerChips");
  if (!state.providers.length) { container.innerHTML = ""; return; }
  container.innerHTML = `<span class="provider-chips-label">快速填入：</span>${state.providers.map((item, index) =>
    `<button type="button" class="provider-chip" data-provider="${index}">${escapeHtml(item.name)}</button>`).join("")}`;
  container.querySelectorAll("[data-provider]").forEach((button) => button.addEventListener("click", () => {
    const preset = state.providers[Number(button.dataset.provider)];
    if (!preset) return;
    el("modelFormName").value = preset.name;
    el("modelFormBaseUrl").value = preset.base_url || "";
    el("modelFormModel").value = preset.model || "";
    el("modelFormApiKey").focus();
  }));
}

function renderModelList() {
  const container = el("modelList");
  if (!state.models.length) {
    container.innerHTML = '<div class="model-empty">还没有配置模型。选择上方的服务商模板，填写 API Key 即可。</div>';
    return;
  }
  container.innerHTML = state.models.map((item) => `
    <article class="model-card ${item.id === state.activeModelId ? "active" : ""}">
      <button type="button" class="model-card-main" data-activate-model="${escapeHtml(item.id)}">
        <span class="model-radio"></span>
        <span class="model-card-copy">
          <strong>${escapeHtml(item.name)}${item.source === "env" ? '<em class="model-tag">.env</em>' : ""}</strong>
          <small>${escapeHtml(item.model)} · ${escapeHtml(item.base_url)}</small>
          <small class="model-key ${item.has_key ? "" : "missing"}">${item.has_key ? `API Key ${escapeHtml(item.key_hint)}` : "缺少 API Key"}</small>
        </span>
      </button>
      ${item.source === "env" ? "" : `
        <button type="button" class="model-card-action" data-edit-model="${escapeHtml(item.id)}">编辑</button>
        <button type="button" class="model-card-action danger" data-delete-model="${escapeHtml(item.id)}">删除</button>`}
    </article>`).join("");

  container.querySelectorAll("[data-activate-model]").forEach((button) => button.addEventListener("click", () => activateModel(button.dataset.activateModel)));
  container.querySelectorAll("[data-edit-model]").forEach((button) => button.addEventListener("click", () => startEditModel(button.dataset.editModel)));
  container.querySelectorAll("[data-delete-model]").forEach((button) => button.addEventListener("click", () => deleteModel(button.dataset.deleteModel)));
}

async function activateModel(modelId) {
  if (!modelId) return;
  try {
    const data = await api(`/api/models/${encodeURIComponent(modelId)}/activate`, { method: "POST", body: "{}" });
    applyModelPayload(data);
    showToast("已切换模型");
  } catch (error) { showToast(error.message); }
}

function applyModelPayload(data) {
  state.models = data.models || [];
  state.activeModelId = data.active_id || null;
  renderModelSelect();
  renderModelList();
  renderModelSummary();
}

function resetModelForm() {
  state.editingModelId = null;
  el("modelFormId").value = "";
  el("modelForm").reset();
  el("modelFormTemperature").value = "0.7";
  el("modelFormTimeout").value = "120";
  el("modelFormTitle").textContent = "添加模型";
  el("modelFormNote").textContent = "";
  el("modelFormNote").className = "model-form-note";
  el("modelFormCancel").classList.add("hidden");
  el("modelFormDelete").classList.add("hidden");
}

function startEditModel(modelId) {
  const model = state.models.find((item) => item.id === modelId);
  if (!model) return;
  state.editingModelId = modelId;
  el("modelFormId").value = model.id;
  el("modelFormName").value = model.name;
  el("modelFormModel").value = model.model;
  el("modelFormBaseUrl").value = model.base_url;
  el("modelFormApiKey").value = "";
  el("modelFormTemperature").value = String(model.temperature);
  el("modelFormTimeout").value = String(model.timeout);
  el("modelFormTitle").textContent = `编辑模型 · ${model.name}`;
  el("modelFormNote").textContent = model.has_key
    ? `已保存 API Key（${model.key_hint}）。留空表示保持原值。`
    : "尚未填写 API Key。";
  el("modelFormNote").className = "model-form-note";
  el("modelFormCancel").classList.remove("hidden");
  el("modelFormDelete").classList.remove("hidden");
  el("modelFormName").focus();
}

function collectModelForm() {
  const apiKey = el("modelFormApiKey").value.trim();
  const payload = {
    name: el("modelFormName").value.trim(),
    model: el("modelFormModel").value.trim(),
    base_url: el("modelFormBaseUrl").value.trim(),
    temperature: Number(el("modelFormTemperature").value || 0.7),
    timeout: Number(el("modelFormTimeout").value || 120),
  };
  const modelId = el("modelFormId").value.trim();
  if (modelId) payload.id = modelId;
  // 未输入时不传 api_key，后端保留原值
  if (apiKey) payload.api_key = apiKey;
  return payload;
}

async function submitModelForm(event) {
  event.preventDefault();
  const saveButton = el("modelFormSave");
  saveButton.disabled = true;
  try {
    const data = await api("/api/models", { method: "POST", body: JSON.stringify(collectModelForm()) });
    applyModelPayload(data);
    resetModelForm();
    showToast("模型已保存");
  } catch (error) {
    showToast(error.message);
  } finally {
    saveButton.disabled = false;
  }
}

async function testModelConnection() {
  const button = el("modelFormTest");
  const note = el("modelFormNote");
  button.disabled = true;
  note.className = "model-form-note";
  note.textContent = "正在测试连接…";
  try {
    const data = await api("/api/models/test", { method: "POST", body: JSON.stringify(collectModelForm()) });
    if (data.ok) {
      note.className = "model-form-note ok";
      note.textContent = `连接成功 · ${data.model} · ${data.latency_ms} ms${data.reply ? ` · 回复：${data.reply.slice(0, 40)}` : ""}`;
    } else {
      note.className = "model-form-note error";
      note.textContent = `连接失败：${data.error}`;
    }
  } catch (error) {
    note.className = "model-form-note error";
    note.textContent = `连接失败：${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function deleteModel(modelId) {
  if (!modelId || !window.confirm("删除这个模型配置？此操作不可撤销。")) return;
  try {
    const data = await api(`/api/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
    applyModelPayload(data);
    if (state.editingModelId === modelId) resetModelForm();
    showToast("模型已删除");
  } catch (error) { showToast(error.message); }
}

/* ------------------------------------------------------------------ 设置 */

function openSettings(page = "home") {
  el("settingsModal").classList.remove("hidden");
  showSettingsPage(page);
}

function showSettingsPage(page) {
  el("settingsHome").classList.toggle("hidden", page !== "home");
  el("settingsModels").classList.toggle("hidden", page !== "models");
  el("settingsTitle").textContent = page === "models" ? "模型" : "设置";
  el("settingsSubtitle").textContent = page === "models"
    ? "OpenAI 兼容协议 · 保存在本机数据目录"
    : "MyFitness Agent · 本地运行";
  if (page === "models") loadModels();
}

/* -------------------------------------------------------------- 定时任务 */

async function loadTasks(openModal = false) {
  if (openModal) el("taskModal").classList.remove("hidden");
  el("taskList").innerHTML = '<div class="task-empty">正在读取定时任务…</div>';
  try {
    const data = await api("/api/scheduled-tasks");
    state.tasks = data.tasks || [];
    state.taskTypes = data.task_types || {};
    const enabledCount = state.tasks.filter((task) => task.enabled).length;
    el("taskCount").textContent = state.tasks.length ? `${enabledCount}/${state.tasks.length}` : "0";
    const runtime = el("taskRuntimeStatus");
    runtime.classList.toggle("running", Boolean(data.scheduler_running));
    runtime.querySelector("span").textContent = data.scheduler_running ? "调度器运行中" : "调度器未运行";
    runtime.querySelector("small").textContent = data.timezone || "Asia/Shanghai";
    renderTasks();
  } catch (error) {
    el("taskList").innerHTML = `<div class="task-empty"><strong>读取失败</strong>${escapeHtml(error.message)}</div>`;
    showToast(error.message);
  }
}

function renderTasks() {
  const container = el("taskList");
  if (!state.tasks.length) {
    container.innerHTML = '<div class="task-empty"><strong>暂无定时任务</strong>可在对话中说“每天早上 7 点生成日报”来创建。</div>';
    return;
  }
  container.innerHTML = state.tasks.map((task) => {
    const editing = Number(state.editingTaskId) === Number(task.id);
    const lastRun = task.last_run_at ? new Date(task.last_run_at).toLocaleString("zh-CN", { hour12: false }) : "尚未执行";
    return `
      <article class="task-card" data-task-id="${task.id}">
        <div class="task-card-main">
          <div class="task-type-icon">${icons.clock}</div>
          <div class="task-card-copy">
            <strong>${escapeHtml(task.label)}</strong>
            <span>${escapeHtml(task.content_label)} · 每天 <b>${escapeHtml(task.time_of_day)}</b> · ${escapeHtml(lastRun)}</span>
          </div>
          <button class="task-edit-button" data-edit-task="${task.id}">${editing ? "收起" : "编辑"}</button>
          <label class="switch" title="${task.enabled ? "关闭任务" : "开启任务"}">
            <input type="checkbox" data-toggle-task="${task.id}" ${task.enabled ? "checked" : ""} />
            <span></span>
          </label>
        </div>
        ${editing ? taskEditorHtml(task) : ""}
      </article>`;
  }).join("");

  container.querySelectorAll("[data-edit-task]").forEach((button) => button.addEventListener("click", () => {
    const taskId = Number(button.dataset.editTask);
    state.editingTaskId = state.editingTaskId === taskId ? null : taskId;
    renderTasks();
  }));
  container.querySelectorAll("[data-toggle-task]").forEach((checkbox) => checkbox.addEventListener("change", async () => {
    checkbox.disabled = true;
    await updateTask(Number(checkbox.dataset.toggleTask), { enabled: checkbox.checked });
  }));
  container.querySelectorAll(".task-cancel-edit").forEach((button) => button.addEventListener("click", () => {
    state.editingTaskId = null;
    renderTasks();
  }));
  container.querySelectorAll(".task-edit-form").forEach((form) => form.addEventListener("submit", submitTaskEdit));
}

function taskEditorHtml(task) {
  const options = Object.entries(state.taskTypes).map(([value, label]) =>
    `<option value="${escapeHtml(value)}" ${task.task_type === value ? "selected" : ""}>${escapeHtml(label)}</option>`
  ).join("");
  return `
    <form class="task-editor task-edit-form" data-task-id="${task.id}">
      <div class="task-form-grid">
        <label class="task-field full"><span>任务名称</span><input name="label" maxlength="128" required value="${escapeHtml(task.label)}" /></label>
        <label class="task-field"><span>任务内容</span><select name="task_type">${options}</select></label>
        <label class="task-field"><span>每天执行时间</span><input name="time_of_day" type="time" required value="${escapeHtml(task.time_of_day)}" /></label>
      </div>
      <p class="task-editor-note">任务按 Asia/Shanghai 时区执行。修改任务内容时，同一类型只能保留一个任务。</p>
      <div class="task-form-actions"><button type="button" class="task-cancel-edit">取消</button><button type="submit" class="task-save">提交修改</button></div>
    </form>`;
}

async function submitTaskEdit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const saveButton = form.querySelector(".task-save");
  saveButton.disabled = true;
  const formData = new FormData(form);
  await updateTask(Number(form.dataset.taskId), {
    label: formData.get("label"),
    task_type: formData.get("task_type"),
    time_of_day: formData.get("time_of_day"),
  }, true);
}

async function updateTask(taskId, changes, closeEditor = false) {
  try {
    const result = await api(`/api/scheduled-tasks/${taskId}`, {
      method: "PATCH",
      body: JSON.stringify(changes),
    });
    if (result.scheduler_error) showToast(`数据库已保存，但调度器重载失败：${result.scheduler_error}`);
    else showToast("定时任务已更新");
    if (closeEditor) state.editingTaskId = null;
    await loadTasks(false);
  } catch (error) {
    showToast(error.message);
    await loadTasks(false);
  }
}

/* ------------------------------------------------------------------ 对话 */

function showProgress() {
  welcome.classList.add("hidden");
  messages.insertAdjacentHTML("beforeend", `
    <article class="message progress-message" id="progressMessage">
      <div class="message-avatar">${icons.agent}</div>
      <div class="message-content"><div class="typing-dots"><i></i><i></i><i></i></div><span>正在理解你的问题…</span></div>
    </article>`);
  scrollToBottom();
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || state.sending) return;
  if (!state.activeId) await newSession();
  state.sending = true;
  sendButton.disabled = true;
  input.value = "";
  resizeInput();
  welcome.classList.add("hidden");
  messages.insertAdjacentHTML("beforeend", messageHtml({ role: "user", content: text }));
  showProgress();
  try {
    const result = await api(`/api/sessions/${encodeURIComponent(state.activeId)}/messages`, {
      method: "POST",
      body: JSON.stringify({ message: text }),
    });
    renderConversation(result);
    await refreshSessions();
  } catch (error) {
    el("progressMessage")?.remove();
    messages.insertAdjacentHTML("beforeend", `<article class="message assistant"><div class="message-avatar">!</div><div class="message-content"><p>处理失败：${escapeHtml(error.message)}</p></div></article>`);
    showToast(error.message);
  } finally {
    state.sending = false;
    sendButton.disabled = false;
    input.focus();
    scrollToBottom();
  }
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

function scrollToBottom() {
  const scroller = el("messageScroll");
  scroller.scrollTop = scroller.scrollHeight;
}

function showToast(text) {
  const toast = el("toast");
  toast.textContent = text;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2600);
}

/* ------------------------------------------------------------------ 绑定 */

function bindEvents() {
  el("openTaskManager").addEventListener("click", () => loadTasks(true));
  el("closeTaskManager").addEventListener("click", () => el("taskModal").classList.add("hidden"));
  el("refreshTasks").addEventListener("click", () => loadTasks(false));
  el("taskModal").addEventListener("click", (event) => {
    if (event.target === el("taskModal")) el("taskModal").classList.add("hidden");
  });

  el("newChatButton").addEventListener("click", newSession);
  el("sessionFilter").addEventListener("input", renderSessionList);
  sendButton.addEventListener("click", sendMessage);
  input.addEventListener("input", resizeInput);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); }
  });

  // 布局：左栏 / 右栏在任意宽度下都可折叠
  el("toggleSidebar").addEventListener("click", () => togglePanel("sidebar"));
  el("collapseSidebar").addEventListener("click", () => setPanel("sidebar", false));
  el("toggleInspector").addEventListener("click", () => togglePanel("inspector"));
  el("closeInspector").addEventListener("click", () => setPanel("inspector", false));
  el("closeSessions").addEventListener("click", () => setPanel("sidebar", false));

  // 产物
  el("closeArtifact").addEventListener("click", closeArtifactView);
  el("copyArtifactPath").addEventListener("click", copyArtifactPath);

  // 设置 / 模型
  el("openSettings").addEventListener("click", () => openSettings("home"));
  el("closeSettings").addEventListener("click", () => el("settingsModal").classList.add("hidden"));
  el("settingsModal").addEventListener("click", (event) => {
    if (event.target === el("settingsModal")) el("settingsModal").classList.add("hidden");
  });
  el("settingsBack").addEventListener("click", () => showSettingsPage("home"));
  el("settingsHome").querySelectorAll("[data-settings-page]").forEach((button) => {
    button.addEventListener("click", () => showSettingsPage(button.dataset.settingsPage));
  });
  el("modelForm").addEventListener("submit", submitModelForm);
  el("modelFormTest").addEventListener("click", testModelConnection);
  el("modelFormCancel").addEventListener("click", resetModelForm);
  el("modelFormDelete").addEventListener("click", () => deleteModel(state.editingModelId));
  el("modelSelect").addEventListener("change", (event) => activateModel(event.target.value));

  document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { input.value = button.dataset.prompt; resizeInput(); input.focus(); }));
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); newSession(); }
    if (event.key !== "Escape") return;
    if (!el("settingsModal").classList.contains("hidden")) {
      if (!el("settingsModels").classList.contains("hidden")) showSettingsPage("home");
      else el("settingsModal").classList.add("hidden");
      return;
    }
    el("taskModal").classList.add("hidden");
  });
}

async function init() {
  initLayout();
  bindEvents();
  resetModelForm();
  renderArtifactList();
  try {
    await api("/api/health");
    el("connectionStatus").textContent = "已连接 · JSON 持久化";
    await Promise.all([refreshSessions(), loadTasks(false), loadModels()]);
    if (state.sessions.length) await loadSession(state.sessions[0].session_id);
    else await newSession();
  } catch (error) {
    el("connectionStatus").textContent = "连接失败";
    showToast(error.message);
  }
}

init();
