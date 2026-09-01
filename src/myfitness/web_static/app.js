const LAYOUT_KEY = "myfitness.layout";
const SIDEBAR_WIDTH_KEY = "myfitness.sidebarWidth";
const INSPECTOR_WIDTH_KEY = "myfitness.inspectorWidth";
const THEME_KEY = "myfitness.theme";
const NARROW_INSPECTOR = 1180;
const NARROW_SIDEBAR = 760;
const SIDEBAR_WIDTH_MIN = 220;
const SIDEBAR_WIDTH_MAX = 480;
const SIDEBAR_WIDTH_DEFAULT = 272;
const INSPECTOR_WIDTH_MIN = 280;
const INSPECTOR_WIDTH_MAX = 560;
const INSPECTOR_WIDTH_DEFAULT = 360;

const state = {
  sessions: [],
  activeId: null,
  currentSession: null,
  draft: true,
  sending: false,
  draft: true,
  tasks: [],
  taskTypes: {},
  editingTaskId: null,
  models: [],
  providers: [],
  activeModelId: null,
  editingModelId: null,
  knowledge: [],
  editingKnowledgeId: null,
  artifacts: [],
  artifactPdfUrl: null,
  openArtifactPath: null,
  theme: "dark",
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

const MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
let mermaidLoader = null;

function chartColors() {
  const style = getComputedStyle(document.documentElement);
  return [
    style.getPropertyValue("--chart-1").trim() || "#b7f34a",
    style.getPropertyValue("--chart-2").trim() || "#7fc7ff",
    style.getPropertyValue("--chart-3").trim() || "#c4a7ff",
    style.getPropertyValue("--chart-4").trim() || "#ffbd73",
  ];
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[char]));
}

function safeHref(href) {
  const decoded = String(href || "").replace(/&amp;/g, "&");
  if (/^(https?:|mailto:)/i.test(decoded)) return href;
  if (decoded.startsWith("#") || decoded.startsWith("/")) return href;
  return "#";
}

function formatInline(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) => (
      `<a href="${safeHref(href)}" target="_blank" rel="noopener noreferrer">${label}</a>`
    ))
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\*)([^*]+)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/(^|[^_])_([^_]+)_/g, "$1<em>$2</em>");
}

function splitTableRow(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function isTableRow(line) {
  return /^\s*\|.+\|\s*$/.test(line);
}

function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function renderMarkdownTables(text) {
  const lines = text.split("\n");
  const out = [];
  for (let i = 0; i < lines.length; i += 1) {
    if (isTableRow(lines[i]) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const header = splitTableRow(lines[i]);
      const aligns = splitTableRow(lines[i + 1]).map((cell) => {
        const left = cell.startsWith(":");
        const right = cell.endsWith(":");
        if (left && right) return "center";
        if (right) return "right";
        return "left";
      });
      i += 2;
      const rows = [];
      while (i < lines.length && isTableRow(lines[i])) {
        rows.push(splitTableRow(lines[i]));
        i += 1;
      }
      i -= 1;
      const head = header.map((cell, index) => (
        `<th style="text-align:${aligns[index] || "left"}">${formatInline(cell)}</th>`
      )).join("");
      const body = rows.map((row) => `<tr>${row.map((cell, index) => (
        `<td style="text-align:${aligns[index] || "left"}">${formatInline(cell)}</td>`
      )).join("")}</tr>`).join("");
      out.push(`<div class="md-table-wrap"><table class="md-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
      continue;
    }
    out.push(lines[i]);
  }
  return out.join("\n");
}

function renderMarkdownLists(text) {
  text = text.replace(/(?:^\d+\. .+(?:\n|$))+/gm, (block) => {
    const items = block.trim().split("\n").map((line) => (
      `<li>${formatInline(line.replace(/^\d+\.\s+/, ""))}</li>`
    )).join("");
    return `<ol>${items}</ol>\n`;
  });
  return text.replace(/(?:^[-*] .+(?:\n|$))+/gm, (block) => {
    const items = block.trim().split("\n").map((line) => (
      `<li>${formatInline(line.replace(/^[-*]\s+/, ""))}</li>`
    )).join("");
    return `<ul>${items}</ul>\n`;
  });
}

function parseQuotedList(inner) {
  return [...String(inner || "").matchAll(/"([^"]*)"/g)].map((match) => match[1]);
}

function parseXyChart(source) {
  const text = String(source || "").trim();
  if (!/^xychart-beta\b/i.test(text)) return null;
  const title = (text.match(/title\s+"([^"]*)"/) || [])[1] || "";
  const xMatch = text.match(/x-axis\s+(?:"[^"]*"\s+)?\[([\s\S]*?)\]/);
  const xLabels = xMatch ? parseQuotedList(xMatch[1]) : [];
  const yMatch = text.match(/y-axis\s+"([^"]*)"(?:\s+([-\d.]+)\s+-->\s+([-\d.]+))?/);
  const series = [];
  const seriesRe = /^\s*(line|bar)\s+\[([^\]]*)\]/gm;
  let match;
  while ((match = seriesRe.exec(text))) {
    series.push({
      type: match[1].toLowerCase(),
      values: match[2].split(",").map((value) => Number(value.trim())).filter(Number.isFinite),
    });
  }
  if (!xLabels.length || !series.length) return null;
  let yMin = yMatch && yMatch[2] != null ? Number(yMatch[2]) : NaN;
  let yMax = yMatch && yMatch[3] != null ? Number(yMatch[3]) : NaN;
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    const values = series.flatMap((item) => item.values);
    yMin = Math.min(...values);
    yMax = Math.max(...values);
  }
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  return { title, xLabels, yLabel: yMatch ? yMatch[1] : "", yMin, yMax, series };
}

function formatTick(value) {
  const abs = Math.abs(value);
  if (abs >= 100) return String(Math.round(value));
  const rounded = Math.round(value * 100) / 100;
  return String(rounded);
}

function renderXyChartSvg(chart) {
  const width = 720;
  const height = 300;
  const pad = { t: chart.title ? 38 : 18, r: 20, b: 42, l: 58 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const count = Math.max(chart.xLabels.length, ...chart.series.map((item) => item.values.length), 1);
  const xAt = (index) => pad.l + (count === 1 ? innerW / 2 : (index / (count - 1)) * innerW);
  const yAt = (value) => pad.t + (1 - (value - chart.yMin) / (chart.yMax - chart.yMin)) * innerH;
  const ticks = 4;
  let marks = "";
  for (let i = 0; i <= ticks; i += 1) {
    const value = chart.yMin + ((chart.yMax - chart.yMin) * i) / ticks;
    const y = yAt(value);
    marks += `<line class="chart-grid" x1="${pad.l}" y1="${y}" x2="${width - pad.r}" y2="${y}"></line>`;
    marks += `<text class="chart-tick" x="${pad.l - 8}" y="${y + 3}" text-anchor="end">${formatTick(value)}</text>`;
  }
  const colors = chartColors();
  const barSeries = chart.series.filter((item) => item.type === "bar");
  let plots = "";
  chart.series.forEach((item, seriesIndex) => {
    const color = colors[seriesIndex % colors.length];
    if (item.type === "bar") {
      const group = count === 1 ? innerW * 0.36 : innerW / count;
      const barW = Math.max(4, (group * 0.62) / Math.max(barSeries.length, 1));
      const offset = barSeries.indexOf(item);
      item.values.forEach((value, index) => {
        const x = xAt(index) - (barSeries.length * barW) / 2 + offset * barW;
        const y = yAt(value);
        const barH = Math.max(0, yAt(chart.yMin) - y);
        plots += `<rect x="${x}" y="${y}" width="${barW}" height="${barH}" rx="2" fill="${color}" opacity="0.92"><title>${escapeHtml(chart.xLabels[index] || "")}: ${value}</title></rect>`;
      });
      return;
    }
    const points = item.values.map((value, index) => `${xAt(index)},${yAt(value)}`).join(" ");
    const baseline = `${xAt(0)},${yAt(chart.yMin)} ${points} ${xAt(item.values.length - 1)},${yAt(chart.yMin)}`;
    plots += `<polygon fill="${color}" opacity="0.12" points="${baseline}"></polygon>`;
    plots += `<polyline fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" points="${points}"></polyline>`;
    item.values.forEach((value, index) => {
      plots += `<circle cx="${xAt(index)}" cy="${yAt(value)}" r="3.4" fill="${color}" stroke="var(--chart-bg)" stroke-width="1.2"><title>${escapeHtml(chart.xLabels[index] || "")}: ${value}</title></circle>`;
    });
  });
  const step = Math.max(1, Math.ceil(count / 8));
  let labels = "";
  chart.xLabels.forEach((label, index) => {
    if (index % step !== 0 && index !== count - 1) return;
    labels += `<text class="chart-tick" x="${xAt(index)}" y="${height - 14}" text-anchor="middle">${escapeHtml(label)}</text>`;
  });
  const title = chart.title
    ? `<text class="chart-title" x="${width / 2}" y="22" text-anchor="middle">${escapeHtml(chart.title)}</text>`
    : "";
  const yTitle = chart.yLabel
    ? `<text class="chart-axis-title" x="14" y="${pad.t + innerH / 2}" text-anchor="middle" transform="rotate(-90 14 ${pad.t + innerH / 2})">${escapeHtml(chart.yLabel)}</text>`
    : "";
  return `<div class="chart-card"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(chart.title || chart.yLabel || "统计图")}">${title}${yTitle}${marks}${plots}${labels}</svg></div>`;
}

function renderMermaidBlock(code) {
  const chart = parseXyChart(code);
  if (chart) return renderXyChartSvg(chart);
  return `<div class="mermaid-wrap"><pre class="mermaid">${escapeHtml(code)}</pre></div>`;
}

function renderMarkdown(source) {
  const blocks = [];
  const normalized = String(source ?? "").replace(/\r\n/g, "\n");
  let text = normalized.replace(/```([\w-]*)[ \t]*\n([\s\S]*?)```/g, (_, language, code) => {
    const lang = (language || "").toLowerCase();
    const html = lang === "mermaid"
      ? renderMermaidBlock(code.trim())
      : `<pre><code${lang ? ` data-language="${escapeHtml(lang)}"` : ""}>${escapeHtml(code.trim())}</code></pre>`;
    return `\n@@CODEBLOCK${blocks.push(html) - 1}@@\n`;
  });
  text = escapeHtml(text);
  text = renderMarkdownTables(text);
  text = text
    .replace(/^#{3} (.+)$/gm, (_, title) => `<h3>${formatInline(title)}</h3>`)
    .replace(/^#{2} (.+)$/gm, (_, title) => `<h2>${formatInline(title)}</h2>`)
    .replace(/^# (.+)$/gm, (_, title) => `<h1>${formatInline(title)}</h1>`)
    .replace(/^(?:-\s*){3,}$/gm, "<hr>")
    .replace(/(?:^&gt; .+\n?)+/gm, (block) => {
      const body = block.replace(/^&gt; /gm, "").trim().replace(/\n/g, "<br>");
      return `<blockquote>${formatInline(body)}</blockquote>`;
    });
  text = renderMarkdownLists(text);
  text = text.split(/\n{2,}/).map((block) => {
    const trimmed = block.trim();
    if (!trimmed) return "";
    if (/^(<h\d|<ul|<ol|<pre|<table|<div|<blockquote|<hr|@@CODEBLOCK)/.test(trimmed)) return trimmed;
    return `<p>${formatInline(trimmed).replace(/\n/g, "<br>")}</p>`;
  }).join("");
  return text.replace(/@@CODEBLOCK(\d+)@@/g, (_, index) => blocks[Number(index)]);
}

function loadMermaid() {
  if (!mermaidLoader) {
    mermaidLoader = import(MERMAID_CDN).then((mod) => {
      const mermaid = mod.default;
      mermaid.initialize({
        startOnLoad: false,
        theme: state.theme === "light" ? "default" : "dark",
        securityLevel: "strict",
        fontFamily: 'Inter, "Microsoft YaHei", sans-serif',
        themeVariables: state.theme === "light"
          ? { background: "#ffffff", primaryTextColor: "#1a1c20", lineColor: "#6eae1a" }
          : { darkMode: true, background: "#111214", primaryTextColor: "#f4f4f5", lineColor: "#b7f34a" },
      });
      return mermaid;
    }).catch((error) => {
      mermaidLoader = null;
      throw error;
    });
  }
  return mermaidLoader;
}

function fitMermaidSvg(node) {
  const svg = node.querySelector("svg");
  if (!svg || !svg.viewBox || !svg.viewBox.baseVal) return;
  const box = svg.viewBox.baseVal;
  if (!box.width || !box.height) return;
  svg.removeAttribute("height");
  svg.setAttribute("width", "100%");
  svg.style.width = "100%";
  svg.style.height = "auto";
  svg.style.aspectRatio = `${box.width} / ${box.height}`;
}

async function hydrateMermaid(root) {
  const nodes = [...(root || document).querySelectorAll(".mermaid")].filter((node) => !node.getAttribute("data-processed"));
  if (!nodes.length) return;
  try {
    const mermaid = await loadMermaid();
    await mermaid.run({ nodes, suppressErrors: true });
    nodes.forEach(fitMermaidSvg);
  } catch {
    nodes.forEach((node) => node.classList.add("mermaid-fallback"));
  }
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

function clampSidebarWidth(value) {
  return Math.min(SIDEBAR_WIDTH_MAX, Math.max(SIDEBAR_WIDTH_MIN, value));
}

function clampInspectorWidth(value) {
  return Math.min(INSPECTOR_WIDTH_MAX, Math.max(INSPECTOR_WIDTH_MIN, value));
}

function readOpenWidth(cssVar, fallback) {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(cssVar).trim();
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function applyOpenWidth(cssVar, width, clampFn) {
  document.documentElement.style.setProperty(cssVar, `${clampFn(width)}px`);
}

function persistOpenWidth(storageKey, width, clampFn) {
  try {
    localStorage.setItem(storageKey, String(clampFn(width)));
  } catch { /* 隐私模式下 localStorage 不可用，忽略 */ }
}

function initPanelResize({
  handleId,
  cssVar,
  storageKey,
  defaultWidth,
  clampFn,
  panelName,
  direction,
}) {
  const handle = el(handleId);
  if (!handle) return;

  let savedWidth = defaultWidth;
  try {
    const stored = Number.parseInt(localStorage.getItem(storageKey) || "", 10);
    if (Number.isFinite(stored)) savedWidth = stored;
  } catch { /* ignore */ }
  applyOpenWidth(cssVar, savedWidth, clampFn);

  let dragging = false;
  let startX = 0;
  let startWidth = savedWidth;

  const canResize = () => {
    if (document.body.classList.contains(`hide-${panelName}`)) return false;
    if (!isNarrow(panelName)) return true;
    return panelNode(panelName).classList.contains("open");
  };

  const stopDragging = (event) => {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("panel-resizing");
    persistOpenWidth(storageKey, readOpenWidth(cssVar, defaultWidth), clampFn);
    if (event?.pointerId != null && handle.hasPointerCapture(event.pointerId)) {
      handle.releasePointerCapture(event.pointerId);
    }
  };

  handle.addEventListener("pointerdown", (event) => {
    if (!canResize() || event.button !== 0) return;
    dragging = true;
    startX = event.clientX;
    startWidth = readOpenWidth(cssVar, defaultWidth);
    document.body.classList.add("panel-resizing");
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  handle.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const delta = (event.clientX - startX) * direction;
    applyOpenWidth(cssVar, startWidth + delta, clampFn);
  });

  handle.addEventListener("pointerup", stopDragging);
  handle.addEventListener("pointercancel", stopDragging);
  window.addEventListener("blur", () => stopDragging());
}

function initPanelResizes() {
  initPanelResize({
    handleId: "sidebarResizeHandle",
    cssVar: "--sidebar-open-width",
    storageKey: SIDEBAR_WIDTH_KEY,
    defaultWidth: SIDEBAR_WIDTH_DEFAULT,
    clampFn: clampSidebarWidth,
    panelName: "sidebar",
    direction: 1,
  });
  initPanelResize({
    handleId: "inspectorResizeHandle",
    cssVar: "--inspector-open-width",
    storageKey: INSPECTOR_WIDTH_KEY,
    defaultWidth: INSPECTOR_WIDTH_DEFAULT,
    clampFn: clampInspectorWidth,
    panelName: "inspector",
    direction: -1,
  });
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
    <button class="session-item ${!state.draft && item.session_id === state.activeId ? "active" : ""}" data-session="${item.session_id}">
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

function showDraftConversation() {
  state.activeId = null;
  state.draft = true;
  state.currentSession = { session_id: null, title: "新对话", messages: [] };
  el("chatTitle").textContent = "新对话";
  welcome.classList.remove("hidden");
  messages.innerHTML = "";
  closeArtifactView();
  syncArtifacts([]);
  renderSessionList();
}

function newSession() {
  if (state.sending) return;
  showDraftConversation();
  input.focus();
  if (isNarrow("sidebar")) setPanel("sidebar", false);
}

async function loadSession(sessionId) {
  if (state.sending) return;
  try {
    const session = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
    state.activeId = session.session_id;
    state.draft = false;
    renderSessionList();
    renderConversation(session);
    if (isNarrow("sidebar")) setPanel("sidebar", false);
  } catch (error) { showToast(error.message); }
}

function renderConversation(session) {
  state.currentSession = session;
  state.draft = !session.session_id;
  el("chatTitle").textContent = session.title || "新对话";
  const items = session.messages || [];
  welcome.classList.toggle("hidden", items.length > 0);
  messages.innerHTML = items.map(messageHtml).join("");
  bindArtifactCards();
  closeArtifactView();
  syncArtifacts(items);
  hydrateMermaid(messages).then(scrollToBottom);
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
      <span class="artifact-card-icon ${artifact.kind === "chart" ? "chart" : artifact.kind === "document" ? "document" : "report"}">${artifact.kind === "chart" ? "◫" : artifact.kind === "document" ? "📄" : "▤"}</span>
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
      <span class="artifact-row-icon ${artifact.kind === "chart" ? "chart" : artifact.kind === "document" ? "document" : "report"}">${artifact.kind === "chart" ? "◫" : artifact.kind === "document" ? "📄" : "▤"}</span>
      <span class="artifact-row-copy">
        <strong>${escapeHtml(artifact.title || "会话产物")}</strong>
        <small>${escapeHtml(artifact.subtitle || (artifact.kind === "chart" ? "统计图" : "报告"))}</small>
      </span>
    </button>`).join("");
  container.querySelectorAll("[data-artifact-path]").forEach((button) => {
    button.addEventListener("click", () => openArtifact(button.dataset.artifactPath));
  });
}

async function renderPdfArtifact(path, name, container) {
  if (state.artifactPdfUrl) {
    URL.revokeObjectURL(state.artifactPdfUrl);
    state.artifactPdfUrl = null;
  }
  const response = await fetch(`/api/artifact/file?path=${encodeURIComponent(path)}`);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.error) detail = payload.error;
    } catch { /* ignore */ }
    throw new Error(`PDF 加载失败：${detail}`);
  }
  const blob = await response.blob();
  if (!blob.size) throw new Error("PDF 文件为空");
  const objectUrl = URL.createObjectURL(blob);
  state.artifactPdfUrl = objectUrl;
  container.innerHTML = `
    <embed class="artifact-pdf-frame" type="application/pdf" src="${objectUrl}" title="${escapeHtml(name)}" />
    <a class="artifact-pdf-open" href="${objectUrl}" target="_blank" rel="noopener noreferrer">在新标签页打开 PDF</a>
  `;
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
    const kind = data.kind === "chart" ? "统计图" : data.kind === "document" ? "文档" : "报告";
    const formatLabel = data.format ? data.format.toUpperCase() : "";
    el("artifactMeta").textContent = [kind, formatLabel, formatSize(data.size), relativeTime(data.modified_at)]
      .filter(Boolean)
      .join(" · ");
    const body = el("artifactBody");
    if (data.preview_type === "pdf") {
      await renderPdfArtifact(path, data.name, body);
    } else if (data.preview_type === "docx_html") {
      body.innerHTML = `<div class="artifact-docx-preview">${data.preview_html || ""}</div>`;
    } else {
      body.innerHTML = renderMarkdown(data.content || "");
      hydrateMermaid(body);
    }
  } catch (error) {
    el("artifactName").textContent = "读取失败";
    el("artifactBody").innerHTML = `<div class="artifact-loading">${escapeHtml(error.message)}</div>`;
    showToast(error.message);
  }
}

function closeArtifactView() {
  state.openArtifactPath = null;
  if (state.artifactPdfUrl) {
    URL.revokeObjectURL(state.artifactPdfUrl);
    state.artifactPdfUrl = null;
  }
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

function readTheme() {
  try {
    return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function applyTheme(theme, { persist = true, rerender = false } = {}) {
  state.theme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = state.theme;
  document.documentElement.style.colorScheme = state.theme;
  if (persist) {
    try { localStorage.setItem(THEME_KEY, state.theme); } catch { /* 隐私模式忽略 */ }
  }
  const summary = el("settingsThemeSummary");
  if (summary) summary.textContent = state.theme === "light" ? "亮色背景" : "深色背景";
  document.querySelectorAll("[data-theme-option]").forEach((button) => {
    button.classList.toggle("active", button.dataset.themeOption === state.theme);
  });
  mermaidLoader = null;
  if (rerender && state.currentSession) {
    const artifact = state.openArtifactPath;
    renderConversation(state.currentSession);
    if (artifact) openArtifact(artifact);
  }
}

function initTheme() {
  applyTheme(readTheme(), { persist: false });
}

function openSettings(page = "home") {
  el("settingsModal").classList.remove("hidden");
  showSettingsPage(page);
}

function showSettingsPage(page) {
  el("settingsHome").classList.toggle("hidden", page !== "home");
  el("settingsModels").classList.toggle("hidden", page !== "models");
  el("settingsKnowledge").classList.toggle("hidden", page !== "knowledge");
  el("settingsAppearance").classList.toggle("hidden", page !== "appearance");
  el("settingsTitle").textContent = page === "models"
    ? "模型"
    : page === "knowledge"
      ? "知识库"
      : page === "appearance"
        ? "外观"
        : "设置";
  el("settingsSubtitle").textContent = page === "models"
    ? "OpenAI 兼容协议 · 保存在本机数据目录"
    : page === "knowledge"
      ? "自定义文档 · 写入 pgvector 语义检索"
      : page === "appearance"
        ? "背景主题保存在本机浏览器"
        : "MyFitness Agent · 本地运行";
  if (page === "models") loadModels();
  if (page === "knowledge") loadKnowledge();
  if (page === "appearance") applyTheme(state.theme, { persist: false });
}

/* -------------------------------------------------------------- 知识库 */

function resetKnowledgeForm() {
  state.editingKnowledgeId = null;
  el("knowledgeFormId").value = "";
  el("knowledgeFormTitle").textContent = "添加知识";
  el("knowledgeFormName").value = "";
  el("knowledgeFormContent").value = "";
  el("knowledgeFormCancel").classList.add("hidden");
  el("knowledgeFormDelete").classList.add("hidden");
  resetKnowledgeFileInput();
  document.querySelectorAll(".knowledge-card").forEach((node) => node.classList.remove("active"));
}

function resetKnowledgeFileInput() {
  const input = el("knowledgeFileInput");
  if (input) input.value = "";
  const hint = el("knowledgeFileHint");
  if (hint) hint.textContent = "支持 md / pdf / doc / docx / txt / html，解析后可再编辑保存";
}

function renderKnowledgeSummary(data) {
  const summary = el("settingsKnowledgeSummary");
  if (!summary) return;
  const count = data?.entry_count ?? 0;
  summary.textContent = count ? `${count} 条 · ${data.indexed_chunks ?? 0} 向量块` : "管理 RAG 参考文档";
  el("knowledgeEntryCount").textContent = String(count);
  el("knowledgeChunkCount").textContent = String(data?.indexed_chunks ?? 0);
  const status = el("knowledgeStatus");
  if (!data?.available) {
    status.textContent = data?.embedding_configured
      ? "RAG 未就绪：请确认 PostgreSQL 已安装 pgvector 并运行 myfitness rag init"
      : "RAG 未就绪：请配置 Embedding API（DeepSeek 等聊天模型没有 /embeddings，需单独设置 EMBEDDING_BASE_URL）";
    return;
  }
  status.textContent = `RAG 已就绪 · 总向量块 ${data.total_chunks ?? 0}`;
}

function renderKnowledgeList() {
  const container = el("knowledgeList");
  if (!state.knowledge.length) {
    container.innerHTML = '<div class="knowledge-empty">还没有知识条目。可在下方添加训练原则、饮食偏好等参考内容。</div>';
    return;
  }
  container.innerHTML = state.knowledge.map((entry) => `
    <article class="knowledge-card ${Number(state.editingKnowledgeId) === Number(entry.id) ? "active" : ""}" data-knowledge-id="${entry.id}">
      <strong>${escapeHtml(entry.title)}${entry.kind === "memory" ? '<span class="knowledge-kind">长期记忆</span>' : ""}</strong>
      <p>${escapeHtml(entry.preview || "")}</p>
      <small>更新于 ${escapeHtml(formatLocalTime(entry.updated_at))}</small>
    </article>
  `).join("");
  container.querySelectorAll("[data-knowledge-id]").forEach((node) => {
    node.addEventListener("click", () => editKnowledge(Number(node.dataset.knowledgeId)));
  });
}

async function loadKnowledgeSummary() {
  try {
    const data = await api("/api/knowledge");
    renderKnowledgeSummary(data);
  } catch {
    /* 设置首页摘要失败时静默忽略 */
  }
}

async function loadKnowledge() {
  el("knowledgeList").innerHTML = '<div class="knowledge-empty">正在加载知识库…</div>';
  try {
    const data = await api("/api/knowledge");
    state.knowledge = data.entries || [];
    renderKnowledgeSummary(data);
    renderKnowledgeList();
  } catch (error) {
    el("knowledgeList").innerHTML = `<div class="knowledge-empty">${escapeHtml(error.message)}</div>`;
    showToast(error.message);
  }
}

function editKnowledge(entryId) {
  const entry = state.knowledge.find((item) => Number(item.id) === Number(entryId));
  if (!entry) return;
  state.editingKnowledgeId = entry.id;
  el("knowledgeFormId").value = String(entry.id);
  el("knowledgeFormTitle").textContent = "编辑知识";
  el("knowledgeFormName").value = entry.title;
  el("knowledgeFormContent").value = entry.content;
  el("knowledgeFormCancel").classList.remove("hidden");
  el("knowledgeFormDelete").classList.remove("hidden");
  resetKnowledgeFileInput();
  renderKnowledgeList();
}

async function parseKnowledgeFile(event) {
  const input = event.target;
  const file = input.files && input.files[0];
  if (!file) return;
  const hint = el("knowledgeFileHint");
  hint.textContent = `正在解析 ${file.name}…`;
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const response = await fetch("/api/knowledge/parse", { method: "POST", body: form });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `解析失败 (${response.status})`);
    const titleInput = el("knowledgeFormName");
    if (!state.editingKnowledgeId || !titleInput.value.trim()) {
      titleInput.value = data.title || "";
    }
    el("knowledgeFormContent").value = data.content || "";
    const extra = data.truncated ? "（已截断到长度上限）" : "";
    hint.textContent = `已解析 ${data.filename} · ${data.format.toUpperCase()} · ${data.char_count} 字${extra}，请核对后保存`;
    showToast(data.truncated ? "文件已解析并截断，请核对后保存" : "文件已解析，请核对后保存");
  } catch (error) {
    hint.textContent = "解析失败，请改用 md / pdf / docx / txt，或直接粘贴文本";
    showToast(error.message);
  } finally {
    input.value = "";
  }
}

async function submitKnowledgeForm(event) {
  event.preventDefault();
  const payload = {
    title: el("knowledgeFormName").value.trim(),
    content: el("knowledgeFormContent").value.trim(),
  };
  const saveButton = el("knowledgeFormSave");
  saveButton.disabled = true;
  try {
    if (state.editingKnowledgeId) {
      await api(`/api/knowledge/${state.editingKnowledgeId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
      });
      showToast("知识条目已更新并重新索引");
    } else {
      await api("/api/knowledge", { method: "POST", body: JSON.stringify(payload) });
      showToast("已添加到知识库");
    }
    resetKnowledgeForm();
    await loadKnowledge();
  } catch (error) {
    showToast(error.message);
  } finally {
    saveButton.disabled = false;
  }
}

async function deleteKnowledgeEntry() {
  if (!state.editingKnowledgeId) return;
  if (!window.confirm("确定删除这条知识？相关向量块也会一并移除。")) return;
  try {
    await api(`/api/knowledge/${state.editingKnowledgeId}`, { method: "DELETE" });
    showToast("已删除");
    resetKnowledgeForm();
    await loadKnowledge();
  } catch (error) {
    showToast(error.message);
  }
}

async function reindexKnowledge(full = false) {
  try {
    const data = await api("/api/knowledge/reindex", {
      method: "POST",
      body: JSON.stringify({ full }),
    });
    showToast(full ? "全部 RAG 索引重建完成" : "知识库索引重建完成");
    await loadKnowledge();
    return data;
  } catch (error) {
    showToast(error.message);
  }
}

function formatLocalTime(value) {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString("zh-CN", { hour12: false });
  } catch {
    return value;
  }
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

function updateProgress(text) {
  const span = document.querySelector("#progressMessage span");
  if (span && text) span.textContent = text;
}

function createTypewriter(contentEl) {
  let shown = "";
  let queue = "";
  let timer = 0;
  let settle = null;

  function paint(streaming) {
    contentEl.innerHTML = renderMarkdown(shown);
    contentEl.classList.toggle("streaming", streaming);
    scrollToBottom();
  }

  function tick() {
    timer = 0;
    if (!queue) {
      if (settle) {
        const done = settle;
        settle = null;
        done();
      }
      return;
    }
    const step = queue.length <= 16 ? queue.length : Math.min(8, Math.ceil(queue.length / 18) + 2);
    shown += queue.slice(0, step);
    queue = queue.slice(step);
    paint(true);
    timer = window.setTimeout(tick, 16);
  }

  return {
    push(text) {
      if (!text) return;
      queue += text;
      if (!timer) timer = window.setTimeout(tick, 0);
    },
    finish() {
      if (!queue && !timer) {
        contentEl.classList.remove("streaming");
        return Promise.resolve();
      }
      return new Promise((resolve) => {
        const prev = settle;
        settle = () => {
          if (prev) prev();
          contentEl.classList.remove("streaming");
          resolve();
        };
        if (!timer) timer = window.setTimeout(tick, 0);
      });
    },
  };
}

function consumeSse(buffer) {
  const parts = buffer.split(/\r?\n\r?\n/);
  const rest = parts.pop() ?? "";
  const events = [];
  for (const block of parts) {
    if (!block.trim()) continue;
    let event = "message";
    const dataLines = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) continue;
    try {
      events.push({ event, data: JSON.parse(dataLines.join("\n")) });
    } catch {
      events.push({ event: "error", data: { error: "流式数据解析失败" } });
    }
  }
  return { events, rest };
}

function ensureAssistantStream() {
  el("progressMessage")?.remove();
  let node = el("streamingMessage");
  if (node) return node.querySelector(".message-content");
  messages.insertAdjacentHTML("beforeend", `
    <article class="message assistant" id="streamingMessage">
      <div class="message-avatar">${icons.agent}</div>
      <div class="message-content streaming"></div>
    </article>`);
  return el("streamingMessage").querySelector(".message-content");
}

async function readSessionStream(text) {
  const response = await fetch("/api/sessions/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: text,
      ...(state.activeId ? { session_id: state.activeId } : {}),
    }),
  });
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `请求失败 (${response.status})`);
  }
  if (!contentType.includes("text/event-stream") || !response.body) {
    throw new Error("服务器未返回流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let typewriter = null;
  let finalPayload = null;
  let streamError = null;

  const handleEvent = async (event, data) => {
    if (event === "progress") {
      updateProgress(data.text);
      return;
    }
    if (event === "session" && data.session_id) {
      state.activeId = data.session_id;
      state.draft = false;
      if (data.title) el("chatTitle").textContent = data.title;
      refreshSessions().catch(() => {});
      return;
    }
    if (event === "delta") {
      if (!typewriter) typewriter = createTypewriter(ensureAssistantStream());
      typewriter.push(data.text || "");
      return;
    }
    if (event === "done") {
      finalPayload = data;
      return;
    }
    if (event === "error") {
      streamError = new Error(data.error || "流式输出失败");
    }
  };

  while (!streamError) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const consumed = consumeSse(buffer);
    buffer = consumed.rest;
    for (const item of consumed.events) {
      await handleEvent(item.event, item.data || {});
      if (streamError) break;
    }
  }
  if (buffer.trim() && !streamError) {
    const consumed = consumeSse(`${buffer}\n\n`);
    for (const item of consumed.events) {
      await handleEvent(item.event, item.data || {});
    }
  }
  if (streamError) throw streamError;
  if (typewriter) await typewriter.finish();
  if (!finalPayload) throw new Error("会话中断，未收到完整回复");
  return finalPayload;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || state.sending) return;
  state.sending = true;
  sendButton.disabled = true;
  input.value = "";
  resizeInput();
  welcome.classList.add("hidden");
  messages.insertAdjacentHTML("beforeend", messageHtml({ role: "user", content: text }));
  showProgress();
  try {
    const result = await readSessionStream(text);
    renderConversation(result);
    await refreshSessions();
  } catch (error) {
    el("progressMessage")?.remove();
    el("streamingMessage")?.remove();
    messages.insertAdjacentHTML("beforeend", `<article class="message assistant"><div class="message-avatar">!</div><div class="message-content"><p>处理失败：${escapeHtml(error.message)}</p></div></article>`);
    showToast(error.message);
    if (state.draft) state.activeId = null;
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
  document.querySelectorAll(".settings-back").forEach((button) => {
    button.addEventListener("click", () => showSettingsPage("home"));
  });
  el("settingsHome").querySelectorAll("[data-settings-page]").forEach((button) => {
    button.addEventListener("click", () => showSettingsPage(button.dataset.settingsPage));
  });
  el("settingsAppearance").querySelectorAll("[data-theme-option]").forEach((button) => {
    button.addEventListener("click", () => applyTheme(button.dataset.themeOption, { rerender: true }));
  });
  el("modelForm").addEventListener("submit", submitModelForm);
  el("modelFormTest").addEventListener("click", testModelConnection);
  el("modelFormCancel").addEventListener("click", resetModelForm);
  el("modelFormDelete").addEventListener("click", () => deleteModel(state.editingModelId));
  el("modelSelect").addEventListener("change", (event) => activateModel(event.target.value));

  el("knowledgeForm").addEventListener("submit", submitKnowledgeForm);
  el("knowledgeFormCancel").addEventListener("click", resetKnowledgeForm);
  el("knowledgeFormDelete").addEventListener("click", deleteKnowledgeEntry);
  el("knowledgeFileInput").addEventListener("change", parseKnowledgeFile);
  el("knowledgeReindex").addEventListener("click", () => reindexKnowledge(false));
  el("knowledgeReindexAll").addEventListener("click", () => reindexKnowledge(true));

  document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { input.value = button.dataset.prompt; resizeInput(); input.focus(); }));
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); newSession(); }
    if (event.key !== "Escape") return;
    if (!el("settingsModal").classList.contains("hidden")) {
      if (el("settingsHome").classList.contains("hidden")) showSettingsPage("home");
      else el("settingsModal").classList.add("hidden");
      return;
    }
    el("taskModal").classList.add("hidden");
  });
}

async function init() {
  initTheme();
  initPanelResizes();
  initLayout();
  bindEvents();
  resetModelForm();
  renderArtifactList();
  try {
    await api("/api/health");
    el("connectionStatus").textContent = "已连接 · JSON 持久化";
    await Promise.all([refreshSessions(), loadTasks(false), loadModels(), loadKnowledgeSummary()]);
    showDraftConversation();
    input.focus();
  } catch (error) {
    el("connectionStatus").textContent = "连接失败";
    showToast(error.message);
  }
}

init();
