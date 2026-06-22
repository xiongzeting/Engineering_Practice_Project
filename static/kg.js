// 知识图谱（KG）页面：用 vis-network 渲染节点-边图 + 节点详情侧栏。
// 入口：window.initKgPage()，由 app.js 的 routePage() 在用户切到 /kg 路由时调用。
//
// 渲染策略：
// - 优先调 vis-network（CDN 加载）画力导向图；
// - 若 vis-network 没加载到 / 容器尺寸为 0（首次切页时常见），优雅
//   回退成 HTML 列表 + chip 形式，保证数据始终可见。

// vis-network Network 实例（每次重渲染都会 destroy 重建）。
let kgNetwork = null;
// KG 本体：所有节点（id/name/stage/prerequisites/...）。
let kgNodesData = [];
// KG 本体：所有边（from/to）。
let kgEdgesData = [];
// 当前用户的掌握度，按 node_id 索引：{state, accuracy, attempts}。
let kgMasteryMap = {};

// 四种掌握状态对应的颜色（与 compute_user_mastery 的 state 一一对应）。
const KG_STATE_COLORS = {
  gray: "#95a5a6",     // 未测：学生从未做过该节点的题
  green: "#27ae60",    // 掌握：accuracy ≥ 0.8
  yellow: "#f39c12",   // 待巩固：0.5 ≤ accuracy < 0.8
  red: "#c0392b",      // 薄弱：accuracy < 0.5
};

// 状态 → 中文标签（用于侧栏 pill）。
const KG_STATE_LABELS = {
  gray: "未测",
  green: "掌握",
  yellow: "待巩固",
  red: "薄弱",
};

// 学段 key → 中文标签。
const KG_STAGE_LABELS = {
  primary: "小学",
  middle: "初中",
  high: "高中",
};

// 学段排序权重：渲染时按 primary → middle → high 排。
const KG_STAGE_ORDER = { primary: 0, middle: 1, high: 2 };

async function initKgPage() {
  // KG 页入口。先等 vis-network 加载完成（最多 4 秒），再绑定事件、拉数据。
  console.log("[KG] initKgPage called; window.vis =", typeof window.vis);
  if (typeof window.vis === "undefined") {
    // CDN 可能整段失败：轮询 200ms 一次，4 秒还没到就报错而不是无限轮询。
    let waited = 0;
    const timer = setInterval(() => {
      waited += 200;
      if (typeof window.vis !== "undefined") {
        clearInterval(timer);
        initKgPage();
      } else if (waited >= 4000) {
        clearInterval(timer);
        showKgFatalError("知识图谱脚本加载失败（vis-network CDN 不可达），请检查网络后刷新。");
      }
    }, 200);
    return;
  }
  // 绑定刷新按钮 + 学段筛选下拉。
  const refreshBtn = document.getElementById("kgRefreshBtn");
  if (refreshBtn) refreshBtn.onclick = () => loadKgData();
  const filter = document.getElementById("kgStageFilter");
  if (filter) filter.onchange = () => renderKgNetwork();
  // 等两帧 RAF：让 pane 先变得可见、容器有非零尺寸，再让 vis 测量。
  // 没这两帧的话 vis 会按 0×0 渲染然后卡死。
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      loadKgData();
    });
  });
}

function showKgFatalError(message) {
  // 把错误信息同时显示在侧栏 + 画布上，让用户立刻看到。
  const panel = document.getElementById("kgSidePanel");
  const network = document.getElementById("kgNetwork");
  if (panel) {
    panel.innerHTML = `<div class="kg-detail"><p class="kg-detail-desc" style="color:#c0392b">${message}</p></div>`;
  }
  if (network) {
    network.innerHTML = `<div style="display:grid;place-items:center;height:100%;color:#7f8c8d;padding:1rem;text-align:center">${message}</div>`;
  }
  if (typeof setStatus === "function") setStatus(message, "error");
}

async function loadKgData() {
  // 并发拉本体（/api/kg/ontology）和当前用户掌握度（/api/kg/mastery）。
  try {
    console.log("[KG] loadKgData fetching /api/kg/ontology and /api/kg/mastery");
    const [ontoResp, mastResp] = await Promise.all([
      apiFetchJson("/api/kg/ontology"),
      apiFetchJson("/api/kg/mastery"),
    ]);
    console.log("[KG] ontology HTTP", ontoResp.status, "mastery HTTP", mastResp.status);
    if (!ontoResp.ok) throw new Error(`ontology ${ontoResp.status}`);
    if (!mastResp.ok) throw new Error(`mastery ${mastResp.status}`);
    const onto = await ontoResp.json();
    const mast = await mastResp.json();
    kgNodesData = onto.nodes || [];
    kgEdgesData = onto.edges || [];
    // mastery 是数组，转成 {id: mastery_obj} 方便查询。
    kgMasteryMap = {};
    for (const m of mast.nodes || []) kgMasteryMap[m.id] = m;
    console.log("[KG] loaded", kgNodesData.length, "nodes,", kgEdgesData.length, "edges");
    if (!kgNodesData.length) {
      showKgFatalError("本体数据为空（data/kg_ontology.yaml 未加载或解析失败）。");
      return;
    }
    renderKgNetwork();
  } catch (e) {
    console.error("[KG] loadKgData failed:", e);
    showKgFatalError(`加载知识图谱失败: ${e.message}`);
  }
}

function renderKgNetwork() {
  // 主渲染函数：按学段筛选 → vis-network 画图 → 追加 HTML 列表兜底。
  const container = document.getElementById("kgNetwork");
  if (!container) {
    console.error("[KG] renderKgNetwork: #kgNetwork not found");
    return;
  }
  // 下拉筛选：all/primary/middle/high。
  const stageFilter =
    document.getElementById("kgStageFilter")?.value || "all";

  // 先按学段筛选，再按学段顺序 + id 排序，确保每次渲染节点顺序稳定。
  const visibleNodes = kgNodesData.filter(
    (n) => stageFilter === "all" || n.stage === stageFilter
  );
  visibleNodes.sort(
    (a, b) =>
      (KG_STAGE_ORDER[a.stage] ?? 9) - (KG_STAGE_ORDER[b.stage] ?? 9) ||
      a.id.localeCompare(b.id)
  );
  const visibleIds = new Set(visibleNodes.map((n) => n.id));

  // 诊断信息：数据是否到位、容器尺寸、vis 是否就绪。
  const w = container.clientWidth;
  const h = container.clientHeight;
  const visReady = typeof window.vis !== "undefined" && !!window.vis.Network;
  console.log(`[KG] renderKgNetwork: vis=${visReady} container=${w}x${h} nodes=${visibleNodes.length}`);

  // Fallback：vis 没加载或容器尺寸为 0 时，画一个 HTML 列表兜底。
  if (!visReady || w === 0 || h === 0) {
    console.warn("[KG] falling back to list mode");
    const fallback = renderKgFallbackList(visibleNodes);
    container.appendChild(fallback);
    return;
  }

  // 节点按学段分列：primary=第0列、middle=第1列、high=第2列。
  // 这样视觉上"小学→初中→高中"是从左到右的 progression。
  const stageCols = { primary: 0, middle: 1, high: 2 };
  const stageGroups = { primary: [], middle: [], high: [] };
  for (const n of visibleNodes) {
    const stage = stageGroups[n.stage] ? n.stage : "primary";
    stageGroups[stage].push(n);
  }
  // 列宽 / 行高：手动给每个节点算 (x, y)，禁用物理引擎，避免力导向布局抖动。
  const colWidth = 260;
  const rowHeight = 95;

  // 节点 DataSet：vis 要求每个节点有 id + label + style。
  const nodes = new window.vis.DataSet(
    visibleNodes.map((n) => {
      const m = kgMasteryMap[n.id] || { state: "gray" };
      const color = KG_STATE_COLORS[m.state] || KG_STATE_COLORS.gray;
      const col = stageCols[n.stage] ?? 0;
      const rowIdx = stageGroups[n.stage].indexOf(n);
      return {
        id: n.id,
        label: `${n.id}\n${n.name}`,  // 节点显示两行：编号 + 名称
        x: col * colWidth,
        y: rowIdx * rowHeight,
        physics: false,                // 关闭物理：手工布局，不要乱跑
        fixed: { x: true, y: false },  // x 固定（保列对齐），y 可拖（用户可微调）
        color: { background: color, border: "#2c3e50", highlight: { background: color, border: "#34495e" } },
        font: { color: "#fff", size: 13, face: "PingFang SC" },
      };
    })
  );

  // 边 DataSet：只保留两端节点都可见的边（按学段筛选后可能有边需要隐藏）。
  const edges = new window.vis.DataSet(
    kgEdgesData
      .filter((e) => visibleIds.has(e.from) && visibleIds.has(e.to))
      .map((e, i) => ({
        id: `e${i}`,
        from: e.from,
        to: e.to,
        arrows: "to",
        // cubicBezier + horizontal：让边从左侧节点平滑弯到右侧节点。
        smooth: { enabled: true, type: "cubicBezier", forceDirection: "horizontal", roundness: 0.4 },
        color: { color: "#bdc3c7", highlight: "#34495e" },
      }))
  );

  // 全局选项：节点为方框、固定物理、可悬停。
  const options = {
    autoResize: false,
    physics: { enabled: false },
    layout: { hierarchical: { enabled: false }, randomSeed: 42 },
    nodes: { shape: "box", margin: 12, borderWidth: 2, widthConstraint: { maximum: 200 } },
    edges: { arrows: { to: { scaleFactor: 0.6 } } },
    interaction: { hover: true, tooltipDelay: 100, navigationButtons: false, keyboard: false },
  };

  // destroy 旧实例再建新的，避免重复绑定事件。
  if (kgNetwork) kgNetwork.destroy();
  kgNetwork = new window.vis.Network(container, { nodes, edges }, options);
  // 点击节点：在侧栏展示详情。
  kgNetwork.on("click", (params) => {
    const id = params.nodes[0];
    if (!id) return;
    showNodeDetail(id);
  });
  // 一帧后再 redraw + fit。多次 fit 曾导致画布异常放大，所以只做一次。
  requestAnimationFrame(() => {
    if (!kgNetwork) return;
    kgNetwork.redraw();
    kgNetwork.fit({ animation: false });
  });

  // 在画布下方追加一份"节点列表（按学段分组的 chip）"——
  // 即使 vis 渲染成功也保留这份列表，因为有些用户更习惯扫读 chip。
  // 列表挂在 pane（不是画布）上，避免把画布挤大。
  const pane = container.closest(".account-pane") || container.parentElement?.parentElement;
  const existingList = pane?.querySelector("[data-kg-listfallback]");
  if (existingList) existingList.remove();
  const listWrap = document.createElement("div");
  listWrap.setAttribute("data-kg-listfallback", "1");
  listWrap.style.cssText = "margin-top:16px;padding:12px;background:#fff;border:1px solid #e5e7eb;border-radius:8px";
  const listTitle = document.createElement("div");
  listTitle.style.cssText = "font-weight:600;margin-bottom:8px;color:#2c3e50";
  listTitle.textContent = `节点列表（${visibleNodes.length} 个，按学段分组）`;
  listWrap.appendChild(listTitle);
  // 每个学段一行：标题 + chip 行。
  for (const stage of ["primary", "middle", "high"]) {
    const group = stageGroups[stage];
    if (!group?.length) continue;
    const row = document.createElement("div");
    row.style.cssText = "margin-bottom:8px";
    const head = document.createElement("div");
    head.style.cssText = "font-size:12px;color:#7f8c8d;margin-bottom:4px";
    head.textContent = KG_STAGE_LABELS[stage] || stage;
    row.appendChild(head);
    const chips = document.createElement("div");
    chips.style.cssText = "display:flex;flex-wrap:wrap;gap:6px";
    for (const n of group) {
      const m = kgMasteryMap[n.id] || { state: "gray" };
      const color = KG_STATE_COLORS[m.state] || KG_STATE_COLORS.gray;
      const chip = document.createElement("span");
      chip.style.cssText = `display:inline-block;padding:4px 10px;border-radius:12px;color:#fff;cursor:pointer;font-size:12px;background:${color}`;
      chip.textContent = `${n.id} · ${n.name}`;
      // 点 chip 也能弹出节点详情，与点击 vis 节点体验一致。
      chip.onclick = () => showNodeDetail(n.id);
      chips.appendChild(chip);
    }
    row.appendChild(chips);
    listWrap.appendChild(row);
  }
  if (pane) pane.appendChild(listWrap);
}

function renderKgFallbackList(visibleNodes) {
  // vis-network 渲染不了时的纯 HTML 列表兜底：自适应 grid + 卡片。
  // 每张卡片背景色仍然反映掌握度，点击也能看详情。
  const wrap = document.createElement("div");
  wrap.style.cssText =
    "padding:16px;overflow:auto;height:100%;background:#fafbfc";
  const title = document.createElement("div");
  title.style.cssText = "font-weight:600;margin-bottom:8px;color:#2c3e50";
  title.textContent = `知识图谱（列表回退模式，共 ${visibleNodes.length} 个节点）`;
  wrap.appendChild(title);
  const hint = document.createElement("div");
  hint.style.cssText = "font-size:12px;color:#7f8c8d;margin-bottom:12px";
  hint.textContent = "力导向图未能渲染（容器尺寸为 0 或 vis-network 未加载），暂时以列表显示。";
  wrap.appendChild(hint);
  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px";
  for (const n of visibleNodes) {
    const m = kgMasteryMap[n.id] || { state: "gray" };
    const color = KG_STATE_COLORS[m.state] || KG_STATE_COLORS.gray;
    const card = document.createElement("div");
    card.style.cssText =
      `padding:8px 10px;border-radius:6px;color:#fff;cursor:pointer;background:${color}`;
    card.textContent = `${n.id} · ${n.name}`;
    card.onclick = () => showNodeDetail(n.id);
    grid.appendChild(card);
  }
  wrap.appendChild(grid);
  return wrap;
}

function showNodeDetail(nodeId) {
  // 在右侧侧栏渲染单个节点的详情：状态 pill + 元数据 + 描述 + 错因标签。
  const panel = document.getElementById("kgSidePanel");
  if (!panel) return;
  const node = kgNodesData.find((n) => n.id === nodeId);
  if (!node) return;
  const m = kgMasteryMap[nodeId] || { state: "gray", attempts: 0 };

  // 前置依赖：把前置节点的 id 翻译成"id·name"展示，找不到就只显示 id。
  const prereqLabels = (node.prerequisites || []).map((pid) => {
    const p = kgNodesData.find((n) => n.id === pid);
    return p ? `${pid}·${p.name}` : pid;
  });

  // accuracy 是 [0,1] 小数，展示成百分比；null/undefined 时显示"—"。
  const accuracyLabel =
    m.accuracy !== null && m.accuracy !== undefined
      ? `${(m.accuracy * 100).toFixed(0)}%`
      : "—";

  panel.innerHTML = `
    <div class="kg-detail">
      <div class="kg-detail-head">
        <span class="kg-state-pill state-${m.state}">${KG_STATE_LABELS[m.state]}</span>
        <h3>${node.id} · ${node.name}</h3>
      </div>
      <dl class="kg-detail-meta">
        <dt>学段</dt><dd>${KG_STAGE_LABELS[node.stage] || node.stage} · G${node.grade}</dd>
        <dt>掌握度</dt><dd>${accuracyLabel}</dd>
        <dt>累计题数</dt><dd>${m.attempts || 0}</dd>
        <dt>前置知识点</dt><dd>${prereqLabels.length ? prereqLabels.join("，") : "无"}</dd>
      </dl>
      ${node.description ? `<p class="kg-detail-desc">${node.description}</p>` : ""}
      ${(node.error_type_hints || []).length ? `<div class="kg-tags">${node.error_type_hints.map((t) => `<span class="tag tag-${t}">${t}</span>`).join("")}</div>` : ""}
    </div>
  `;
}

// 暴露给 app.js 调用的唯一入口。
window.initKgPage = initKgPage;
