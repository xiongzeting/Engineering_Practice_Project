// ============================================================
// MathGrade 前端控制器（SPA）
// ------------------------------------------------------------
// 本文件承担单页应用的全部行为：路由切换、API 调用、DOM 渲染。
// 三大块：
//   1. 路由（routePage）：根据 URL pathname 切 page-view，处理角色权限
//      （学生看作业批改 + 个人中心；教师/管理员强制跳到后台）。
//   2. 业务流水线：
//      - 批改流水线（assignmentPages / 问题切分 / 逐步评分）
//      - 错题本（loadWrongAnswers，含逐步 LaTeX 渲染 + 内联笔记）
//      - 知识图谱（依赖 kg.js 的 vis-network 可视化）
//      - 收藏夹 / 读取历史 / 批改历史
//   3. 后台（admin）：三角色共享同一外壳，按角色动态隐藏 tab——
//      admin: 概览/用户管理/班级/基础KG
//      teacher: 班级/班级KG定制（无概览、无全用户）
//
// 约定：
//   - 几乎所有 DOM 查询走 $(id) 这个简写（=document.getElementById）
//   - 所有后端调用走 apiFetchJson(url, opts)，自动带 Authorization header
//     并在 401 时跳登录
//   - 长 HTML 模板用反引号字符串，复杂片段优先 createElement 避免 XSS
//   - 状态消息走 setStatus(text, kind)，会显示在右下角 toast
// ============================================================

const $ = (id) => document.getElementById(id);

const pageTitle = $("pageTitle");
const adminNavLink = $("adminNavLink");
const loginEntryBtn = $("loginEntryBtn");
const userCenterBtn = $("userCenterBtn");
const serviceBadge = $("serviceBadge");
const metricOcrCount = $("metricOcrCount");
const metricGradeCount = $("metricGradeCount");
const metricAvgScore = $("metricAvgScore");
const metricLastTime = $("metricLastTime");

const imageInput = $("image");
const assignmentTitleInput = $("assignmentTitle");
const pageCountBadge = $("pageCountBadge");
const pagesStage = $("pagesStage");
const clearPagesBtn = $("clearPagesBtn");
const reviewStage = $("reviewStage");
const scoreStage = $("scoreStage");
const confirmReviewBtn = $("confirmReviewBtn");
const reOcrBtn = $("reOcrBtn");
const togglePasteBtn = $("togglePasteBtn");
const pastePanel = $("pastePanel");
const imagePreview = $("imagePreview");
const imagePlaceholder = $("imagePlaceholder");
const batchList = $("batchList");
const manualTextInput = $("manualText");
const ocrTextInput = $("ocrText");
const ocrRendered = $("ocrRendered");
const stepPreview = $("stepPreview");
const questionPager = $("questionPager");
const questionPageInfo = $("questionPageInfo");
const prevQuestionBtn = $("prevQuestionBtn");
const nextQuestionBtn = $("nextQuestionBtn");
const deleteQuestionBtn = $("deleteQuestionBtn");
const pagePager = $("pagePager");
const pagePagerInfo = $("pagePagerInfo");
const prevPageBtn = $("prevPageBtn");
const nextPageBtn = $("nextPageBtn");
const ocrBtn = $("ocrBtn");

const gradeForm = $("gradeForm");
const referenceSolutionInput = $("referenceSolution");
const gradeCurrentQuestionOnlyInput = $("gradeCurrentQuestionOnly");
const gradeBtn = $("gradeBtn");
const resultSection = $("resultSection");
const totalScoreEl = $("totalScore");
const feedbackEl = $("feedback");
const feedbackTags = $("feedbackTags");
const knowledgePanel = $("knowledgePanel");
const stepTable = $("stepTable");
const favoriteBtn = $("favoriteBtn");
const downloadSimpleReportBtn = $("downloadSimpleReportBtn");
const downloadDetailReportBtn = $("downloadDetailReportBtn");
const previewDetailReportBtn = $("previewDetailReportBtn");
const regenerateDetailReportBtn = $("regenerateDetailReportBtn");
const detailReportModal = $("detailReportModal");
const detailReportMask = $("detailReportMask");
const detailReportPreview = $("detailReportPreview");
const closeDetailReportBtn = $("closeDetailReportBtn");
const downloadDetailFromModalBtn = $("downloadDetailFromModalBtn");
const newAssignmentBtn = $("newAssignmentBtn");

let cachedDetailReport = { recordId: null, markdown: null, status: null };
let detailReportPollTimer = null;
let detailReportCurrentRecordId = null;

const ocrHistoryEl = $("ocrHistory");
const gradingHistoryEl = $("gradingHistory");
const favoritesList = $("favoritesList");
const accountTabs = $("accountTabs");
const refreshAccountBtn = $("refreshAccountBtn");
const taskStepper = $("taskStepper");
const serviceBadgeAccount = null;
const globalStatus = $("globalStatus");

const loginForm = $("loginForm");
const logoutBtn = $("logoutBtn");
const adminLogoutBtn = $("adminLogoutBtn");
const authUsernameInput = $("authUsername");
const authPasswordInput = $("authPassword");
const authModalTitle = $("authModalTitle");
const authError = $("authError");
const roleSelector = $("roleSelector");
const loginBtn = $("loginBtn");
const authModeSwitch = $("authModeSwitch");
const authInfo = $("authInfo");
const profilePanel = $("profilePanel");
const authModal = $("authModal");
const authModalMask = $("authModalMask");
const closeAuthModalBtn = $("closeAuthModalBtn");

const loadAdminBtn = $("loadAdminBtn");
const adminOverview = $("adminOverview");

let authToken = localStorage.getItem("auth_token") || "";
let authUser = null;
let currentSegments = [];
let currentQuestionGroups = [];
// Signatures of questions the user deleted in the review stage. These must
// not resurface in the grading stage's question list.
let deletedQuestionSignatures = new Set();

function signatureOfQuestionText(text) {
  return String(text || "")
    .replace(/\s+/g, "")
    .slice(0, 200);
}
let currentQuestionIndex = 0;
let currentBatchIndex = 0;
let previewObjectUrl = "";
let lastGradeResult = null;
let lastFavoriteId = null;
let assignmentPages = [];
let pendingAfterLogin = null;
let currentAccountTab = "profile";
let authMode = "login";

const pageConfig = {
  "/": ["grading", "作业批改"],
  "/grading": ["grading", "作业批改"],
  "/account": ["account", "个人中心"],
  "/admin": ["admin", "管理后台"],
};

function apiFetch(url, options = {}) {
  // fetch 的薄包装：自动塞 Authorization header。token 来自 authToken（localStorage）。
  // 失败不上报，由调用方判断 resp.ok / resp.status。
  const headers = new Headers(options.headers || {});
  if (authToken) headers.set("Authorization", `Bearer ${authToken}`);
  return fetch(url, { ...options, headers });
}

async function apiFetchJson(url, options = {}) {
  // apiFetch + 401 自动跳登录。任何 401 都视为 token 失效（过期/被踢/手改），
  // 立即清 token + 弹登录框。其它状态码原样返回，调用方按业务处理。
  const resp = await apiFetch(url, options);
  if (resp.status === 401 && authToken) {
    setAuth("", null);
    openAuthModal();
    throw new Error("登录已失效，请重新登录。");
  }
  return resp;
}

function routePage() {
  // SPA 路由：根据 window.location.pathname 决定显示哪张 page-view。
  // 兼容旧路由：/history、/favorites、/wrong-answers 都重定向到 /account 的对应 tab。
  // 权限规则：
  //   - teacher/admin 强制只看后台（其它路径自动跳 /admin）；
  //   - account 页要求登录，未登录跳回 / 并弹登录框；
  //   - admin 页要求 teacher/admin，学生访问会被踢回 /account。
  // 路由切完后，按当前页触发对应的懒加载（KG 页初始化 vis、admin 页拉概览）。
  const pathname = window.location.pathname;
  if (pathname === "/history" || pathname === "/favorites") {
    // 旧路径兼容：重写到 /account?tab=xxx，然后重新进入 routePage。
    const tab = pathname === "/history" ? "grading" : "favorites";
    history.replaceState(null, "", "/account");
    currentAccountTab = tab;
    return routePage();
  }
  if (pathname === "/wrong-answers") {
    history.replaceState(null, "", "/account");
    currentAccountTab = "wrong-answers";
    return routePage();
  }
  let [key, title] = pageConfig[pathname] || pageConfig["/"];
  // teacher/admin 只能看后台：强制把其它路径改写成 /admin。
  const isStaff = authUser?.role === "teacher" || authUser?.role === "admin";
  if (isStaff && key !== "admin") {
    history.replaceState(null, "", "/admin");
    key = "admin";
    title = authUser?.role === "admin" ? "系统后台" : "教师后台";
  }
  if (key === "account" && !authUser) {
    history.replaceState(null, "", "/");
    key = "grading";
    title = "作业批改";
    openAuthModal();
  }
  if (key === "admin" && !isStaff) {
    // 学生访问 /admin：踢回 /account 并提示
    history.replaceState(null, "", "/account");
    [key, title] = pageConfig["/account"];
    setTimeout(() => alert("后台仅教师 / 管理员账号可见，请先使用相应账号登录。"), 0);
  }
  if (key === "account" && authUser) title = "个人中心";
  // 切 page-view 显隐 + 高亮当前导航项
  document.querySelectorAll(".page-view").forEach((view) => {
    view.classList.toggle("active", view.dataset.page === key);
  });
  document.querySelectorAll("[data-nav]").forEach((link) => {
    link.classList.toggle("active", link.dataset.nav === key);
  });
  if (pageTitle) pageTitle.textContent = title;
  taskStepper?.classList.toggle("hidden", key !== "grading");
  if (key === "grading") updateStepper();
  if (key === "account" && authUser) {
    switchAccountTab(currentAccountTab);
    refreshWorkspace();
  }
  if (key === "admin") {
    switchAdminTab(currentAdminTab);
    loadAdminOverview();
  }
}

function switchAccountTab(name) {
  // 个人中心 tab 切换：name 必须是 7 个合法 tab 之一，非法回退到 profile。
  // 切到 kg tab 时等 50ms 再调 initKgPage，让 pane 先 visible 再让 vis 测量尺寸。
  // 切到 wrong-answers / my-classes 时按需拉数据。
  const valid = ["profile", "ocr", "grading", "favorites", "wrong-answers", "my-classes", "kg"];
  if (!valid.includes(name)) name = "profile";
  currentAccountTab = name;
  document.querySelectorAll(".account-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  document.querySelectorAll(".account-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.dataset.tabContent === name);
    pane.classList.toggle("hidden", pane.dataset.tabContent !== name);
  });
  if (name === "kg" && typeof window.initKgPage === "function") {
    setTimeout(window.initKgPage, 50);
  }
  if (name === "wrong-answers" && authToken) {
    loadWrongAnswersPage();
  }
  if (name === "my-classes" && authToken) {
    loadMyClasses();
  }
}

let currentAdminTab = "overview";
function switchAdminTab(name) {
  // 后台 tab 切换。teacher 角色被禁止访问 overview / users tab
  // （看不到全平台数据），所以遇到这两个会自动回退到 classes。
  // 切完按 tab 名触发对应数据加载。
  const valid = ["overview", "users", "classes", "kg"];
  if (!valid.includes(name)) name = "overview";
  // teacher 没有 overview / users 权限：强制回退。
  if (authUser?.role === "teacher" && (name === "overview" || name === "users")) {
    name = "classes";
  }
  currentAdminTab = name;
  // 按角色隐藏不可见 tab 按钮（teacher 看不到 overview/users 按钮）
  const isTeacher = authUser?.role === "teacher";
  document.querySelectorAll("[data-admin-tab]").forEach((btn) => {
    const tabName = btn.dataset.adminTab;
    const hiddenForTeacher = isTeacher && (tabName === "overview" || tabName === "users");
    btn.classList.toggle("hidden", hiddenForTeacher);
    btn.classList.toggle("active", tabName === name);
  });
  document.querySelectorAll("[data-admin-pane]").forEach((pane) => {
    pane.classList.toggle("active", pane.dataset.adminPane === name);
    pane.classList.toggle("hidden", pane.dataset.adminPane !== name);
  });
  if (name === "overview") loadAdminOverview();
  if (name === "users") loadAdminUsers();
  if (name === "classes") loadAdminClasses();
  if (name === "kg") loadClassKgView();
}

// 4 阶段流水线：每个阶段编号映射到一组 DOM id。
// 切阶段时把这组的 hidden 去掉、其它 stage 加上 hidden。
const PAGE_STAGES = {
  1: ["uploadStage", "pagesStage"],   // 上传 + 已读取页列表
  2: ["reviewStage"],                  // 校对内容
  3: ["scoreStage"],                   // 评分设置
  4: ["resultSection"],                // 批改结果
};
let currentPage = 1;
// 已达到的最远阶段。用 canReachPage 阻止用户跳到尚未解锁的阶段（防"步 stepper 跳步"）。
let maxPageReached = 1;

function allStageIds() {
  // 所有 stage 的 DOM id 平铺，便于"先全部隐藏再显示当前阶段"。
  return Object.values(PAGE_STAGES).flat();
}

function showStage(page) {
  // 切换到指定阶段。会自动更新 maxPageReached、显隐 stage DOM、刷新顶部 stepper。
  // 特例：阶段 1 的 pagesStage 只在 assignmentPages 非空时才显示（避免空列表区域突兀）。
  if (!PAGE_STAGES[page]) return;
  currentPage = page;
  if (page > maxPageReached) maxPageReached = page;
  // 先把所有 stage 隐藏，再点亮当前阶段的 stage。
  for (const id of allStageIds()) {
    $(id)?.classList.add("hidden");
  }
  for (const id of PAGE_STAGES[page]) {
    const el = $(id);
    if (!el) continue;
    if (id === "pagesStage" && !assignmentPages.length) continue;
    el.classList.remove("hidden");
  }
  updateStepper();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function canReachPage(page) {
  // 阶段 1 永远可达；其它阶段只有走到过才能回退（防止用户跳过 OCR 直接评分）。
  if (page <= 1) return true;
  if (page <= maxPageReached) return true;
  return false;
}

function updateStepper() {
  // 根据 currentPage 更新顶部 stepper 的样式：
  // - 当前步：active；
  // - 已完成步：done + 显示 ✓；
  // - 可回访步（走过又退回的）：clickable，允许点击回退。
  if (!taskStepper) return;
  taskStepper.querySelectorAll(".stepper-item").forEach((item) => {
    const step = Number(item.dataset.step || "0");
    item.classList.toggle("active", step === currentPage);
    item.classList.toggle("done", step < currentPage);
    item.classList.toggle("clickable", canReachPage(step));
    const dot = item.querySelector(".stepper-dot");
    if (!dot) return;
    if (step < currentPage) dot.textContent = "✓";
    else dot.textContent = String(step);
  });
}

function bindStepperNavigation() {
  // 给每个 stepper-item 绑定点击事件，仅在 canReachPage 为 true 时生效。
  if (!taskStepper) return;
  taskStepper.querySelectorAll(".stepper-item").forEach((item) => {
    item.addEventListener("click", () => {
      const step = Number(item.dataset.step || "0");
      if (!canReachPage(step)) return;
      showStage(step);
    });
  });
}

function setStepperLoading(on) {
  // OCR 进行中时，把 stepper 第 2 步显示为 loading 状态（CSS 旋转动画）。
  if (!taskStepper) return;
  const step2 = taskStepper.querySelector('.stepper-item[data-step="2"]');
  if (!step2) return;
  step2.classList.toggle("loading", Boolean(on));
  if (on) {
    step2.classList.add("active");
    step2.classList.remove("done");
  }
}

function sanitizeUserMessage(text) {
  // 把后端原始错误信息里的技术细节替换成对用户友好的中文文案。
  // 例如把 "LLM_API_KEY 未配置" → "AI 识别服务未配置..."，
  // 把 "/responses、/chat/completions" → "评分接口"，避免暴露内部接口名。
  return String(text || "")
    .replace(/LLM_API_KEY\s*未[配置设置]/gi, "AI 识别服务未配置，请改用「粘贴文本批改」，或联系管理员设置密钥")
    .replace(/未配置LLM_API_KEY[，。]?/gi, "AI 识别服务未配置，请改用「粘贴文本批改」，或联系管理员设置密钥。")
    .replace(/LLM_API_KEY/gi, "AI 识别服务密钥")
    .replace(/\/responses|\/chat\/completions/gi, "评分接口")
    .replace(/ReadTimeout/gi, "请求超时");
}

let globalStatusTimer = null;

function setStatus(message, type = "") {
  // 全局 toast：右下角弹一条消息，自动消失。
  // type: "error" 显示 6 秒；其它显示 3.5 秒。
  // message 为空字符串时隐藏 toast（用于手动清屏）。
  const text = sanitizeUserMessage(message);
  if (globalStatus) {
    globalStatus.textContent = text;
    globalStatus.className = `global-toast ${type}`.trim();
    if (text) globalStatus.classList.remove("hidden");
    else globalStatus.classList.add("hidden");
    if (globalStatusTimer) {
      clearTimeout(globalStatusTimer);
      globalStatusTimer = null;
    }
    if (text) {
      const duration = type === "error" ? 6000 : 3500;
      globalStatusTimer = setTimeout(() => {
        globalStatus.classList.add("hidden");
        globalStatusTimer = null;
      }, duration);
    }
  }
}

function compactText(text, max = 120) {
  const normalized = String(text || "").replace(/\s+/g, " ").trim();
  return normalized.length > max ? `${normalized.slice(0, max)}...` : normalized || "暂无文本";
}

function formatTime(value) {
  if (!value) return "暂无";
  const date = new Date(String(value).replace(" ", "T"));
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", { hour12: false });
}

const ROLE_LABELS = { student: "学生", teacher: "教师", admin: "管理员" };

function roleLabel(role) {
  return ROLE_LABELS[String(role || "").toLowerCase()] || role || "";
}

function renderAuthInfo() {
  if (authInfo) {
    if (authUser) {
      const initial = String(authUser.username || "?").trim().charAt(0).toUpperCase() || "?";
      const role = String(authUser.role || "student").toLowerCase();
      const roleClass = role === "teacher" ? "teacher" : "student";
      authInfo.classList.remove("empty");
      authInfo.innerHTML = `
        <div class="profile-avatar">${initial}</div>
        <div class="profile-info">
          <span class="profile-name"></span>
          <span class="profile-role">账号角色<span class="role-chip ${roleClass}"></span></span>
        </div>
      `;
      authInfo.querySelector(".profile-name").textContent = authUser.username || "";
      authInfo.querySelector(".role-chip").textContent = roleLabel(role);
    } else {
      authInfo.classList.add("empty");
      authInfo.textContent = "未登录";
    }
  }
  loginEntryBtn?.classList.toggle("hidden", Boolean(authUser));
  userCenterBtn?.classList.toggle("hidden", !authUser);
  const userCenterNav = document.getElementById("userCenterNav");
  const isStaff = authUser?.role === "teacher" || authUser?.role === "admin";
  // Staff only get the backend — hide grading + personal center nav.
  document.querySelectorAll("[data-nav='grading']").forEach((el) => {
    el.classList.toggle("hidden", isStaff);
  });
  userCenterNav?.classList.toggle("hidden", !authUser || isStaff);
  userCenterBtn?.classList.toggle("hidden", !authUser || isStaff);
  profilePanel?.classList.toggle("hidden", !authUser || isStaff);
  adminNavLink?.classList.toggle("hidden", !isStaff);
  if (adminNavLink && isStaff) {
    adminNavLink.textContent = authUser?.role === "admin" ? "系统后台" : "教师后台";
  }
  const adminTitle = $("adminPageTitle");
  if (adminTitle && isStaff) {
    adminTitle.textContent = authUser?.role === "admin" ? "系统后台" : "教师后台";
  }
  // "My Classes" tab is student-only.
  document.querySelectorAll("[data-tab='my-classes']").forEach((btn) => {
    btn.classList.toggle("hidden", !authUser || isStaff);
  });
}

function setAuth(token, user) {
  authToken = token || "";
  authUser = user || null;
  if (authToken) localStorage.setItem("auth_token", authToken);
  else localStorage.removeItem("auth_token");
  renderAuthInfo();
  refreshWorkspace();
  routePage();
  if (authToken && pendingAfterLogin) {
    const next = pendingAfterLogin;
    pendingAfterLogin = null;
    setTimeout(next, 0);
  }
}

async function loadCurrentUser() {
  if (!authToken) {
    renderAuthInfo();
    return;
  }
  try {
    const resp = await apiFetch("/api/auth/me");
    const data = await resp.json();
    if (!resp.ok) {
      setAuth("", null);
      return;
    }
    authUser = data;
    if (authToken) localStorage.setItem("auth_token", authToken);
    else localStorage.removeItem("auth_token");
    renderAuthInfo();
    refreshWorkspace();
  } catch (_) {
    setAuth("", null);
  }
}

function requireLogin(nextAction = null) {
  if (authToken) return true;
  pendingAfterLogin = nextAction;
  setStatus("请先登录后继续批改。", "error");
  openAuthModal();
  return false;
}

function openAuthModal(mode = "login") {
  if (!authModal) return;
  setAuthMode(mode);
  authError?.classList.add("hidden");
  authError && (authError.textContent = "");
  authModal.classList.remove("hidden");
  // Only show the admin radio when no admin exists in the system (bootstrap).
  refreshAdminBootstrapHint();
}

async function refreshAdminBootstrapHint() {
  const option = document.getElementById("adminRoleOption");
  if (!option) return;
  try {
    const resp = await fetch("/api/auth/bootstrap-status");
    const data = await resp.json();
    option.classList.toggle("hidden", !data.admin_exists ? false : true);
    if (data.admin_exists) {
      const radio = option.querySelector("input[type='radio']");
      if (radio && radio.checked) {
        radio.checked = false;
        document.querySelector('input[name="authRole"][value="student"]')?.setAttribute("checked", "checked");
      }
    }
  } catch (_) {
    option.classList.add("hidden");
  }
}

function closeAuthModal() {
  authModal?.classList.add("hidden");
  if (authUsernameInput) authUsernameInput.value = "";
  if (authPasswordInput) authPasswordInput.value = "";
  if (authError) {
    authError.textContent = "";
    authError.classList.add("hidden");
  }
  setAuthMode("login");
}

function setAuthMode(mode) {
  authMode = mode === "register" ? "register" : "login";
  if (authModalTitle) authModalTitle.textContent = authMode === "register" ? "注册 MathGrade" : "登录 MathGrade";
  if (loginBtn) loginBtn.textContent = authMode === "register" ? "注册" : "登录";
  if (authModeSwitch) authModeSwitch.textContent = authMode === "register" ? "已有账号？去登录" : "没有账号？去注册";
  roleSelector?.classList.toggle("hidden", authMode !== "register");
  if (authPasswordInput) {
    authPasswordInput.autocomplete = authMode === "register" ? "new-password" : "current-password";
  }
}

function setAuthError(message) {
  if (!authError) {
    alert(message);
    return;
  }
  authError.textContent = message || "";
  authError.classList.toggle("hidden", !message);
}

function setServiceBadge(text, kind) {
  if (!serviceBadge) return;
  const cls = `status-pill ${kind || ""}`.trim();
  const hidden = !text;
  serviceBadge.textContent = text || "";
  serviceBadge.className = hidden ? "status-pill hidden" : cls;
}

async function loadDashboard() {
  if (!authToken) {
    if (metricOcrCount) metricOcrCount.textContent = "0";
    if (metricGradeCount) metricGradeCount.textContent = "0";
    if (metricAvgScore) metricAvgScore.textContent = "0";
    if (metricLastTime) metricLastTime.textContent = "暂无";
    setServiceBadge("", "");
    return;
  }
  try {
    const resp = await apiFetchJson("/api/dashboard");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "工作台加载失败");
    const stats = data.stats || {};
    if (metricOcrCount) metricOcrCount.textContent = String(stats.ocr_count || 0);
    if (metricGradeCount) metricGradeCount.textContent = String(stats.grading_count || 0);
    if (metricAvgScore) metricAvgScore.textContent = Number(stats.average_score || 0).toFixed(1);
    if (metricLastTime) metricLastTime.textContent = formatTime(stats.last_grading_time);
    setServiceBadge("", "");
  } catch (_) {
    setServiceBadge("服务连接异常", "error");
  }
}

function refreshWorkspace() {
  loadDashboard();
  loadHistory();
  loadFavorites();
}

function renderMath(container, latex, displayMode = false) {
  if (!container) return;
  container.innerHTML = "";
  const text = String(latex || "").trim();
  if (!text) return;
  if (window.katex) {
    try {
      window.katex.render(text, container, { throwOnError: true, displayMode });
      return;
    } catch (_) {
      // Render as plain text if the OCR line is not valid LaTeX.
    }
  }
  const fallback = document.createElement("span");
  fallback.className = "math-fallback";
  fallback.textContent = text;
  container.appendChild(fallback);
}

function renderRichLine(container, line, displayMode = false) {
  if (!container) return;
  container.innerHTML = "";
  const text = String(line ?? "");
  if (!text) return;
  // Fallback: line has no $...$ but looks like pure LaTeX → wrap whole line.
  const hasDollarWrap = /\$[^$]+\$/.test(text);
  const looksLikeLatex = /\\(frac|int|sum|sqrt|lim|sin|cos|tan|log|ln|cdot|times|alpha|beta|pi|theta|infty|Delta|Sigma|Pi|Omega|partial|nabla|dot|hat|vec|bar|mathrm|mathbb|left|right|displaystyle|dfrac|tfrac|text)/.test(text);
  const effective = (!hasDollarWrap && looksLikeLatex) ? `$${text}$` : text;
  const parts = effective.split(/(\$[^$]+\$)/g);
  let hasMath = false;
  for (const part of parts) {
    if (!part) continue;
    if (part.length >= 2 && part.startsWith("$") && part.endsWith("$")) {
      const math = part.slice(1, -1);
      const span = document.createElement("span");
      span.className = displayMode ? "math-segment math-display" : "math-segment";
      if (window.katex) {
        try {
          window.katex.render(math, span, { throwOnError: true, displayMode });
          container.appendChild(span);
          hasMath = true;
          continue;
        } catch (_) {
          // fall through to plain text
        }
      }
      span.textContent = math;
      container.appendChild(span);
    } else {
      container.appendChild(document.createTextNode(part));
    }
  }
  if (!hasMath && !container.childNodes.length) {
    container.textContent = text;
  }
}

function renderOcrPanel(text) {
  if (!ocrRendered) return;
  ocrRendered.innerHTML = "";
  const lines = String(text || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!lines.length) {
    ocrRendered.textContent = "等待读取结果";
    return;
  }
  for (const line of lines) {
    const row = document.createElement("div");
    row.className = "math-line";
    const isPureMath = line.startsWith("$") && line.endsWith("$") && line.indexOf("$", 1) === line.length - 1;
    renderRichLine(row, line, isPureMath);
    ocrRendered.appendChild(row);
  }
}

function renderStepPreviewFromText(text) {
  if (!stepPreview) return;
  stepPreview.innerHTML = "";
  String(text || "").split(/\r?\n/).map((v) => v.trim()).filter(Boolean).forEach((line) => {
    const li = document.createElement("li");
    renderRichLine(li, line, false);
    stepPreview.appendChild(li);
  });
}

function setQuestionGroups(groups) {
  currentQuestionGroups = Array.isArray(groups) ? groups.filter((g) => String(g?.text || "").trim()) : [];
  currentQuestionIndex = 0;
  // A fresh OCR result replaces whatever was reviewed before — clear stale deletions.
  deletedQuestionSignatures = new Set();
  applyQuestionPage();
}

function applyQuestionPage() {
  if (!questionPager || !ocrTextInput) return;
  if (!currentQuestionGroups.length) {
    questionPager.classList.add("hidden");
    return;
  }
  const idx = Math.max(0, Math.min(currentQuestionIndex, currentQuestionGroups.length - 1));
  currentQuestionIndex = idx;
  const group = currentQuestionGroups[idx];
  ocrTextInput.value = String(group?.text || "").trim();
  renderOcrPanel(ocrTextInput.value);
  renderStepPreviewFromText(ocrTextInput.value);
  questionPager.classList.remove("hidden");
  if (questionPageInfo) questionPageInfo.textContent = `题目 ${idx + 1} / ${currentQuestionGroups.length}`;
  if (prevQuestionBtn) prevQuestionBtn.disabled = idx <= 0;
  if (nextQuestionBtn) nextQuestionBtn.disabled = idx >= currentQuestionGroups.length - 1;
}

function deleteCurrentQuestion() {
  if (!currentQuestionGroups.length) return;
  const idx = currentQuestionIndex;
  const removed = currentQuestionGroups.splice(idx, 1)[0];
  const removedSig = signatureOfQuestionText(removed?.text || "");
  if (removedSig) deletedQuestionSignatures.add(removedSig);
  if (currentQuestionIndex >= currentQuestionGroups.length) {
    currentQuestionIndex = Math.max(0, currentQuestionGroups.length - 1);
  }
  applyQuestionPage();
  if (currentPage === 3) renderQuestionWeightList();
  const label = removed?.qno ?? `第 ${idx + 1} 题`;
  setStatus(`已删除 ${label}。剩余 ${currentQuestionGroups.length} 题（批改阶段会跳过已删除题）。`, "info");
}

function updateImagePreview(file) {
  if (!imagePreview || !imagePlaceholder) return;
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = "";
  if (!file) {
    imagePreview.style.display = "none";
    imagePreview.removeAttribute("src");
    imagePlaceholder.style.display = "grid";
    return;
  }
  previewObjectUrl = URL.createObjectURL(file);
  imagePreview.src = previewObjectUrl;
  imagePreview.style.display = "block";
  imagePlaceholder.style.display = "none";
}

function pageIdentity(file) {
  return `${file.name}|${file.size}|${file.lastModified}`;
}

function appendFilesToAssignment(files) {
  const existing = new Set(assignmentPages.map((p) => (p.file ? pageIdentity(p.file) : "")));
  let added = 0;
  for (const file of files) {
    if (!file || (file.type && !file.type.startsWith("image/"))) continue;
    const id = pageIdentity(file);
    if (existing.has(id)) continue;
    assignmentPages.push({
      file,
      title: file.name,
      objectUrl: URL.createObjectURL(file),
      ocrText: "",
      questionText: "",
      questionGroups: [],
      problems: [],
      status: "待读取",
      index: assignmentPages.length,
    });
    existing.add(id);
    added += 1;
  }
  return added;
}

function handleIncomingFiles(files) {
  if (!imageInput) return;
  const imageFiles = files.filter((f) => !f.type || f.type.startsWith("image/"));
  if (!imageFiles.length) {
    setStatus("只支持图片文件（JPG / PNG / WebP 等）。", "error");
    return;
  }
  const added = appendFilesToAssignment(imageFiles);
  if (imageInput) imageInput.value = "";
  if (added > 0) {
    currentBatchIndex = assignmentPages.length - added;
    renderBatchList();
    if (added === 1) {
      setStatus(`已添加 ${assignmentPages[assignmentPages.length - 1]?.title || "新图片"}。`, "info");
    } else {
      setStatus(`已添加 ${added} 张图片，共 ${assignmentPages.length} 页。`, "info");
    }
  } else if (imageFiles.length) {
    setStatus("这些图片已经在作业页中了。", "info");
  }
  updateStepper();
}

function renderBatchList() {
  if (!batchList) return;
  batchList.innerHTML = "";
  if (!assignmentPages.length) {
    if (currentPage === 1) pagesStage?.classList.add("hidden");
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "选择图片后，这里会显示整份作业的页面队列。";
    batchList.appendChild(empty);
    if (pageCountBadge) pageCountBadge.textContent = "0 页";
    updateStepper();
    return;
  }
  if (currentPage === 1) pagesStage?.classList.remove("hidden");
  if (pageCountBadge) pageCountBadge.textContent = `${assignmentPages.length} 页`;
  if (currentBatchIndex >= assignmentPages.length) currentBatchIndex = 0;
  assignmentPages.forEach((page, index) => {
    const row = document.createElement("div");
    const failed = page.status === "读取失败";
    row.className = `page-item ${index === currentBatchIndex ? "active" : ""} ${failed ? "failed" : ""}`.trim();
    row.setAttribute("role", "button");
    row.tabIndex = 0;
    const sizeText = page.file ? `${(page.file.size / 1024 / 1024).toFixed(2)}MB` : "文本";
    const subtitle = failed ? (page.error || "读取失败") : `${page.status} · ${sizeText}`;
    const thumb = page.objectUrl
      ? `<img class="page-thumb" src="${page.objectUrl}" alt="第 ${index + 1} 页缩略图" />`
      : `<span class="page-thumb"></span>`;
    row.innerHTML = `${thumb}<span><strong>第 ${index + 1} 页</strong><span>${subtitle}</span></span><button type="button" class="page-remove" aria-label="删除此页">×</button>`;
    row.addEventListener("click", (event) => {
      if (event.target.closest(".page-remove")) return;
      currentBatchIndex = index;
      updateImagePreview(page.file || null);
      applyPageToReview(index);
      renderBatchList();
    });
    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      row.click();
    });
    row.querySelector(".page-remove")?.addEventListener("click", (event) => {
      event.stopPropagation();
      removePageAt(index);
    });
    batchList.appendChild(row);
  });
  updateImagePreview(assignmentPages[currentBatchIndex]?.file || null);
  updateReOcrVisibility();
  updatePagePager();
  updateStepper();
}

function removePageAt(index) {
  const page = assignmentPages[index];
  if (!page) return;
  if (page.objectUrl) URL.revokeObjectURL(page.objectUrl);
  assignmentPages.splice(index, 1);
  if (!assignmentPages.length) {
    currentBatchIndex = 0;
    clearAssignmentPages();
    setStatus("已删除最后一页，作业页清空。", "info");
    return;
  }
  if (index < currentBatchIndex) currentBatchIndex -= 1;
  else if (currentBatchIndex >= assignmentPages.length) currentBatchIndex = assignmentPages.length - 1;
  const next = assignmentPages[currentBatchIndex];
  updateImagePreview(next?.file || null);
  applyPageToReview(currentBatchIndex);
  renderBatchList();
  setStatus(`已删除第 ${index + 1} 页，剩余 ${assignmentPages.length} 页。`, "info");
}

function clearAssignmentPages() {
  assignmentPages.forEach((page) => {
    if (page.objectUrl) URL.revokeObjectURL(page.objectUrl);
  });
  assignmentPages = [];
  currentBatchIndex = 0;
  if (imageInput) imageInput.value = "";
  renderBatchList();
  currentPage = 1;
  maxPageReached = 1;
  showStage(1);
}

async function readAssignmentPage(file, fallbackText = "", pageInfo = "") {
  // 单页 OCR 调用。两条路：
  //   - 有 file：走 /api/ocr-vision-only（视觉多模态模型识别图片）；
  //   - 无 file（粘贴文本批改）：走 /api/ocr（直接喂文本，跳过图片识别）。
  // 进度条在调用前 show、finally 里 hide，保证异常路径也会关闭。
  // 失败时抛错给调用方，detail 优先用后端返回的错误描述。
  const fd = new FormData();
  let endpoint = "/api/ocr-vision-only";
  if (file) {
    fd.append("image", file);
  } else {
    endpoint = "/api/ocr";
    fd.append("extracted_text", fallbackText);
  }
  showOcrProgress(pageInfo);
  try {
    const resp = await apiFetchJson(endpoint, { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "读取失败");
    return data;
  } finally {
    hideOcrProgress();
  }
}

// OCR 进度条阶段文案：按经过秒数切到下一档。
// 真实进度后端无法给（LLM 调用是黑盒），所以只能假装有进度——
// 每过几秒换一句文案，让用户感觉"它在干活"。
const OCR_STAGES = [
  { after: 0, label: "上传图片..." },
  { after: 4, label: "识别公式与文字..." },
  { after: 12, label: "整理步骤结构..." },
  { after: 25, label: "正在生成最终结果..." },
  { after: 50, label: "仍在思考中，请稍候..." },
  { after: 100, label: "复杂作业识别较慢，耐心等待..." },
];
let ocrProgressTimerHandle = null;
let ocrProgressStartTime = 0;

function showOcrProgress(pageInfo = "") {
  // 启动 OCR 进度条：显示页号 + 起始文案 + 0s 计时器。
  // 250ms tick 一次，刷新经过秒数和 stage 文案。
  const el = $("ocrProgressInline");
  if (!el) return;
  const pageEl = $("ocrProgressInlinePage");
  const stageEl = $("ocrProgressInlineStage");
  const timerEl = $("ocrProgressInlineTimer");
  if (pageEl) pageEl.textContent = pageInfo;
  if (stageEl) stageEl.textContent = OCR_STAGES[0].label;
  if (timerEl) timerEl.textContent = "0s";
  ocrProgressStartTime = Date.now();
  el.classList.remove("hidden");
  if (ocrProgressTimerHandle) clearInterval(ocrProgressTimerHandle);
  ocrProgressTimerHandle = setInterval(() => {
    const elapsed = Math.floor((Date.now() - ocrProgressStartTime) / 1000);
    if (timerEl) timerEl.textContent = `${elapsed}s`;
    // 找到 elapsed >= after 的最大那一档作为当前文案。
    let stage = OCR_STAGES[0];
    for (const s of OCR_STAGES) {
      if (elapsed >= s.after) stage = s;
    }
    if (stageEl && stageEl.textContent !== stage.label) {
      stageEl.textContent = stage.label;
    }
  }, 250);
}

function hideOcrProgress() {
  // OCR 完成（或异常）时调用：隐藏进度条 + 清定时器。
  const el = $("ocrProgressInline");
  if (el) el.classList.add("hidden");
  if (ocrProgressTimerHandle) {
    clearInterval(ocrProgressTimerHandle);
    ocrProgressTimerHandle = null;
  }
}

const GRADE_STAGES = [
  { after: 0, label: "分析作业内容..." },
  { after: 3, label: "按题切分步骤..." },
  { after: 10, label: "调用大模型评分..." },
  { after: 25, label: "整理批改结果..." },
  { after: 60, label: "仍在思考，请稍候..." },
];
let gradeProgressTimerHandle = null;
let gradeProgressStartTime = 0;
let gradingActive = false;
let gradingPaused = false;
let gradingAbort = false;

function updateGradeProgressTitle(done, total) {
  const titleEl = $("gradeProgressInlineTitle");
  if (!titleEl) return;
  if (gradingAbort) {
    titleEl.textContent = "正在停止";
  } else if (gradingPaused) {
    titleEl.textContent = total > 0 ? `已暂停（完成 ${done} / ${total} 题）` : "已暂停";
  } else if (total > 0) {
    titleEl.textContent = `正在批改 第 ${Math.min(done + 1, total)} / ${total} 题`;
  } else {
    titleEl.textContent = "正在批改";
  }
}

function showGradeProgress(done = 0, total = 0) {
  const el = $("gradeProgressInline");
  if (!el) return;
  const stageEl = $("gradeProgressInlineStage");
  const timerEl = $("gradeProgressInlineTimer");
  if (stageEl) stageEl.textContent = GRADE_STAGES[0].label;
  if (timerEl) timerEl.textContent = "0s";
  gradeProgressStartTime = Date.now();
  el.classList.remove("hidden");
  updateGradeProgressTitle(done, total);
  if (gradeProgressTimerHandle) clearInterval(gradeProgressTimerHandle);
  gradeProgressTimerHandle = setInterval(() => {
    const elapsed = Math.floor((Date.now() - gradeProgressStartTime) / 1000);
    if (timerEl) timerEl.textContent = `${elapsed}s`;
    let stage = GRADE_STAGES[0];
    for (const s of GRADE_STAGES) {
      if (elapsed >= s.after) stage = s;
    }
    if (stageEl && stageEl.textContent !== stage.label) {
      stageEl.textContent = stage.label;
    }
  }, 250);
}

function hideGradeProgress() {
  const el = $("gradeProgressInline");
  if (el) el.classList.add("hidden");
  if (gradeProgressTimerHandle) {
    clearInterval(gradeProgressTimerHandle);
    gradeProgressTimerHandle = null;
  }
}

function setPauseButtonState() {
  const btn = $("pauseGradeBtn");
  if (!btn) return;
  if (!gradingActive) {
    btn.classList.add("hidden");
    return;
  }
  btn.classList.remove("hidden");
  btn.textContent = gradingAbort ? "正在停止" : (gradingPaused ? "继续批改" : "暂停批改");
  btn.disabled = gradingAbort;
}

function togglePauseGrading() {
  if (!gradingActive || gradingAbort) return;
  gradingPaused = !gradingPaused;
  setPauseButtonState();
  updateGradeProgressTitle(currentGradeDone, currentGradeTotal);
  setStatus(gradingPaused ? "已暂停，完成当前题后等待继续。" : "继续批改中...", "info");
}

function abortGrading() {
  if (!gradingActive || gradingAbort) return;
  gradingAbort = true;
  gradingPaused = false;
  setPauseButtonState();
  updateGradeProgressTitle(currentGradeDone, currentGradeTotal);
  setStatus("正在停止批改，完成当前题目后结束。", "info");
}

let currentGradeDone = 0;
let currentGradeTotal = 0;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function applyOcrResult(data) {
  currentSegments = data.segments || [];
  const groups = data.question_groups || [];
  const problems = Array.isArray(data.problems) ? data.problems : [];
  if (assignmentPages[currentBatchIndex]) {
    assignmentPages[currentBatchIndex].ocrText = data.ocr_text || ocrTextInput?.value || "";
    assignmentPages[currentBatchIndex].questionText = data.question_text || "";
    assignmentPages[currentBatchIndex].questionGroups = groups;
    assignmentPages[currentBatchIndex].problems = problems;
    assignmentPages[currentBatchIndex].status = "已读取";
  }
  setQuestionGroups(groups);
  if (!currentQuestionGroups.length && ocrTextInput) {
    ocrTextInput.value = data.ocr_text || "";
    renderOcrPanel(ocrTextInput.value);
    renderStepPreviewFromText(ocrTextInput.value);
  }
  if (currentPage === 1) {
    if (maxPageReached < 2) maxPageReached = 2;
  }
  renderBatchList();
  updateReOcrVisibility();
  updateStepper();
}

function applyPageToReview(index) {
  const page = assignmentPages[index];
  if (!page) return;
  if (ocrTextInput) ocrTextInput.value = page.ocrText || "";
  renderOcrPanel(page.ocrText || "");
  renderStepPreviewFromText(page.ocrText || "");
  setQuestionGroups(Array.isArray(page.questionGroups) ? page.questionGroups : []);
}

function updatePagePager() {
  if (!pagePager) return;
  const total = assignmentPages.length;
  if (total <= 1) {
    pagePager.classList.add("hidden");
    return;
  }
  pagePager.classList.remove("hidden");
  const idx = Math.max(0, Math.min(currentBatchIndex, total - 1));
  if (pagePagerInfo) pagePagerInfo.textContent = `第 ${idx + 1} / ${total} 页`;
  if (prevPageBtn) prevPageBtn.disabled = idx <= 0;
  if (nextPageBtn) nextPageBtn.disabled = idx >= total - 1;
}

function switchToPage(index) {
  if (!assignmentPages.length) return;
  const idx = Math.max(0, Math.min(index, assignmentPages.length - 1));
  if (idx === currentBatchIndex) return;
  currentBatchIndex = idx;
  const page = assignmentPages[currentBatchIndex];
  updateImagePreview(page?.file || null);
  applyPageToReview(currentBatchIndex);
  renderBatchList();
}

function updateReOcrVisibility() {
  if (!reOcrBtn) return;
  const page = assignmentPages[currentBatchIndex];
  const hasFile = Boolean(page?.file);
  reOcrBtn.classList.toggle("hidden", !hasFile);
}

async function readCurrentPage() {
  if (!requireLogin(readCurrentPage)) return;
  const page = assignmentPages[currentBatchIndex];
  const file = page?.file || null;
  const fallbackText = manualTextInput?.value.trim() || "";
  if (!file && !fallbackText) {
    setStatus("请上传图片或填写手工文本。", "error");
    return;
  }
  if (ocrBtn) ocrBtn.disabled = true;
  setStatus("正在读取作业...");
  setStepperLoading(true);
  try {
    const data = await readAssignmentPage(file, fallbackText, file ? "正在识别作业图片" : "正在解析文本");
    if (!file && fallbackText && !assignmentPages.length) {
      assignmentPages = [{
        file: null,
        title: assignmentTitleInput?.value || "文本作业",
        objectUrl: "",
        ocrText: fallbackText,
        status: "文本输入",
        index: 0,
      }];
      if (pageCountBadge) pageCountBadge.textContent = "1 页";
    }
    applyOcrResult(data);
    setStatus("读取完成，可确认后批改。", "ok");
    showStage(2);
    refreshWorkspace();
  } catch (err) {
    if (page) {
      page.status = "读取失败";
      page.error = err.message || "读取失败";
      renderBatchList();
    }
    setStatus(err.message || "读取失败", "error");
  } finally {
    setStepperLoading(false);
    if (ocrBtn) ocrBtn.disabled = false;
  }
}

async function readWholeAssignment() {
  if (!requireLogin(readWholeAssignment)) return;
  if (assignmentPages.length <= 1) {
    await readCurrentPage();
    return;
  }
  if (ocrBtn) ocrBtn.disabled = true;
  setStepperLoading(true);
  let lastData = null;
  let lastSuccessIndex = -1;
  let successCount = 0;
  let failCount = 0;
  let firstError = "";
  const total = assignmentPages.length;
  try {
    for (let i = 0; i < total; i += 1) {
      if (!assignmentPages[i].file) continue;
      currentBatchIndex = i;
      renderBatchList();
      setStatus(`正在读取：${i + 1} / ${total}`);
      try {
        const data = await readAssignmentPage(assignmentPages[i].file, "", `第 ${i + 1} / ${total} 页`);
        assignmentPages[i].ocrText = data.ocr_text || "";
        assignmentPages[i].questionText = data.question_text || "";
        assignmentPages[i].status = "已读取";
        assignmentPages[i].error = "";
        successCount += 1;
        applyOcrResult(data);
        if (successCount < total) {
          setStatus(`已读取 ${successCount} / ${total} 页，继续中...`, "info");
        }
        lastData = data;
        lastSuccessIndex = i;
      } catch (err) {
        assignmentPages[i].status = "读取失败";
        assignmentPages[i].error = err.message || "读取失败";
        failCount += 1;
        if (!firstError) firstError = err.message || "读取失败";
      }
    }
    if (lastData && lastSuccessIndex >= 0) {
      currentBatchIndex = lastSuccessIndex;
      applyOcrResult(lastData);
    } else {
      renderBatchList();
    }
    if (failCount === 0) {
      setStatus(`整份作业读取完成：${total} 页。`, "ok");
    } else if (successCount === 0) {
      setStatus(`全部 ${failCount} 页读取失败：${firstError}`, "error");
    } else {
      setStatus(`读取完成：成功 ${successCount} 页，失败 ${failCount} 页。${firstError ? `（${firstError}）` : ""}`, "info");
    }
    if (successCount > 0) {
      showStage(2);
    }
    refreshWorkspace();
  } catch (err) {
    setStatus(err.message || "读取失败", "error");
  } finally {
    setStepperLoading(false);
    if (ocrBtn) ocrBtn.disabled = false;
  }
}

function buildKnowledgeTags(text, result) {
  const source = `${text}\n${result?.feedback || ""}`;
  const tags = [];
  if (/x|方程|=/.test(source)) tags.push("一元方程");
  if (/移项|-\s*\d|=\s*.*[-+]/.test(source)) tags.push("移项");
  if (/同类项|合并|2x|3x|x\s*=/.test(source)) tags.push("合并同类项");
  if (/除以|系数|x\s*=/.test(source)) tags.push("系数化为 1");
  if (/导数|求导|f'/.test(source)) tags.push("导数");
  if (/函数|代入|f\(/.test(source)) tags.push("函数求值");
  if (/面积|三角形|圆|角/.test(source)) tags.push("几何基础");
  const lowCount = (result?.step_scores || []).filter((s) => Number(s.score || 0) < 60).length;
  if (lowCount) tags.push(`重点复核 ${lowCount} 步`);
  if (!tags.length) tags.push("暂未匹配知识点");
  return [...new Set(tags)];
}

function classifyTag(tag) {
  if (tag.startsWith("重点复核")) return "warn";
  if (tag === "暂未匹配知识点") return "muted";
  return "info";
}

function renderKnowledgeDiagnosis(text, result) {
  if (!knowledgePanel) return;
  knowledgePanel.innerHTML = "";
  buildKnowledgeTags(text, result).forEach((tag) => {
    const el = document.createElement("span");
    el.className = `tag-${classifyTag(tag)}`;
    el.textContent = tag;
    if (tag.startsWith("重点复核")) {
      el.classList.add("tag-clickable");
      el.title = "点击定位到低分步骤";
      el.addEventListener("click", () => focusLowScoreRows());
    }
    knowledgePanel.appendChild(el);
  });
}

function focusLowScoreRows() {
  if (!stepTable) return;
  const rows = Array.from(stepTable.querySelectorAll("tr"));
  const lowRows = rows.filter((tr) => tr.querySelector(".score-cell.score-low"));
  if (!lowRows.length) {
    setStatus("没有低分步骤。", "info");
    return;
  }
  lowRows.forEach((tr) => {
    tr.classList.remove("row-highlight");
    void tr.offsetWidth;
    tr.classList.add("row-highlight");
  });
  lowRows[0].scrollIntoView({ behavior: "smooth", block: "center" });
  setStatus(`已定位 ${lowRows.length} 个低分步骤。`, "info");
}

function scoreClass(value) {
  const v = Number(value || 0);
  if (v >= 85) return "score-high";
  if (v >= 60) return "score-mid";
  return "score-low";
}

function renderRows(steps, scores) {
  if (!stepTable) return;
  stepTable.innerHTML = "";
  const map = new Map((scores || []).map((s) => [s.index, s]));
  for (const step of steps || []) {
    const score = map.get(step.index);
    const row = document.createElement("tr");
    const idxTd = document.createElement("td");
    idxTd.textContent = String(step.index);
    const contentTd = document.createElement("td");
    contentTd.className = "math-cell";
    renderMath(contentTd, step.normalized, false);
    const scoreTd = document.createElement("td");
    scoreTd.className = score ? `score-cell ${scoreClass(score.score)}` : "score-cell";
    scoreTd.textContent = score ? Number(score.score).toFixed(2) : "-";
    const reasonTd = document.createElement("td");
    reasonTd.textContent = score ? score.reason : "无";
    row.append(idxTd, contentTd, scoreTd, reasonTd);
    stepTable.appendChild(row);
  }
}

const QUESTION_START_RES = [
  /^\s*第\s*(\d+)\s*题/,
  /^\s*(\d+)\s*[.、)](?![\d.])\s*\S/,
  /^\s*(\d+)\s*[.、)]\s*$/,
];

function detectQuestionNumberFromLine(line) {
  const t = String(line || "").trim();
  if (!t) return null;
  for (const re of QUESTION_START_RES) {
    const m = t.match(re);
    if (m) {
      const n = parseInt(m[1], 10);
      if (n >= 1 && n <= 200) return n;
    }
  }
  return null;
}

function aggregateAllOcrText() {
  if (assignmentPages && assignmentPages.length) {
    const texts = assignmentPages
      .map((p) => String(p?.ocrText || "").trim())
      .filter(Boolean);
    if (texts.length) return texts.join("\n\n");
  }
  return String(ocrTextInput?.value || "").trim();
}

function aggregateAllQuestionText() {
  if (assignmentPages && assignmentPages.length) {
    const texts = assignmentPages
      .map((p) => String(p?.questionText || "").trim())
      .filter(Boolean);
    if (texts.length) return texts.join("\n\n");
  }
  return "";
}

function getAllQuestions() {
  // Prefer the per-problem structure produced by the OCR review pass — it has
  // a clean stem/steps split and exact qno, so the scorer never sees the
  // question stem as step 1. Fall back to text-based detection when problems
  // is empty (e.g. legacy OCR records or manual text entry).
  if (assignmentPages && assignmentPages.length) {
    const problems = assignmentPages
      .flatMap((p) => (Array.isArray(p?.problems) ? p.problems : []))
      .filter((pr) => pr && (String(pr.question_text || "").trim() || (Array.isArray(pr.step_lines) && pr.step_lines.length)));
    if (problems.length) {
      let filtered = problems;
      if (deletedQuestionSignatures.size) {
        filtered = filtered.filter(
          (pr) => !deletedQuestionSignatures.has(signatureOfQuestionText(String(pr.question_text || ""))),
        );
      }
      return filtered.map((pr, i) => ({
        qno: i + 1,
        text: [pr.question_text || "", ...((pr.step_lines || []))].join("\n").trim(),
        question_text: String(pr.question_text || "").trim(),
        step_lines: Array.isArray(pr.step_lines) ? pr.step_lines.map((s) => String(s)) : [],
        hasProblem: true,
      }));
    }
  }
  const fullText = aggregateAllOcrText();
  if (!fullText) return [];
  let questions = detectQuestionsFromText(fullText);
  if (!questions.length) questions = [{ qno: 1, text: fullText }];
  if (deletedQuestionSignatures.size) {
    questions = questions.filter(
      (q) => !deletedQuestionSignatures.has(signatureOfQuestionText(q.text || "")),
    );
  }
  // Renumber 1..N after deletions so qno stays contiguous.
  return questions.map((q, i) => ({ ...q, qno: i + 1 }));
}

function detectQuestionsFromText(text) {
  const src = String(text || "");
  if (!src.trim()) return [];
  const lines = src.split(/\r?\n/);
  const raw = [];
  let curOcrQno = null;
  let curBuf = [];
  const flush = () => {
    const joined = curBuf.map((s) => s.trim()).filter(Boolean).join("\n").trim();
    if (joined) {
      raw.push({ ocrQno: curOcrQno == null ? null : curOcrQno, text: joined });
    }
    curBuf = [];
  };
  for (const ln of lines) {
    const qno = detectQuestionNumberFromLine(ln);
    if (qno != null) {
      flush();
      curOcrQno = qno;
      curBuf = [ln];
    } else {
      curBuf.push(ln);
    }
  }
  flush();
  // Renumber sequentially 1..N to avoid duplicate-qno bugs (OCR sometimes labels
  // two different questions with the same "第 1 题" across pages).
  return raw.map((r, i) => ({
    qno: i + 1,
    ocrQno: r.ocrQno,
    text: r.text,
  }));
}

function getDefaultQuestionWeights(questions) {
  if (!questions || !questions.length) return {};
  const each = 100 / questions.length;
  const out = {};
  for (const q of questions) {
    out[String(q.qno)] = Math.round(each * 100) / 100;
  }
  return out;
}

let currentQuestionWeights = {};

function renderQuestionWeightList() {
  const wrap = $("questionWeightList");
  const rowsEl = $("questionWeightRows");
  const totalEl = $("questionWeightTotal");
  if (!wrap || !rowsEl) return;
  const questions = getAllQuestions();
  if (questions.length < 2) {
    wrap.classList.add("hidden");
    currentQuestionWeights = {};
    return;
  }
  wrap.classList.remove("hidden");
  if (!Object.keys(currentQuestionWeights).length || questions.some((q) => currentQuestionWeights[String(q.qno)] == null)) {
    currentQuestionWeights = getDefaultQuestionWeights(questions);
  }
  rowsEl.innerHTML = "";
  for (const q of questions) {
    const row = document.createElement("div");
    row.className = "question-weight-row";
    const label = document.createElement("div");
    label.className = "qw-label";
    label.textContent = `第 ${q.qno} 题`;
    const preview = document.createElement("div");
    preview.className = "qw-preview";
    renderRichLine(preview, q.text.replace(/\s+/g, " ").slice(0, 80), false);
    const inputWrap = document.createElement("div");
    inputWrap.className = "qw-input-wrap";
    const input = document.createElement("input");
    input.className = "qw-input";
    input.type = "number";
    input.min = "0";
    input.step = "1";
    input.value = String(currentQuestionWeights[String(q.qno)] ?? "");
    input.addEventListener("input", () => {
      const v = parseFloat(input.value);
      currentQuestionWeights[String(q.qno)] = Number.isFinite(v) ? Math.max(0, v) : 0;
      updateQuestionWeightTotal();
    });
    inputWrap.appendChild(input);
    const span = document.createElement("span");
    span.textContent = "分";
    inputWrap.appendChild(span);
    row.append(label, preview, inputWrap);
    rowsEl.appendChild(row);
  }
  updateQuestionWeightTotal();
}

function updateQuestionWeightTotal() {
  const totalEl = $("questionWeightTotal");
  if (!totalEl) return;
  const sum = Object.values(currentQuestionWeights).reduce((a, b) => a + (Number(b) || 0), 0);
  const rounded = Math.round(sum * 100) / 100;
  totalEl.textContent = `总分 ${rounded}`;
  totalEl.style.color = Math.abs(rounded - 100) < 0.001 ? "" : "#b25c10";
}

function collectQuestionWeights() {
  if (!currentQuestionWeights || !Object.keys(currentQuestionWeights).length) return null;
  const out = {};
  for (const [k, v] of Object.entries(currentQuestionWeights)) {
    out[String(k)] = Number(v) || 0;
  }
  return out;
}

let currentResultQuestions = [];
let currentResultQuestionIdx = 0;
let currentResultStepIdx = 0;

function renderGradeResult(data, textForGrade) {
  const tScoreEl = $("totalScore");
  const tDenomEl = $("totalScoreDenom");
  // Display the actual points earned over the user's configured total
  // (e.g. 92 / 96) rather than a confusing 0-100 percentage next to a non-100 max.
  const rawScore = Number(data.total_raw_score != null ? data.total_raw_score : data.total_score || 0);
  if (tScoreEl) tScoreEl.textContent = rawScore.toFixed(2);
  if (tDenomEl) {
    const max = Number(data.total_max_score || 100);
    const pct = max > 0 ? (rawScore / max) * 100 : 0;
    tDenomEl.textContent = `/ ${max % 1 === 0 ? max : max.toFixed(1)}（${pct.toFixed(1)}%）`;
  }
  if (feedbackEl) feedbackEl.textContent = data.feedback || "";
  const qResultsEl = $("questionResults");
  const flatWrap = $("flatResultWrap");
  const questions = Array.isArray(data.questions) ? data.questions : [];
  currentResultQuestions = questions;
  if (qResultsEl && questions.length >= 1) {
    qResultsEl.classList.remove("hidden");
    if (flatWrap) flatWrap.classList.add("hidden");
    const safeIdx = Math.min(currentResultQuestionIdx, questions.length - 1);
    renderResultQuestion(safeIdx < 0 ? 0 : safeIdx);
  } else {
    if (qResultsEl) qResultsEl.classList.add("hidden");
    if (flatWrap) {
      flatWrap.classList.remove("hidden");
      renderRows(data.steps, data.step_scores);
    }
  }
  renderFeedbackTags(data);
  renderKnowledgeDiagnosis(textForGrade, data);
}

function renderResultQuestion(idx) {
  if (!currentResultQuestions.length) return;
  if (idx < 0) idx = 0;
  if (idx >= currentResultQuestions.length) idx = currentResultQuestions.length - 1;
  currentResultQuestionIdx = idx;
  const q = currentResultQuestions[idx];
  const info = $("resultQuestionInfo");
  if (info) info.textContent = `第 ${q.qno} 题 / 共 ${currentResultQuestions.length} 题`;
  const prev = $("resultPrevQuestionBtn");
  const next = $("resultNextQuestionBtn");
  if (prev) prev.disabled = idx <= 0;
  if (next) next.disabled = idx >= currentResultQuestions.length - 1;
  const scoreEl = $("resultQuestionScore");
  if (scoreEl) {
    const pct = q.max_score > 0 ? (q.score / q.max_score) * 100 : 0;
    scoreEl.innerHTML = `<span class="${scoreClass(pct)}">${Number(q.score || 0).toFixed(2)}</span><span class="denom">/ ${Number(q.max_score || 0).toFixed(1)} 分</span>`;
  }
  const fbEl = $("resultQuestionFeedback");
  if (fbEl) fbEl.textContent = q.feedback || "";

  const body = $("resultStepBody");
  if (!body) return;
  body.innerHTML = "";
  const steps = q.steps || [];
  if (!steps.length) {
    const empty = document.createElement("div");
    empty.className = "result-step-row";
    empty.innerHTML = '<span class="step-label">提示</span><span class="step-value">该题未识别到步骤。</span>';
    body.appendChild(empty);
    return;
  }
  const scoreMap = new Map((q.step_scores || []).map((s) => [s.index, s]));
  steps.forEach((step, i) => {
    const score = scoreMap.get(step.index);
    const row = document.createElement("div");
    row.className = "result-step-row";
    const label = document.createElement("span");
    label.className = "step-label";
    label.textContent = `步骤 ${i + 1}`;
    const val = document.createElement("div");
    val.className = "step-value math-cell";
    renderRichLine(val, step.normalized, false);
    const meta = document.createElement("div");
    meta.className = "step-meta";
    const scoreTag = document.createElement("span");
    scoreTag.className = `step-score-tag ${score ? scoreClass((score.score / Math.max(1, q.max_score)) * 100) : ""}`;
    scoreTag.textContent = score ? `${Number(score.score).toFixed(2)} 分` : "-";
    meta.appendChild(scoreTag);
    if (score && score.reason) {
      const reasonTag = document.createElement("div");
      reasonTag.className = "step-reason";
      reasonTag.textContent = score.reason;
      meta.appendChild(reasonTag);
    }
    row.append(label, val, meta);
    body.appendChild(row);
  });
}

function renderFeedbackTags(result) {
  if (!feedbackTags) return;
  feedbackTags.innerHTML = "";
  const scores = result?.step_scores || [];
  const avg = scores.length ? scores.reduce((sum, s) => sum + Number(s.score || 0), 0) / scores.length : 0;
  [`步骤数：${result?.steps?.length || 0}`, `平均步骤分：${avg.toFixed(1)}`].forEach((tag) => {
    const el = document.createElement("span");
    el.textContent = tag;
    feedbackTags.appendChild(el);
  });
}

async function gradeHomework(event) {
  // 批改主流程入口。流程：
  //   1. 聚合所有 OCR 文本（aggregateAllOcrText）+ 切出题目列表（getAllQuestions）；
  //   2. 合并用户自定的题分（collectQuestionWeights）与默认平均分配；
  //   3. 逐题串行调 /api/grade，每完成一题立即更新结果区（增量渲染）；
  //   4. 支持暂停 / 继续中止；
  //   5. 全部完成后 POST /api/grading/session 落库为一条会话记录，
  //      拿到 record_id 后开始轮询精确报告（pollDetailReport）。
  //
  // 全程维护三个状态：gradingActive（是否在跑）、gradingPaused（是否暂停）、
  // gradingAbort（是否请求中止）。finally 里统一复位，防止 UI 卡在 loading。
  event?.preventDefault();
  if (!requireLogin(() => gradeHomework())) return;
  const fullText = aggregateAllOcrText();
  if (!fullText) {
    setStatus("请先读取或输入解题内容。", "error");
    return;
  }
  const textForGrade = fullText;
  const fullQuestionText = aggregateAllQuestionText();
  let questions = getAllQuestions();
  if (!questions.length) questions = [{ qno: 1, text: textForGrade }];

  // 题分合并：默认平均分配 → 用户调整覆盖。
  const userWeights = collectQuestionWeights() || {};
  const defaultWeights = getDefaultQuestionWeights(questions);
  const weights = { ...defaultWeights, ...userWeights };

  const reference = referenceSolutionInput?.value.trim() || "";
  const mergedReference = reference;

  // 状态复位 + UI 进入 loading。
  gradingActive = true;
  gradingPaused = false;
  gradingAbort = false;
  currentGradeDone = 0;
  currentGradeTotal = questions.length;
  currentResultQuestionIdx = 0;
  if (gradeBtn) gradeBtn.disabled = true;
  setPauseButtonState();
  setStatus("正在批改...");
  showGradeProgress(0, questions.length);

  // 清空结果区上一次的内容，避免新旧数据混在一起。
  const qResultsEl = $("questionResults");
  const flatWrap = $("flatResultWrap");
  if (qResultsEl) {
    const stepBody = $("resultStepBody");
    if (stepBody) stepBody.innerHTML = "";
    const infoEl = $("resultQuestionInfo");
    if (infoEl) infoEl.textContent = "第 - 题";
    const scoreEl = $("resultQuestionScore");
    if (scoreEl) scoreEl.innerHTML = "";
    const fbEl = $("resultQuestionFeedback");
    if (fbEl) fbEl.textContent = "";
    const stepInfo = $("resultStepInfo");
    if (stepInfo) stepInfo.textContent = "步骤 -";
    qResultsEl.classList.add("hidden");
  }
  if (flatWrap) flatWrap.classList.add("hidden");
  showStage(4);
  if (totalScoreEl) totalScoreEl.textContent = "-";
  const denomEl = $("totalScoreDenom");
  if (denomEl) {
    // 暂时把所有题的满分加起来显示在分母上，让用户知道"目标满分"是多少。
    const prelim = questions.reduce((a, q) => a + (Number(weights[String(q.qno)]) || 0), 0);
    denomEl.textContent = `/ ${prelim.toFixed(1)}`;
  }
  if (feedbackEl) feedbackEl.textContent = "";

  const collected = [];
  try {
    // 逐题评分。每跑完一题就增量重渲染结果，让用户能"边等边看"。
    for (let i = 0; i < questions.length; i += 1) {
      // 暂停轮询：每 250ms 检查一次是否继续。
      while (gradingActive && gradingPaused && !gradingAbort) {
        await sleep(250);
      }
      if (gradingAbort || !gradingActive) break;

      const q = questions[i];
      const maxScore = Number(weights[String(q.qno)]) || 100 / questions.length;
      currentGradeDone = i;
      updateGradeProgressTitle(i, questions.length);

      const fd = new FormData();
      // 如果 OCR review 阶段产出了"题干 + 步骤行"结构（hasProblem），就按结构化
      // 字段发；后端会跳过正则题号识别，避免把题干当成步骤评分。
      // extracted_text 作为 fallback / ocr_text 回显保留。
      if (q.hasProblem) {
        fd.append("extracted_text", q.text || "");
        if (q.question_text) fd.append("question_text", q.question_text);
        fd.append("step_lines", JSON.stringify(q.step_lines || []));
        fd.append("qno", String(q.qno));
      } else {
        fd.append("extracted_text", q.text);
        if (fullQuestionText) fd.append("question_text", fullQuestionText);
      }
      if (mergedReference) fd.append("reference_solution", mergedReference);
      fd.append("use_llm", "true");
      fd.append("question_max_scores", JSON.stringify({ [q.qno]: maxScore }));

      let result;
      try {
        const resp = await apiFetchJson("/api/grade", { method: "POST", body: fd });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "评分失败");
        const qResult = (data.questions && data.questions[0]) || {
          qno: q.qno,
          max_score: maxScore,
          score: 0,
          steps: data.steps || [],
          step_scores: data.step_scores || [],
          feedback: data.feedback || "",
        };
        // 强制使用前端位置序 qno：后端可能从 OCR 文本里解析 qno，
        // 偶尔会出现重复（比如两次"第 1 题"），用前端的循环序更可靠。
        qResult.qno = q.qno;
        // 附上整卷题干文本，供 /api/grading/session 落库后错题本有上下文。
        if (fullQuestionText) qResult.question_text = fullQuestionText;
        result = qResult;
      } catch (err) {
        // 单题失败：构造一个 0 分占位结果，不让整个批次崩。
        result = {
          qno: q.qno,
          max_score: maxScore,
          score: 0,
          steps: [],
          step_scores: [],
          feedback: `评分失败：${err.message || "未知错误"}`,
        };
        setStatus(`第 ${q.qno} 题评分失败：${err.message || "未知错误"}`, "error");
      }

      collected.push(result);
      // 每完成一题就重新聚合并渲染：用户能边等边看进度。
      const aggregated = buildAggregatedResult(collected, textForGrade);
      lastGradeResult = aggregated;
      lastFavoriteId = null;
      setFavoriteBtnState(false);
      renderGradeResult(aggregated, textForGrade);
      updateStepper();
      refreshWorkspace();
      currentGradeDone = i + 1;
      updateGradeProgressTitle(i + 1, questions.length);
    }

    if (gradingAbort) {
      setStatus(`已停止批改，完成 ${collected.length} / ${questions.length} 题。`, "info");
    } else {
      setStatus(`批改完成：共 ${collected.length} 题。`, "ok");
    }

    // 所有题跑完 + 未被中止：把整卷落库为一条 session 记录，
    // 后端会异步触发生成精确报告；前端拿到 record_id 后开始轮询。
    if (collected.length && !gradingAbort) {
      try {
        const sessionPayload = {
          ocr_text: textForGrade,
          total_score: lastGradeResult.total_score,
          total_max_score: lastGradeResult.total_max_score,
          feedback: lastGradeResult.feedback,
          engine: lastGradeResult.engine,
          questions: lastGradeResult.questions,
        };
        const sessionRes = await apiFetchJson("/api/grading/session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(sessionPayload),
        });
        if (sessionRes.ok) {
          const sessionData = await sessionRes.json();
          if (sessionData && sessionData.record_id) {
            lastGradeResult.record_id = sessionData.record_id;
            pollDetailReport(sessionData.record_id);
          }
        }
      } catch (e) {
        setDetailReportButtonState("unavailable");
      }
    } else {
      setDetailReportButtonState("unavailable");
    }
  } catch (err) {
    setStatus(err.message || "评分失败", "error");
  } finally {
    gradingActive = false;
    gradingPaused = false;
    gradingAbort = false;
    hideGradeProgress();
    setPauseButtonState();
    if (gradeBtn) gradeBtn.disabled = false;
  }
}

function buildAggregatedResult(questions, ocrText) {
  // 把逐题结果聚合成一个"整卷 result"对象，供 renderGradeResult 用。
  // 关键点：
  //   - total_score：把每题原始分加起来，按 (raw/max)*100 折算到 [0,100]；
  //   - flatSteps / flatScores：所有题的步骤连成一个扁平数组，
  //     index 用跨题累加偏移（避免不同题步骤号冲突）；
  //   - feedback：每题反馈按"第 N 题：xxx"拼接。
  const totalMax = questions.reduce((a, q) => a + (Number(q.max_score) || 0), 0);
  const totalRaw = questions.reduce((a, q) => a + (Number(q.score) || 0), 0);
  const totalScore = totalMax > 0 ? Math.min(100, Math.round((totalRaw / totalMax) * 10000) / 100) : 0;
  const flatSteps = [];
  const flatScores = [];
  let offset = 0;
  for (const q of questions) {
    // 跨题累加 offset，让每题的 step index 在整卷里唯一。
    for (const s of (q.steps || [])) {
      flatSteps.push({ ...s, index: s.index + offset });
    }
    for (const s of (q.step_scores || [])) {
      flatScores.push({ ...s, index: s.index + offset });
    }
    offset += (q.steps || []).length;
  }
  return {
    ocr_text: ocrText,
    questions,
    steps: flatSteps,
    step_scores: flatScores,
    total_score: totalScore,
    total_raw_score: Math.round(totalRaw * 100) / 100,
    total_max_score: Math.round(totalMax * 100) / 100,
    feedback: questions.map((q) => `第 ${q.qno} 题：${q.feedback || ""}`).join("\n\n"),
    engine: "sequential-llm",
    grading_meta: { scoring_mode: "sequential", questions_done: questions.length },
  };
}

function buildSimpleReportMarkdown(result) {
  const totalMax = Number(result.total_max_score || 100);
  const questions = Array.isArray(result.questions) ? result.questions : [];
  const lines = [
    "# 数学作业批改报告（简版）",
    "",
    `- 生成时间：${new Date().toLocaleString("zh-CN", { hour12: false })}`,
    `- 用户：${authUser ? `${authUser.username}（${authUser.role}）` : "未登录"}`,
    `- 总分：${Number(result.total_score || 0).toFixed(2)} / 100` + (questions.length ? `（卷面 ${questions.reduce((a, q) => a + (Number(q.score) || 0), 0).toFixed(2)} / ${totalMax.toFixed(1)}）` : ""),
    `- 评分引擎：${result.engine || "unknown"}`,
    "",
    "## 题目得分",
    "",
    "| 题号 | 得分 | 满分 |",
    "| --- | ---: | ---: |",
  ];
  if (questions.length) {
    for (const q of questions) {
      lines.push(`| 第 ${q.qno} 题 | ${Number(q.score || 0).toFixed(2)} | ${Number(q.max_score || 0).toFixed(1)} |`);
    }
  } else {
    lines.push(`| 第 1 题 | ${Number(result.total_score || 0).toFixed(2)} | ${totalMax.toFixed(1)} |`);
  }
  lines.push("", "## 整体反馈", "", result.feedback || "暂无反馈", "");
  return lines.join("\n");
}

const MD_ESCAPE_MAP = { "&": "&amp;", "<": "&lt;", ">": "&gt;" };
function escapeHtml(s) {
  return String(s || "").replace(/[&<>]/g, (c) => MD_ESCAPE_MAP[c]);
}

function renderInlineMarkdown(text) {
  // Returns an array of DOM nodes; handles $inline$ and **bold**.
  const nodes = [];
  const parts = String(text || "").split(/(\$[^$]+\$|\*\*[^*]+\*\*)/g);
  for (const part of parts) {
    if (!part) continue;
    if (part.startsWith("$") && part.endsWith("$") && part.length >= 2) {
      const math = part.slice(1, -1);
      const span = document.createElement("span");
      if (window.katex) {
        try {
          window.katex.render(math, span, { throwOnError: true, displayMode: false });
          nodes.push(span);
          continue;
        } catch (_) {}
      }
      span.textContent = math;
      nodes.push(span);
    } else if (part.startsWith("**") && part.endsWith("**") && part.length >= 4) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      nodes.push(strong);
    } else {
      nodes.push(document.createTextNode(part));
    }
  }
  return nodes;
}

function renderMarkdownInto(container, md) {
  if (!container) return;
  container.innerHTML = "";
  const lines = String(md || "").split(/\r?\n/);
  let i = 0;
  let listOpen = false;
  const closeList = () => {
    if (listOpen) { listOpen = false; }
  };
  const flushInline = (parent, text) => {
    for (const n of renderInlineMarkdown(text)) parent.appendChild(n);
  };
  while (i < lines.length) {
    const line = lines[i];

    // Code fence
    if (/^```/.test(line)) {
      closeList();
      const code = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i])) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1; // skip closing fence
      const pre = document.createElement("pre");
      pre.className = "md-pre";
      pre.textContent = code.join("\n");
      container.appendChild(pre);
      continue;
    }

    // Headings
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      closeList();
      const level = h[1].length;
      const el = document.createElement(`h${Math.min(level + 2, 6)}`);
      el.className = `md-h md-h-${level}`;
      flushInline(el, h[2]);
      container.appendChild(el);
      i += 1;
      continue;
    }

    // Table (line starts with | and next line is a separator)
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      closeList();
      const headerCells = line.split("|").slice(1, -1).map((c) => c.trim());
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        rows.push(lines[i].split("|").slice(1, -1).map((c) => c.trim()));
        i += 1;
      }
      const table = document.createElement("table");
      table.className = "md-table";
      const thead = document.createElement("thead");
      const trH = document.createElement("tr");
      for (const c of headerCells) {
        const th = document.createElement("th");
        flushInline(th, c);
        trH.appendChild(th);
      }
      thead.appendChild(trH);
      table.appendChild(thead);
      const tbody = document.createElement("tbody");
      for (const r of rows) {
        const tr = document.createElement("tr");
        for (const c of r) {
          const td = document.createElement("td");
          flushInline(td, c);
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      container.appendChild(table);
      continue;
    }

    // Unordered list
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) {
      if (!listOpen) {
        listOpen = true;
      }
      const ul = container.lastElementChild && container.lastElementChild.tagName === "UL"
        ? container.lastElementChild
        : null;
      let target = ul;
      if (!target) {
        target = document.createElement("ul");
        target.className = "md-ul";
        container.appendChild(target);
      }
      const liEl = document.createElement("li");
      flushInline(liEl, li[1]);
      target.appendChild(liEl);
      i += 1;
      continue;
    }

    // Empty line
    if (!line.trim()) {
      closeList();
      i += 1;
      continue;
    }

    // Paragraph
    closeList();
    const p = document.createElement("p");
    p.className = "md-p";
    flushInline(p, line);
    container.appendChild(p);
    i += 1;
  }
}

async function pollDetailReport(recordId) {
  // 轮询精确报告状态。批改完成后后端会异步触发 LLM 生成长篇 Markdown 报告，
  // 前端按 3 秒间隔轮询 /api/grading/{recordId}/detail，直到 status 变成
  // ready（成功）/ failed（失败）才停。失败时 5 秒退避再重试。
  //
  // detailReportCurrentRecordId 用作"当前轮询的 recordId"，
  // 每次新批改都会换新 id，旧 tick 通过比对 id 自动停止。
  if (!recordId) return;
  detailReportCurrentRecordId = recordId;
  cachedDetailReport = { recordId, markdown: null, status: "pending" };
  setDetailReportButtonState("pending");
  if (detailReportPollTimer) clearTimeout(detailReportPollTimer);

  const tick = async () => {
    // 如果用户已经触发了新的批改，停掉旧轮询。
    if (detailReportCurrentRecordId !== recordId) return;
    try {
      const res = await apiFetch(`/api/grading/${recordId}/detail`);
      if (!res.ok) {
        // 5xx 类错误：5 秒后重试，不立刻给用户报错。
        detailReportPollTimer = setTimeout(tick, 5000);
        return;
      }
      const data = await res.json();
      cachedDetailReport = { recordId, markdown: data.markdown, status: data.status };
      if (data.status === "ready") {
        setDetailReportButtonState("ready");
        return;  // 终态：停止轮询
      }
      if (data.status === "failed") {
        setDetailReportButtonState("failed");
        return;  // 终态：停止轮询，用户可点 "重新生成"
      }
      // pending：3 秒后再来一次。
      detailReportPollTimer = setTimeout(tick, 3000);
    } catch (_) {
      // 网络异常：5 秒退避重试。
      detailReportPollTimer = setTimeout(tick, 5000);
    }
  };
  tick();
}

function setDetailReportButtonState(state) {
  if (!downloadDetailReportBtn) return;
  if (state === "pending") {
    downloadDetailReportBtn.disabled = true;
    downloadDetailReportBtn.textContent = "精确报告生成中…";
    previewDetailReportBtn?.classList.add("hidden");
    regenerateDetailReportBtn?.classList.add("hidden");
  } else if (state === "ready") {
    downloadDetailReportBtn.disabled = false;
    downloadDetailReportBtn.textContent = "导出精确报告";
    previewDetailReportBtn?.classList.remove("hidden");
    regenerateDetailReportBtn?.classList.add("hidden");
  } else if (state === "failed") {
    downloadDetailReportBtn.disabled = true;
    downloadDetailReportBtn.textContent = "精确报告生成失败";
    previewDetailReportBtn?.classList.add("hidden");
    regenerateDetailReportBtn?.classList.remove("hidden");
  } else if (state === "unavailable") {
    downloadDetailReportBtn.disabled = true;
    downloadDetailReportBtn.textContent = "精确报告不可用";
    previewDetailReportBtn?.classList.add("hidden");
    regenerateDetailReportBtn?.classList.add("hidden");
  }
}

function openDetailReportModal() {
  if (!cachedDetailReport.markdown) {
    setStatus("精确报告尚未就绪。", "error");
    return;
  }
  if (detailReportPreview) renderMarkdownInto(detailReportPreview, cachedDetailReport.markdown);
  detailReportModal?.classList.remove("hidden");
}

function closeDetailReportModal() {
  detailReportModal?.classList.add("hidden");
}

async function regenerateDetailReport() {
  const recordId = detailReportCurrentRecordId || cachedDetailReport.recordId;
  if (!recordId) return;
  try {
    const res = await apiFetch(`/api/grading/${recordId}/detail/regenerate`, { method: "POST" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setStatus(`重新生成失败：${data.detail || res.status}`, "error");
      return;
    }
    setStatus("已重新开始生成精确报告…");
    pollDetailReport(recordId);
  } catch (e) {
    setStatus(`重新生成异常：${e.message || e}`, "error");
  }
}

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function resetAssignment() {
  assignmentPages.forEach((page) => {
    if (page.objectUrl) URL.revokeObjectURL(page.objectUrl);
  });
  assignmentPages = [];
  currentBatchIndex = 0;
  currentQuestionGroups = [];
  currentQuestionIndex = 0;
  deletedQuestionSignatures = new Set();
  currentSegments = [];
  lastGradeResult = null;
  lastFavoriteId = null;
  setFavoriteBtnState(false);
  if (detailReportPollTimer) {
    clearTimeout(detailReportPollTimer);
    detailReportPollTimer = null;
  }
  detailReportCurrentRecordId = null;
  cachedDetailReport = { recordId: null, markdown: null, status: null };
  setDetailReportButtonState("unavailable");
  if (imageInput) imageInput.value = "";
  if (assignmentTitleInput) assignmentTitleInput.value = "";
  if (ocrTextInput) ocrTextInput.value = "";
  if (manualTextInput) manualTextInput.value = "";
  if (referenceSolutionInput) referenceSolutionInput.value = "";
  updateImagePreview(null);
  renderBatchList();
  renderOcrPanel("");
  renderStepPreviewFromText("");
  currentQuestionWeights = {};
  const wl = $("questionWeightList");
  if (wl) wl.classList.add("hidden");
  const wrows = $("questionWeightRows");
  if (wrows) wrows.innerHTML = "";
  const stepBody = $("resultStepBody");
  if (stepBody) stepBody.innerHTML = "";
  const qr = $("questionResults");
  if (qr) qr.classList.add("hidden");
  questionPager?.classList.add("hidden");
  if (totalScoreEl) totalScoreEl.textContent = "-";
  if (feedbackEl) feedbackEl.textContent = "";
  if (feedbackTags) feedbackTags.innerHTML = "";
  if (knowledgePanel) knowledgePanel.innerHTML = "";
  if (stepTable) stepTable.innerHTML = "";
  currentPage = 1;
  maxPageReached = 1;
  showStage(1);
  uploadStage?.scrollIntoView({ behavior: "smooth", block: "start" });
  setStatus("已重置，可以上传新的作业。", "ok");
}

function renderHistoryList(container, items, type) {
  if (!container) return;
  container.innerHTML = "";
  if (!items?.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    const hint = type === "ocr"
      ? "上传作业图片或粘贴文本后，读取记录会出现在这里。"
      : "完成一次批改后，结果会出现在这里，可以恢复或导出简报。";
    empty.innerHTML = `
      <span class="empty-icon">∅</span>
      <span>暂无记录</span>
      <span class="empty-hint">${hint}</span>
    `;
    container.appendChild(empty);
    return;
  }
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "history-item";
    const title = document.createElement("div");
    title.className = "history-title";
    title.innerHTML = type === "ocr"
      ? `<span>读取 #${item.id}</span>`
      : `<span>批改 #${item.id}</span><span>${Number(item.total_score || 0).toFixed(1)}分</span>`;
    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = `${formatTime(item.created_at)} · ${item.steps_count || 0} 个步骤`;
    const preview = document.createElement("div");
    preview.className = "history-preview";
    preview.textContent = compactText(item.ocr_text);
    const actions = document.createElement("div");
    actions.className = "actions";
    const restore = document.createElement("button");
    restore.type = "button";
    restore.className = "secondary-btn";
    restore.textContent = "恢复到批改区";
    restore.addEventListener("click", async () => {
      history.pushState(null, "", "/grading");
      routePage();
      const ocrText = item.ocr_text || "";
      if (ocrTextInput) ocrTextInput.value = ocrText;
      renderOcrPanel(ocrText);
      renderStepPreviewFromText(ocrText);

      if (type === "grading") {
        try {
          const resp = await apiFetchJson(`/api/grading/${item.id}/payload`);
          if (!resp.ok) throw new Error(`payload ${resp.status}`);
          const data = await resp.json();
          const questions = Array.isArray(data.questions) ? data.questions : [];
          if (!questions.length) throw new Error("历史记录中没有题目数据");
          const aggregated = buildAggregatedResult(questions, ocrText);
          if (data.total_score != null) aggregated.total_score = data.total_score;
          if (data.total_max_score != null) aggregated.total_max_score = data.total_max_score;
          if (data.feedback) aggregated.feedback = data.feedback;
          if (data.engine) aggregated.engine = data.engine;
          lastGradeResult = aggregated;
          lastFavoriteId = null;
          setFavoriteBtnState(false);
          renderGradeResult(aggregated, ocrText);
          showStage(4);
          updateStepper();
          setStatus(`已恢复批改 #${item.id}，共 ${questions.length} 题。`, "ok");
          return;
        } catch (e) {
          console.warn("payload restore failed", e);
          setStatus(`恢复批改失败：${e.message}，已退回 OCR 文本。`, "error");
        }
      }

      // OCR type (or grading fallback): split text into questions and land on review stage.
      // Populate assignmentPages so downstream (weight UI, grade loop, getAllQuestions)
      // sees a proper page with groups. Without this, gradeHomework falls into the
      // text-detection fallback and may collapse to a single question.
      assignmentPages = [{
        file: null,
        title: item.title || "恢复的作业",
        objectUrl: "",
        ocrText,
        questionText: "",
        questionGroups: [],
        problems: [],
        status: "已恢复",
        index: 0,
      }];
      currentBatchIndex = 0;
      try {
        const splitResp = await apiFetchJson("/api/ocr/split", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: ocrText }),
        });
        if (splitResp.ok) {
          const splitData = await splitResp.json();
          const groups = (splitData.question_groups || []).filter((g) => String(g?.text || "").trim());
          if (groups.length) {
            assignmentPages[0].questionGroups = groups;
            // Synthesize per-problem structures from the split groups so the
            // new grading path (which prefers `problems` over text detection)
            // works on restored sessions too. The stem is the first line that
            // looks like a question marker; everything else becomes step_lines.
            assignmentPages[0].problems = groups.map((g, i) => {
              const lines = String(g.text || "").split(/\n+/).map((s) => s.trim()).filter(Boolean);
              const stemIdx = lines.findIndex((ln) => /^\s*(?:第\s*)?\d{1,3}(?:\s*题|\s*[.、)](?![\d.]))/.test(ln));
              const stem = stemIdx >= 0 ? lines[stemIdx] : lines[0];
              const steps = stemIdx >= 0 ? lines.slice(stemIdx + 1) : lines.slice(1);
              return {
                qno: i + 1,
                question_text: stem,
                step_lines: steps,
              };
            });
            setQuestionGroups(groups);
            updateImagePreview(null);
            applyPageToReview(0);
            updatePagePager();
            showStage(2);
            setStatus(`已恢复 ${groups.length} 道题，可在批改区校对。`, "ok");
            return;
          }
        }
      } catch (e) {
        console.warn("split restore failed", e);
      }
      currentQuestionGroups = [];
      questionPager?.classList.add("hidden");
      applyPageToReview(0);
      updatePagePager();
      showStage(2);
      setStatus("已恢复 OCR 文本，请校对内容。", "ok");
    });
    actions.appendChild(restore);
    if (type === "grading") {
      const report = document.createElement("button");
      report.type = "button";
      report.className = "secondary-btn";
      report.textContent = "导出简报";
      report.addEventListener("click", () => {
        downloadTextFile(`grading-${item.id}.md`, `# 批改简报\n\n- 总分：${Number(item.total_score || 0).toFixed(2)} / 100\n- 时间：${formatTime(item.created_at)}\n- 引擎：${item.engine}\n\n## 作业内容\n\n\`\`\`text\n${item.ocr_text || ""}\n\`\`\`\n`);
      });
      actions.appendChild(report);
    }
    card.append(title, meta, preview, actions);
    container.appendChild(card);
  }
}

async function loadHistory() {
  if (!authToken) {
    renderHistoryList(ocrHistoryEl, [], "ocr");
    renderHistoryList(gradingHistoryEl, [], "grading");
    return;
  }
  try {
    const [ocrResp, gradingResp] = await Promise.all([
      apiFetchJson("/api/history/ocr?limit=12"),
      apiFetchJson("/api/history/grading?limit=12"),
    ]);
    const [ocrData, gradingData] = await Promise.all([ocrResp.json(), gradingResp.json()]);
    if (!ocrResp.ok) throw new Error(ocrData.detail || "读取历史加载失败");
    if (!gradingResp.ok) throw new Error(gradingData.detail || "评分历史加载失败");
    renderHistoryList(ocrHistoryEl, ocrData.items || [], "ocr");
    renderHistoryList(gradingHistoryEl, gradingData.items || [], "grading");
  } catch (err) {
    renderHistoryList(ocrHistoryEl, [], "ocr");
    renderHistoryList(gradingHistoryEl, [], "grading");
  }
}

function renderFavorites(items) {
  if (!favoritesList) return;
  favoritesList.innerHTML = "";
  if (!items?.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = `
      <span class="empty-icon">★</span>
      <span>还没有收藏</span>
      <span class="empty-hint">批改完成后点击"收藏作业"，之后可以在这里复习。</span>
    `;
    favoritesList.appendChild(empty);
    return;
  }
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "favorite-card";
    const tags = Array.isArray(item.knowledge_tags) ? item.knowledge_tags : [];
    card.innerHTML = `
      <div class="history-title">
        <span>${item.title || "未命名作业"}</span>
        <span>${Number(item.total_score || 0).toFixed(1)} 分</span>
      </div>
      <div class="history-meta">${formatTime(item.created_at)}</div>
      <div class="tag-row">${tags.map((tag) => `<span>${tag}</span>`).join("")}</div>
      <div class="history-preview">${compactText(item.ocr_text, 180)}</div>
      <div class="actions">
        <button type="button" class="secondary-btn" data-restore="${item.id}">恢复到批改区</button>
        <button type="button" class="secondary-btn" data-delete="${item.id}">取消收藏</button>
      </div>
    `;
    card.querySelector("[data-restore]")?.addEventListener("click", () => {
      history.pushState(null, "", "/");
      routePage();
      if (assignmentTitleInput) assignmentTitleInput.value = item.title || "";
      if (ocrTextInput) ocrTextInput.value = item.ocr_text || "";
      assignmentPages = [{
        file: null,
        title: item.title || "收藏作业",
        objectUrl: "",
        ocrText: item.ocr_text || "",
        status: "收藏",
        index: 0,
      }];
      currentBatchIndex = 0;
      renderBatchList();
      renderOcrPanel(item.ocr_text || "");
      renderStepPreviewFromText(item.ocr_text || "");
      currentPage = 1;
      maxPageReached = 3;
      showStage(2);
      setStatus("已从收藏夹恢复。", "ok");
    });
    card.querySelector("[data-delete]")?.addEventListener("click", async () => {
      if (!authToken) return;
      const resp = await apiFetchJson(`/api/favorites/${item.id}`, { method: "DELETE" });
      if (resp.ok) loadFavorites();
    });
    favoritesList.appendChild(card);
  }
}

async function loadFavorites() {
  if (!authToken) {
    renderFavorites([]);
    return;
  }
  try {
    const resp = await apiFetchJson("/api/favorites?limit=80");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "收藏加载失败");
    renderFavorites(data.items || []);
  } catch (_) {
    renderFavorites([]);
  }
}

async function saveFavorite() {
  if (!requireLogin(saveFavorite)) return;
  if (!lastGradeResult) {
    setStatus("请先完成批改，再收藏作业。", "error");
    return;
  }
  if (lastFavoriteId) {
    setStatus("当前作业已收藏。", "info");
    return;
  }
  const title = (assignmentTitleInput?.value || "").trim() || `作业 ${new Date().toLocaleDateString("zh-CN")}`;
  const tags = buildKnowledgeTags(lastGradeResult.ocr_text || ocrTextInput?.value || "", lastGradeResult);
  const fd = new FormData();
  fd.append("title", title);
  fd.append("ocr_text", lastGradeResult.ocr_text || ocrTextInput?.value || "");
  fd.append("total_score", String(lastGradeResult.total_score || 0));
  fd.append("feedback", lastGradeResult.feedback || "");
  fd.append("knowledge_tags_json", JSON.stringify(tags));
  fd.append("report_json", JSON.stringify(lastGradeResult));
  try {
    const resp = await apiFetchJson("/api/favorites", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "收藏失败");
    lastFavoriteId = data.id || null;
    setFavoriteBtnState(true);
    setStatus("已收藏，可在收藏夹中复习。", "ok");
    loadFavorites();
  } catch (err) {
    setStatus(err.message || "收藏失败", "error");
  }
}

function setFavoriteBtnState(favorited) {
  if (!favoriteBtn) return;
  if (favorited) {
    favoriteBtn.textContent = "已收藏 ✓";
    favoriteBtn.classList.add("favorited");
    favoriteBtn.disabled = true;
  } else {
    favoriteBtn.textContent = "收藏作业";
    favoriteBtn.classList.remove("favorited");
    favoriteBtn.disabled = false;
  }
}

async function loadAdminOverview() {
  if (!adminOverview) return;
  if (!authToken) {
    adminOverview.innerHTML = `<div class="empty-state">请先使用教师账号登录。</div>`;
    return;
  }
  try {
    const resp = await apiFetchJson("/api/admin/overview");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "后台数据加载失败");
    const userText = (data.users_by_role || []).map((x) => `${x.role}: ${x.count}`).join("，") || "暂无用户";
    const recent = (data.recent_gradings || []).map((x) => `<div>${x.username} · ${Number(x.total_score || 0).toFixed(1)}分 · ${formatTime(x.created_at)}</div>`).join("") || "<div>暂无批改</div>";
    adminOverview.innerHTML = `
      <article class="metric-card"><span>用户分布</span><strong>${userText}</strong></article>
      <article class="metric-card"><span>读取总数</span><strong>${data.ocr_count || 0}</strong></article>
      <article class="metric-card"><span>批改总数</span><strong>${data.grading_count || 0}</strong></article>
      <article class="metric-card"><span>平均分</span><strong>${Number(data.average_score || 0).toFixed(1)}</strong></article>
      <article class="admin-wide"><h3>最近批改</h3><div class="admin-list">${recent}</div></article>
    `;
  } catch (err) {
    adminOverview.innerHTML = `<div class="empty-state">${err.message || "后台数据加载失败"}</div>`;
  }
}

// ============================================================
// Admin: user management
// ============================================================

async function loadAdminUsers() {
  const wrap = $("adminUsersList");
  if (!wrap) return;
  wrap.innerHTML = `<div class="empty-state">加载中...</div>`;
  try {
    const resp = await apiFetchJson("/api/admin/users");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "用户列表加载失败");
    const users = data.users || [];
    if (!users.length) {
      wrap.innerHTML = `<div class="empty-state">暂无用户。</div>`;
      return;
    }
    wrap.innerHTML = `<table class="admin-user-table"><thead><tr>
      <th>ID</th><th>用户名</th><th>角色</th><th>批改数</th><th>错题数</th><th>注册时间</th><th>操作</th>
    </tr></thead><tbody>${users.map((u) => `
      <tr data-uid="${u.id}">
        <td>${u.id}</td>
        <td>${escapeHtml(u.username || "")}</td>
        <td>
          <select class="admin-role-select" data-uid="${u.id}">
            <option value="student" ${u.role === "student" ? "selected" : ""}>学生</option>
            <option value="teacher" ${u.role === "teacher" ? "selected" : ""}>教师</option>
            <option value="admin" ${u.role === "admin" ? "selected" : ""}>管理员</option>
          </select>
        </td>
        <td>${u.grading_count || 0}</td>
        <td>${u.wrong_count || 0}</td>
        <td>${formatTime(u.created_at)}</td>
        <td><button type="button" class="text-action admin-del-user" data-uid="${u.id}">删除</button></td>
      </tr>`).join("")}</tbody></table>`;
    wrap.querySelectorAll(".admin-role-select").forEach((sel) => {
      sel.addEventListener("change", async (ev) => {
        const uid = Number(ev.target.dataset.uid);
        const role = ev.target.value;
        await adminUpdateUserRole(uid, role);
      });
    });
    wrap.querySelectorAll(".admin-del-user").forEach((btn) => {
      btn.addEventListener("click", async (ev) => {
        const uid = Number(ev.currentTarget.dataset.uid);
        if (!confirm(`确认删除用户 #${uid}？将级联清理其 OCR / 批改 / 收藏 / 错题。`)) return;
        await adminDeleteUser(uid);
      });
    });
  } catch (err) {
    wrap.innerHTML = `<div class="empty-state">${err.message || "用户列表加载失败"}</div>`;
  }
}

async function adminUpdateUserRole(uid, role) {
  // 改用户角色（student ↔ teacher）。失败时回滚 UI（重拉列表）。
  try {
    const resp = await apiFetchJson(`/api/admin/users/${uid}/role`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || "修改角色失败");
    }
    setStatus(`已将用户 #${uid} 角色更新为 ${role === "teacher" ? "教师" : "学生"}。`, "info");
  } catch (err) {
    setStatus(err.message || "修改角色失败", "error");
    loadAdminUsers();  // 重拉：让 UI 与后端实际状态一致
  }
}

async function adminDeleteUser(uid) {
  // 删除用户。后端会级连清理该用户的 OCR / 批改 / 收藏 / 错题记录
  // （在 db.py 里有 ON DELETE CASCADE 或显式删除）。
  try {
    const resp = await apiFetchJson(`/api/admin/users/${uid}`, { method: "DELETE" });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || "删除失败");
    }
    setStatus(`已删除用户 #${uid}。`, "info");
    loadAdminUsers();
  } catch (err) {
    setStatus(err.message || "删除失败", "error");
  }
}

// ============================================================
// 后台 - 班级（admin: classes）
// ------------------------------------------------------------
// 班级生命周期：
//   1. admin/teacher 在后台创建班级（admin 可指派给任意 teacher，
//      teacher 只能创建自己的班级）；
//   2. 系统生成 6 位邀请码；
//   3. 学生在"个人中心 → 我的班级"输入邀请码加入；
//   4. teacher/admin 可看成员名单、班级报告（平均分 / 错题分布）；
//   5. teacher 可对班级做 KG 覆盖（隐藏某些超纲节点 / 新增校本节点）。
// ============================================================

// 班级学段标签，与后端 classes.py 一致。
const CLASS_STAGE_LABELS = { primary: "小学", middle: "初中", high: "高中" };
// 教师 list 缓存：admin 视角创建班级时下拉选择任课教师。
let adminTeachersCache = [];

async function loadAdminClasses() {
  const wrap = $("adminClassesList");
  if (!wrap) return;
  wrap.innerHTML = `<div class="empty-state">加载中...</div>`;
  try {
    const [classesResp, teachersResp] = await Promise.all([
      apiFetchJson("/api/classes"),
      authUser?.role === "admin" ? apiFetchJson("/api/admin/users").catch(() => null) : Promise.resolve(null),
    ]);
    const data = await classesResp.json();
    if (!classesResp.ok) throw new Error(data.detail || "班级加载失败");
    const items = data.items || [];
    if (authUser?.role === "admin" && teachersResp) {
      const tdata = await teachersResp.json().catch(() => ({ users: [] }));
      adminTeachersCache = (tdata.users || []).filter((u) => u.role === "teacher");
    } else {
      adminTeachersCache = [];
    }
    const toolbar = `
      <div class="admin-classes-toolbar">
        <button type="button" id="adminClassNewBtn" class="secondary-btn">新建班级</button>
        <button type="button" id="adminClassRefreshBtn" class="secondary-btn">刷新</button>
      </div>`;
    if (!items.length) {
      wrap.innerHTML = `${toolbar}<div class="empty-state">还没有班级，点「新建班级」创建第一个。</div>`;
    } else {
      wrap.innerHTML = `${toolbar}<div class="admin-classes-grid">
        ${items.map((c) => `
          <article class="class-card" data-id="${c.id}">
            <header>
              <strong>${escapeHtml(c.name)}</strong>
              <span class="class-stage-tag">${CLASS_STAGE_LABELS[c.stage] || c.stage} · G${c.grade || 0}</span>
            </header>
            <div class="class-card-meta">
              <span>成员 ${c.member_count || 0}</span>
              <span>邀请码 <code class="class-invite">${escapeHtml(c.invite_code)}</code></span>
            </div>
            ${c.description ? `<p class="class-desc">${escapeHtml(c.description)}</p>` : ""}
            <div class="class-card-actions">
              <button type="button" class="text-action class-report-btn">查看报表</button>
              <button type="button" class="text-action class-edit-btn">编辑</button>
              <button type="button" class="text-action class-invite-reset-btn">重置邀请码</button>
              <button type="button" class="text-action class-del-btn">删除</button>
            </div>
          </article>
        `).join("")}
      </div>`;
    }
    $("adminClassNewBtn")?.addEventListener("click", () => openClassEditor(null));
    $("adminClassRefreshBtn")?.addEventListener("click", loadAdminClasses);
    wrap.querySelectorAll(".class-card").forEach((card) => {
      const id = Number(card.dataset.id);
      card.querySelector(".class-report-btn")?.addEventListener("click", () => openClassReport(id));
      card.querySelector(".class-edit-btn")?.addEventListener("click", () => {
        const resp = items.find((x) => x.id === id);
        openClassEditor(resp);
      });
      card.querySelector(".class-invite-reset-btn")?.addEventListener("click", async () => {
        if (!confirm("确认重置邀请码？旧码将失效。")) return;
        try {
          const r = await apiFetchJson(`/api/admin/classes/${id}/regenerate-invite`, { method: "POST" });
          if (!r.ok) throw new Error("重置失败");
          setStatus("邀请码已重置。", "info");
          loadAdminClasses();
        } catch (err) {
          setStatus(err.message || "重置失败", "error");
        }
      });
      card.querySelector(".class-del-btn")?.addEventListener("click", async () => {
        if (!confirm("确认删除该班级？学生与班级的关联会一并清理（学生账号和记录保留）。")) return;
        try {
          const r = await apiFetchJson(`/api/admin/classes/${id}`, { method: "DELETE" });
          if (!r.ok) throw new Error("删除失败");
          setStatus("已删除班级。", "info");
          loadAdminClasses();
        } catch (err) {
          setStatus(err.message || "删除失败", "error");
        }
      });
    });
  } catch (err) {
    wrap.innerHTML = `<div class="empty-state">${err.message || "班级加载失败"}</div>`;
  }
}

function openClassEditor(cls) {
  const modal = $("classEditorModal");
  if (!modal) return;
  modal.classList.remove("hidden");
  $("classEditorTitle").textContent = cls ? "编辑班级" : "新建班级";
  $("classEditorId").value = cls ? cls.id : "";
  $("classEditorName").value = cls ? cls.name : "";
  $("classEditorStage").value = cls ? cls.stage : "middle";
  $("classEditorGrade").value = cls ? cls.grade || 0 : 0;
  $("classEditorDesc").value = cls ? cls.description || "" : "";
  // Admin can reassign class to any teacher; teachers create as themselves.
  const creatorWrap = $("classEditorCreatorWrap");
  const creatorSel = $("classEditorCreator");
  if (creatorWrap && creatorSel) {
    const show = authUser?.role === "admin" && adminTeachersCache.length > 0;
    creatorWrap.classList.toggle("hidden", !show);
    if (show) {
      creatorSel.innerHTML = adminTeachersCache.map((t) =>
        `<option value="${t.id}" ${cls && Number(cls.creator_id) === Number(t.id) ? "selected" : ""}>${escapeHtml(t.username)} (#${t.id})</option>`
      ).join("");
    }
  }
}

function closeClassEditor() {
  $("classEditorModal")?.classList.add("hidden");
}

async function submitClassEditor() {
  const id = $("classEditorId").value;
  const body = {
    name: $("classEditorName").value.trim(),
    stage: $("classEditorStage").value,
    grade: Number($("classEditorGrade").value || 0),
    description: $("classEditorDesc").value.trim(),
  };
  if (authUser?.role === "admin" && !id) {
    const creator = $("classEditorCreator")?.value;
    if (creator) body.creator_id = Number(creator);
  }
  if (!body.name) {
    setStatus("班级名称不能为空。", "error");
    return;
  }
  try {
    const url = id ? `/api/admin/classes/${id}` : "/api/admin/classes";
    const method = id ? "PATCH" : "POST";
    const resp = await apiFetchJson(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || "保存失败");
    }
    setStatus(id ? "班级已更新。" : "班级已创建。", "info");
    closeClassEditor();
    loadAdminClasses();
  } catch (err) {
    setStatus(err.message || "保存失败", "error");
  }
}

async function openClassReport(classId) {
  const modal = $("classReportModal");
  const body = $("classReportBody");
  if (!modal || !body) return;
  modal.classList.remove("hidden");
  body.innerHTML = `<div class="empty-state">加载中...</div>`;
  try {
    const resp = await apiFetchJson(`/api/classes/${classId}/report`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "报表加载失败");
    const errDist = data.error_type_distribution || {};
    const errRow = Object.entries(errDist)
      .filter(([_, v]) => v > 0)
      .map(([k, v]) => `<span class="wrong-tag wrong-tag-${k}">${WRONG_ERROR_LABELS[k] || k} ${v}</span>`)
      .join("") || "<span class='muted'>暂无</span>";
    const recent = (data.recent_gradings || [])
      .map((g) => `<tr><td>${escapeHtml(g.username || "")}</td><td>${Number(g.total_score || 0).toFixed(1)}</td><td>${formatTime(g.created_at)}</td></tr>`)
      .join("") || "<tr><td colspan='3' class='muted'>暂无</td></tr>";
    body.innerHTML = `
      <div class="admin-classes-report">
        <section class="metric-grid">
          <article class="metric-card"><span>成员</span><strong>${data.member_count || 0}</strong></article>
          <article class="metric-card"><span>批改次数</span><strong>${data.grading_count || 0}</strong></article>
          <article class="metric-card"><span>平均分</span><strong>${data.avg_score === null || data.avg_score === undefined ? "—" : Number(data.avg_score).toFixed(1)}</strong></article>
          <article class="metric-card"><span>错题数</span><strong>${data.wrong_answer_count || 0}</strong></article>
        </section>
        <h3>错因分布</h3>
        <div class="class-err-row">${errRow}</div>
        <h3>最近批改</h3>
        <table class="admin-user-table"><thead><tr><th>学生</th><th>分数</th><th>时间</th></tr></thead><tbody>${recent}</tbody></table>
      </div>
    `;
  } catch (err) {
    body.innerHTML = `<div class="empty-state">${err.message || "报表加载失败"}</div>`;
  }
}

function closeClassReport() {
  $("classReportModal")?.classList.add("hidden");
}

// ============================================================
// Admin: KG node management
// ============================================================

// Dispatcher: admin edits the base ontology; teacher edits a class-scoped
// override layer. Both render into #adminKgList so the same tab shell works.
let kgEditingClassId = null;

async function loadClassKgView() {
  const isTeacher = authUser?.role === "teacher";
  if (isTeacher) {
    await loadTeacherClassKgPicker();
  } else {
    await loadAdminKgNodes();
  }
}

async function loadTeacherClassKgPicker() {
  const wrap = $("adminKgList");
  const toolbar = document.querySelector("[data-admin-pane='kg'] .admin-kg-toolbar");
  if (toolbar) toolbar.classList.add("hidden");
  if (!wrap) return;
  wrap.innerHTML = `<div class="empty-state">加载中...</div>`;
  try {
    const resp = await apiFetchJson("/api/classes");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "班级加载失败");
    const items = data.items || [];
    if (!items.length) {
      wrap.innerHTML = `<div class="empty-state">你还没有班级。先到「班级」tab 创建。</div>`;
      return;
    }
    wrap.innerHTML = `
      <div class="kg-class-picker">
        <p class="kg-class-hint">选择班级来在基础知识点之上做定制：</p>
        <div class="kg-class-chips">
          ${items.map((c) => `<button type="button" class="kg-class-chip" data-id="${c.id}">${escapeHtml(c.name)}</button>`).join("")}
        </div>
        <div class="kg-class-target" id="kgClassTarget"></div>
      </div>
    `;
    wrap.querySelectorAll(".kg-class-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        kgEditingClassId = Number(btn.dataset.id);
        wrap.querySelectorAll(".kg-class-chip").forEach((b) => b.classList.toggle("active", b === btn));
        loadTeacherClassKg(kgEditingClassId);
      });
    });
    // Auto-select first class.
    const firstChip = wrap.querySelector(".kg-class-chip");
    if (firstChip) firstChip.click();
  } catch (err) {
    wrap.innerHTML = `<div class="empty-state">${err.message || "加载失败"}</div>`;
  }
}

async function loadTeacherClassKg(classId) {
  const target = $("kgClassTarget");
  if (!target) return;
  target.innerHTML = `<div class="empty-state">加载中...</div>`;
  try {
    const resp = await apiFetchJson(`/api/teacher/classes/${classId}/kg`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "知识点加载失败");
    const nodes = data.nodes || [];
    target.innerHTML = `
      <div class="kg-class-toolbar">
        <button type="button" class="secondary-btn teacher-kg-new">为该班新增节点</button>
        <span class="kg-override-hint">基础节点 + 本班定制（已删除的节点会从该班评分中隐藏，不影响其他班级）</span>
      </div>
      <div class="kg-class-nodes"></div>
    `;
    const list = target.querySelector(".kg-class-nodes");
    if (!nodes.length) {
      list.innerHTML = `<div class="empty-state">暂无节点。</div>`;
    } else {
      list.innerHTML = nodes.map((n) => `
        <article class="kg-admin-card ${n.class_override ? "kg-card-override" : ""}" data-id="${escapeHtml(n.id)}">
          <header>
            <strong>${escapeHtml(n.id)} · ${escapeHtml(n.name || "")}</strong>
            <span class="kg-stage-tag">${escapeHtml(n.stage || "")} · G${escapeHtml(String(n.grade || 0))}${n.class_override ? " · 本班定制" : ""}</span>
          </header>
          <div class="kg-admin-meta">
            <div>前置: ${(n.prerequisites || []).map(escapeHtml).join(", ") || "（无）"}</div>
            <div>关键词: ${(n.keyword_patterns || []).map(escapeHtml).join(", ") || "（无）"}</div>
          </div>
          <p class="kg-admin-desc">${escapeHtml(n.description || "")}</p>
          <div class="kg-admin-actions">
            ${n.class_override && !n.class_added ? `<button type="button" class="text-action kg-restore-btn" data-id="${escapeHtml(n.id)}">还原为基础</button>` : ""}
            ${n.class_added ? `<button type="button" class="text-action kg-del-override-btn" data-id="${escapeHtml(n.id)}">删除本班节点</button>` : ""}
            <button type="button" class="secondary-btn kg-edit-btn" data-id="${escapeHtml(n.id)}">编辑本班副本</button>
            ${!n.class_override ? `<button type="button" class="text-action kg-hide-btn" data-id="${escapeHtml(n.id)}">从本班隐藏</button>` : ""}
          </div>
        </article>
      `).join("");
    }
    target.querySelector(".teacher-kg-new")?.addEventListener("click", () => {
      teacherOpenKgEditor(classId, null);
    });
    list.querySelectorAll(".kg-edit-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const node = nodes.find((n) => n.id === btn.dataset.id);
        if (node) teacherOpenKgEditor(classId, node);
      });
    });
    list.querySelectorAll(".kg-hide-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.dataset.id;
        if (!confirm(`从本班隐藏 ${id}？不影响基础知识点和其他班级。`)) return;
        try {
          const r = await apiFetchJson(`/api/teacher/classes/${classId}/kg/nodes/${encodeURIComponent(id)}`, { method: "DELETE" });
          if (!r.ok) throw new Error("操作失败");
          setStatus(`已从本班隐藏 ${id}。`, "info");
          loadTeacherClassKg(classId);
        } catch (err) { setStatus(err.message || "操作失败", "error"); }
      });
    });
    list.querySelectorAll(".kg-restore-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.dataset.id;
        try {
          const r = await apiFetchJson(`/api/teacher/classes/${classId}/kg/nodes/${encodeURIComponent(id)}/restore`, { method: "POST" });
          if (!r.ok) throw new Error("操作失败");
          setStatus(`已还原 ${id} 为基础版本。`, "info");
          loadTeacherClassKg(classId);
        } catch (err) { setStatus(err.message || "操作失败", "error"); }
      });
    });
    list.querySelectorAll(".kg-del-override-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.dataset.id;
        if (!confirm(`删除本班新增的节点 ${id}？`)) return;
        try {
          const r = await apiFetchJson(`/api/teacher/classes/${classId}/kg/nodes/${encodeURIComponent(id)}/restore`, { method: "POST" });
          if (!r.ok) throw new Error("操作失败");
          setStatus(`已删除 ${id}。`, "info");
          loadTeacherClassKg(classId);
        } catch (err) { setStatus(err.message || "操作失败", "error"); }
      });
    });
  } catch (err) {
    target.innerHTML = `<div class="empty-state">${err.message || "加载失败"}</div>`;
  }
}

function teacherOpenKgEditor(classId, node) {
  const isNew = !node;
  const n = node || { id: "", name: "", stage: "primary", grade: 0, prerequisites: [], keyword_patterns: [], error_type_hints: [], description: "" };
  const id = prompt("节点 ID（如 M14，唯一）:", n.id);
  if (id === null) return;
  const name = prompt("节点名称:", n.name);
  if (name === null) return;
  const stage = prompt("学段 (primary / middle / high):", n.stage || "primary") || "primary";
  const grade = prompt("年级（数字 1-12）:", String(n.grade || 0)) || "0";
  const prereq = prompt("前置知识点 ID，逗号分隔（可空）:", (n.prerequisites || []).join(","));
  const keywords = prompt("关键词正则，逗号分隔（可空）:", (n.keyword_patterns || []).join(","));
  const hints = prompt("错误类型提示 (calculation/sign/variable/logic/other)，逗号分隔:", (n.error_type_hints || []).join(","));
  const desc = prompt("描述:", n.description || "");
  const payload = {
    id: id.trim(),
    name: name.trim(),
    stage: stage.trim(),
    grade: Number(grade) || 0,
    prerequisites: (prereq || "").split(",").map((s) => s.trim()).filter(Boolean),
    keyword_patterns: (keywords || "").split(",").map((s) => s.trim()).filter(Boolean),
    error_type_hints: (hints || "").split(",").map((s) => s.trim()).filter(Boolean),
    description: (desc || "").trim(),
  };
  if (!payload.id || !payload.name) {
    setStatus("ID 与名称必填。", "error");
    return;
  }
  teacherSaveClassKgNode(classId, payload, isNew);
}

async function teacherSaveClassKgNode(classId, payload, isNew) {
  try {
    const resp = await apiFetchJson(`/api/teacher/classes/${classId}/kg/nodes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || "保存失败");
    }
    setStatus(`已 ${isNew ? "为本班新增" : "更新本班副本"} ${payload.id}。`, "info");
    loadTeacherClassKg(classId);
  } catch (err) {
    setStatus(err.message || "保存失败", "error");
  }
}



async function loadAdminKgNodes() {
  const wrap = $("adminKgList");
  if (!wrap) return;
  wrap.innerHTML = `<div class="empty-state">加载中...</div>`;
  try {
    const resp = await apiFetchJson("/api/admin/kg/nodes");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "知识点列表加载失败");
    const nodes = data.nodes || [];
    if (!nodes.length) {
      wrap.innerHTML = `<div class="empty-state">暂无知识点节点。</div>`;
      return;
    }
    wrap.innerHTML = nodes.map((n) => `
      <article class="kg-admin-card" data-id="${escapeHtml(n.id)}">
        <header>
          <strong>${escapeHtml(n.id)} · ${escapeHtml(n.name || "")}</strong>
          <span class="kg-stage-tag">${escapeHtml(n.stage || "")} · G${escapeHtml(String(n.grade || 0))}</span>
        </header>
        <div class="kg-admin-meta">
          <div>前置: ${(n.prerequisites || []).map(escapeHtml).join(", ") || "（无）"}</div>
          <div>关键词: ${(n.keyword_patterns || []).map(escapeHtml).join(", ") || "（无）"}</div>
          <div>错误提示: ${(n.error_type_hints || []).map(escapeHtml).join(", ") || "（无）"}</div>
        </div>
        <p class="kg-admin-desc">${escapeHtml(n.description || "")}</p>
        <div class="kg-admin-actions">
          <button type="button" class="secondary-btn kg-edit-btn" data-id="${escapeHtml(n.id)}">编辑</button>
          <button type="button" class="text-action kg-del-btn" data-id="${escapeHtml(n.id)}">删除</button>
        </div>
      </article>
    `).join("");
    wrap.querySelectorAll(".kg-edit-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const node = nodes.find((n) => n.id === btn.dataset.id);
        if (node) adminOpenKgEditor(node);
      });
    });
    wrap.querySelectorAll(".kg-del-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.dataset.id;
        if (!confirm(`确认删除知识点 ${id}？相关 prerequisite 引用会一并清理。`)) return;
        try {
          const resp = await apiFetchJson(`/api/admin/kg/nodes/${encodeURIComponent(id)}`, { method: "DELETE" });
          if (!resp.ok) throw new Error("删除失败");
          setStatus(`已删除 ${id}。`, "info");
          loadAdminKgNodes();
        } catch (err) {
          setStatus(err.message || "删除失败", "error");
        }
      });
    });
  } catch (err) {
    wrap.innerHTML = `<div class="empty-state">${err.message || "加载失败"}</div>`;
  }
}

function adminOpenKgEditor(node) {
  const isNew = !node;
  const n = node || { id: "", name: "", stage: "primary", grade: 0, prerequisites: [], keyword_patterns: [], error_type_hints: [], description: "" };
  const id = prompt("节点 ID（如 M14，唯一）:", n.id);
  if (id === null) return;
  const name = prompt("节点名称:", n.name);
  if (name === null) return;
  const stage = prompt("学段 (primary / middle / high):", n.stage || "primary") || "primary";
  const grade = prompt("年级（数字 1-12）:", String(n.grade || 0)) || "0";
  const prereq = prompt("前置知识点 ID，逗号分隔（可空）:", (n.prerequisites || []).join(","));
  const keywords = prompt("关键词正则，逗号分隔（可空）:", (n.keyword_patterns || []).join(","));
  const hints = prompt("错误类型提示 (calculation/sign/variable/logic/other)，逗号分隔:", (n.error_type_hints || []).join(","));
  const desc = prompt("描述:", n.description || "");
  const payload = {
    id: id.trim(),
    name: name.trim(),
    stage: stage.trim(),
    grade: Number(grade) || 0,
    prerequisites: (prereq || "").split(",").map((s) => s.trim()).filter(Boolean),
    keyword_patterns: (keywords || "").split(",").map((s) => s.trim()).filter(Boolean),
    error_type_hints: (hints || "").split(",").map((s) => s.trim()).filter(Boolean),
    description: (desc || "").trim(),
  };
  if (!payload.id || !payload.name) {
    setStatus("ID 与名称必填。", "error");
    return;
  }
  adminSaveKgNode(payload, isNew);
}

async function adminSaveKgNode(payload, isNew) {
  try {
    const resp = await apiFetchJson("/api/admin/kg/nodes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || "保存失败");
    }
    setStatus(`已 ${isNew ? "新增" : "更新"} 节点 ${payload.id}。`, "info");
    loadAdminKgNodes();
  } catch (err) {
    setStatus(err.message || "保存失败", "error");
  }
}

// ============================================================
// 错题本（wrong-answer book）
// ------------------------------------------------------------
// 数据来源：批改落库时，后端 db.populate_wrong_answers_from_session 会
//   把每道题里失分严重的步骤抽出来，附上题干、错因、知识点关联，
//   存到 wrong_answers 表。前端按状态（未复习/复习中/已掌握）+
//   知识点筛选展示，允许学生写笔记、改状态。
// ============================================================

// 错题状态标签：后端返回 new/reviewing/mastered，前端翻译成中文。
const WRONG_STATUS_LABELS = { new: "未复习", reviewing: "复习中", mastered: "已掌握" };
// 错因标签：与 knowledge_graph.infer_error_type 的输出对齐。
const WRONG_ERROR_LABELS = { calculation: "计算", sign: "符号", variable: "变量", logic: "方法", other: "其他" };

async function loadWrongAnswersPage() {
  // 错题本 tab 入口：并发拉统计 + 列表。
  await Promise.all([loadWrongStats(), loadWrongAnswers()]);
}

async function loadWrongStats() {
  // 拉错题统计（总数 / 按状态分 / 按错因分 / top 知识点），
  // 渲染成一句中文摘要，并填充知识点筛选下拉。
  const wrap = $("wrongStats");
  if (!wrap) return;
  try {
    const resp = await apiFetchJson("/api/wrong-answers/stats");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "统计加载失败");
    // 拼接摘要："共 12 道错题 · 未复习 8 · 复习中 3 · 已掌握 1 · | 计算 5 / 符号 4"
    const parts = [`共 ${data.total || 0} 道错题`];
    for (const k of ["new", "reviewing", "mastered"]) {
      if (data.by_status && data.by_status[k]) parts.push(`${WRONG_STATUS_LABELS[k]} ${data.by_status[k]}`);
    }
    const errParts = [];
    for (const [k, v] of Object.entries(data.by_error_type || {})) {
      if (WRONG_ERROR_LABELS[k]) errParts.push(`${WRONG_ERROR_LABELS[k]} ${v}`);
    }
    if (errParts.length) parts.push(`| ${errParts.join(" / ")}`);
    wrap.textContent = parts.join(" · ");
    // 把 top_kg_nodes 灌进知识点筛选下拉。保留用户当前选择不变。
    const sel = $("wrongFilterKg");
    if (sel) {
      const currentValue = sel.value;
      sel.innerHTML = '<option value="">全部知识点</option>' +
        (data.top_kg_nodes || []).map((n) => `<option value="${escapeHtml(n.id)}">${escapeHtml(n.name || n.id)} (${n.count})</option>`).join("");
      if (currentValue) sel.value = currentValue;
    }
  } catch (err) {
    wrap.textContent = err.message || "统计加载失败";
  }
}

function renderWrongSteps(container, steps, stepSummary) {
  // 渲染一道错题的步骤列表。
  // 优先用结构化 steps_json（每步独立带 wrong 标记），可精准高亮哪步错；
  // 没有结构化数据时回退到 stepSummary（旧记录），用 `[reason]` 标记解析错因。
  if (!container) return;
  container.innerHTML = "";
  const structured = Array.isArray(steps) && steps.length;
  if (!structured) {
    // 旧记录 fallback：把 stepSummary 字符串按行 + `[xxx]` 标记解析成
    // {text, reasons} 列表，渲染成简单列表。
    const raw = String(stepSummary || "").trim();
    if (!raw) {
      container.innerHTML = '<div class="wrong-steps-empty">（无步骤记录）</div>';
      return;
    }
    const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    let lastReason = "";
    const parsed = [];
    for (const line of lines) {
      const m = line.match(/^\[(.+)\]$/);
      if (m) {
        // 单独一行的 [xxx]：是上一步的错因说明，挂到上一步的 reasons 里。
        const reason = m[1].trim();
        if (reason === lastReason) continue;  // 连续重复的错因去重
        lastReason = reason;
        const tail = parsed[parsed.length - 1];
        if (tail) tail.reasons.push(reason);
        else parsed.push({ text: "", reasons: [reason] });
      } else {
        parsed.push({ text: line, reasons: [] });
        lastReason = "";
      }
    }
    for (const step of parsed) {
      const li = document.createElement("div");
      li.className = "wrong-step";
      const body = document.createElement("div");
      body.className = "wrong-step-body";
      if (step.text) renderRichLine(body, step.text, false);
      li.appendChild(body);
      if (step.reasons.length) {
        const r = document.createElement("div");
        r.className = "wrong-step-reasons";
        r.textContent = step.reasons.map((x) => `[${x}]`).join(" ");
        li.appendChild(r);
      }
      container.appendChild(li);
    }
    return;
  }
  const wrongCount = steps.filter((s) => s && s.wrong).length;
  if (wrongCount > 0) {
    const banner = document.createElement("div");
    banner.className = "wrong-steps-banner";
    banner.textContent = `${wrongCount} / ${steps.length} 步出错`;
    container.appendChild(banner);
  }
  steps.forEach((step, i) => {
    const li = document.createElement("div");
    li.className = "wrong-step" + (step.wrong ? " wrong-step-error" : "");
    const head = document.createElement("div");
    head.className = "wrong-step-head";
    const idx = document.createElement("span");
    idx.className = "wrong-step-idx";
    idx.textContent = `${i + 1}`;
    head.appendChild(idx);
    if (step.wrong) {
      const badge = document.createElement("span");
      badge.className = "wrong-step-badge";
      badge.textContent = `失分 ${Number(step.score).toFixed(1)} / ${Number(step.max_score).toFixed(1)}`;
      head.appendChild(badge);
    } else {
      const ok = document.createElement("span");
      ok.className = "wrong-step-ok";
      ok.textContent = "正确";
      head.appendChild(ok);
    }
    li.appendChild(head);
    const body = document.createElement("div");
    body.className = "wrong-step-body";
    if (step.text) renderRichLine(body, step.text, false);
    li.appendChild(body);
    if (step.reason && step.wrong) {
      const r = document.createElement("div");
      r.className = "wrong-step-reasons";
      r.textContent = `[${step.reason}]`;
      li.appendChild(r);
    }
    container.appendChild(li);
  });
}

function toggleWrongNoteEditor(card) {
  if (!card) return;
  let editor = card.querySelector(".wrong-note-editor");
  if (editor) {
    editor.classList.toggle("hidden");
    return;
  }
  editor = document.createElement("div");
  editor.className = "wrong-note-editor";
  const noteEl = card.querySelector(".wrong-note-text");
  const current = noteEl ? noteEl.textContent : "";
  editor.innerHTML = `
    <textarea rows="3" placeholder="如：容易在符号上出错；下次先化简再代入"></textarea>
    <div class="wrong-note-actions">
      <button type="button" class="secondary-btn wrong-note-save">保存</button>
      <button type="button" class="text-action wrong-note-cancel">取消</button>
    </div>
  `;
  editor.querySelector("textarea").value = current;
  card.querySelector(".wrong-actions").insertAdjacentElement("beforebegin", editor);
  editor.querySelector(".wrong-note-save").addEventListener("click", async () => {
    const note = editor.querySelector("textarea").value.trim();
    try {
      const resp = await apiFetchJson(`/api/wrong-answers/${card.dataset.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ note }),
      });
      if (!resp.ok) throw new Error("保存失败");
      setStatus("笔记已保存。", "info");
      loadWrongAnswersPage();
    } catch (err) {
      setStatus(err.message || "保存失败", "error");
    }
  });
  editor.querySelector(".wrong-note-cancel").addEventListener("click", () => {
    editor.classList.add("hidden");
  });
  editor.querySelector("textarea").focus();
}

async function loadWrongAnswers() {
  const wrap = $("wrongAnswersList");
  if (!wrap) return;
  const status = ($("wrongFilterStatus")?.value || "").trim();
  const kg = ($("wrongFilterKg")?.value || "").trim();
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (kg) params.set("kg_node", kg);
  params.set("limit", "100");
  wrap.innerHTML = `<div class="empty-state">加载中...</div>`;
  try {
    const resp = await apiFetchJson(`/api/wrong-answers?${params.toString()}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "错题列表加载失败");
    const items = data.items || [];
    if (!items.length) {
      wrap.innerHTML = `<div class="empty-state">暂无错题记录。完成批改后，未得满分的题目会自动入库。</div>`;
      return;
    }
    wrap.innerHTML = items.map((it) => {
      const kgTags = (it.kg_nodes || [])
        .map((n) => `<span class="wrong-kg" title="${escapeHtml(n.id)}">${escapeHtml(n.name || n.id)}</span>`)
        .join("");
      const qHtml = it.question_text
        ? `<div class="wrong-q" data-q="${escapeHtml(it.question_text)}"></div>`
        : "";
      return `
      <article class="wrong-card" data-id="${it.id}">
        <header>
          <div>
            <strong>第 ${it.qno} 题</strong>
            <span class="wrong-score">${Number(it.score).toFixed(1)} / ${Number(it.max_score).toFixed(1)}</span>
            <span class="wrong-tag wrong-tag-${it.error_type}">${WRONG_ERROR_LABELS[it.error_type] || it.error_type}</span>
            ${kgTags}
          </div>
          <select class="wrong-status-select" data-id="${it.id}">
            ${Object.entries(WRONG_STATUS_LABELS).map(([k, v]) => `<option value="${k}" ${k === it.status ? "selected" : ""}>${v}</option>`).join("")}
          </select>
        </header>
        ${qHtml}
        <div class="wrong-steps" data-summary="${escapeHtml(it.step_summary || "")}" data-steps="${escapeHtml(JSON.stringify(it.steps || []))}"></div>
        ${it.note ? `<div class="wrong-note">笔记：<span class="wrong-note-text">${escapeHtml(it.note)}</span></div>` : ""}
        <div class="wrong-actions">
          <button type="button" class="text-action wrong-note-btn">${it.note ? "编辑笔记" : "记笔记"}</button>
          <button type="button" class="text-action wrong-del-btn">删除</button>
          <span class="wrong-time">${formatTime(it.created_at)}</span>
        </div>
      </article>
    `;
    }).join("");
    // Render question (LaTeX) + steps (LaTeX) after DOM insert.
    wrap.querySelectorAll(".wrong-card").forEach((card) => {
      const qEl = card.querySelector(".wrong-q");
      if (qEl && qEl.dataset.q) renderRichLine(qEl, qEl.dataset.q, false);
      const stepsEl = card.querySelector(".wrong-steps");
      let parsed = [];
      if (stepsEl && stepsEl.dataset.steps) {
        try { parsed = JSON.parse(stepsEl.dataset.steps); } catch (_) { parsed = []; }
      }
      if (stepsEl) renderWrongSteps(stepsEl, parsed, stepsEl.dataset.summary);
    });
    wrap.querySelectorAll(".wrong-status-select").forEach((sel) => {
      sel.addEventListener("change", async (ev) => {
        await updateWrongStatus(Number(ev.target.dataset.id), ev.target.value);
      });
    });
    wrap.querySelectorAll(".wrong-del-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const card = btn.closest(".wrong-card");
        const id = Number(card?.dataset.id);
        if (!confirm("确认删除这条错题记录？")) return;
        try {
          const resp = await apiFetchJson(`/api/wrong-answers/${id}`, { method: "DELETE" });
          if (!resp.ok) throw new Error("删除失败");
          setStatus("已删除。", "info");
          loadWrongAnswersPage();
        } catch (err) {
          setStatus(err.message || "删除失败", "error");
        }
      });
    });
    wrap.querySelectorAll(".wrong-note-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const card = btn.closest(".wrong-card");
        toggleWrongNoteEditor(card);
      });
    });
  } catch (err) {
    wrap.innerHTML = `<div class="empty-state">${err.message || "加载失败"}</div>`;
  }
}

async function updateWrongStatus(id, status) {
  try {
    const resp = await apiFetchJson(`/api/wrong-answers/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    if (!resp.ok) throw new Error("更新失败");
    setStatus(`已标记为 ${WRONG_STATUS_LABELS[status] || status}。`, "info");
    loadWrongStats();
  } catch (err) {
    setStatus(err.message || "更新失败", "error");
    loadWrongAnswers();
  }
}

// ============================================================
// Student: my classes
// ============================================================

async function loadMyClasses() {
  const wrap = $("myClassesList");
  if (!wrap) return;
  wrap.innerHTML = `<div class="empty-state">加载中...</div>`;
  try {
    const resp = await apiFetchJson("/api/classes");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "班级加载失败");
    const items = data.items || [];
    if (!items.length) {
      wrap.innerHTML = `<div class="empty-state">还没有加入任何班级。上方输入教师提供的邀请码加入。</div>`;
      return;
    }
    wrap.innerHTML = items.map((c) => `
      <article class="my-class-card" data-id="${c.id}">
        <header>
          <strong>${escapeHtml(c.name)}</strong>
          <span class="class-stage-tag">${CLASS_STAGE_LABELS[c.stage] || c.stage} · G${c.grade || 0}</span>
        </header>
        <div class="class-card-meta">
          <span>成员 ${c.member_count || 0}</span>
        </div>
        ${c.description ? `<p class="class-desc">${escapeHtml(c.description)}</p>` : ""}
        <div class="class-card-actions">
          <button type="button" class="text-action my-class-leave-btn">退出班级</button>
        </div>
      </article>
    `).join("");
    wrap.querySelectorAll(".my-class-card").forEach((card) => {
      card.querySelector(".my-class-leave-btn")?.addEventListener("click", async () => {
        if (!confirm("确认退出该班级？退出后教师将看不到你的批改统计。")) return;
        try {
          const r = await apiFetchJson(`/api/classes/${card.dataset.id}/leave`, { method: "POST" });
          if (!r.ok) throw new Error("退出失败");
          setStatus("已退出班级。", "info");
          loadMyClasses();
        } catch (err) {
          setStatus(err.message || "退出失败", "error");
        }
      });
    });
  } catch (err) {
    wrap.innerHTML = `<div class="empty-state">${err.message || "班级加载失败"}</div>`;
  }
}

async function joinMyClass() {
  const input = $("myClassInviteInput");
  if (!input) return;
  const code = input.value.trim();
  if (!code) {
    setStatus("请输入邀请码。", "error");
    return;
  }
  try {
    const resp = await apiFetchJson("/api/classes/join", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ invite_code: code }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "加入失败");
    setStatus(`已加入「${data.class?.name || "班级"}」。`, "info");
    input.value = "";
    loadMyClasses();
  } catch (err) {
    setStatus(err.message || "加入失败", "error");
  }
}

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function bindEvents() {
  document.querySelectorAll(".top-nav a, .brand, .header-actions a, .primary-link").forEach((link) => {
    link.addEventListener("click", (event) => {
      const href = link.getAttribute("href");
      if (!href || href.startsWith("http")) return;
      event.preventDefault();
      history.pushState(null, "", href);
      routePage();
    });
  });
  window.addEventListener("popstate", routePage);
  globalStatus?.addEventListener("click", () => {
    if (globalStatusTimer) {
      clearTimeout(globalStatusTimer);
      globalStatusTimer = null;
    }
    globalStatus?.classList.add("hidden");
  });
  loginEntryBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    openAuthModal();
  });
  authModalMask?.addEventListener("click", closeAuthModal);
  closeAuthModalBtn?.addEventListener("click", closeAuthModal);
  imageInput?.addEventListener("change", () => {
    const newFiles = Array.from(imageInput.files || []);
    handleIncomingFiles(newFiles);
  });
  const dropZone = document.querySelector(".drop-zone");
  if (dropZone) {
    dropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", (event) => {
      if (event.target === dropZone) dropZone.classList.remove("drag-over");
    });
    dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropZone.classList.remove("drag-over");
      const files = Array.from(event.dataTransfer?.files || []);
      handleIncomingFiles(files);
    });
  }
  clearPagesBtn?.addEventListener("click", () => {
    if (!assignmentPages.length) return;
    clearAssignmentPages();
    setStatus("已清空作业页。", "info");
  });
  togglePasteBtn?.addEventListener("click", () => {
    pastePanel?.classList.toggle("hidden");
    const open = !pastePanel?.classList.contains("hidden");
    if (togglePasteBtn) togglePasteBtn.textContent = open ? "收起粘贴" : "或粘贴文本批改";
    if (open) manualTextInput?.focus();
  });
  ocrBtn?.addEventListener("click", readWholeAssignment);
  reOcrBtn?.addEventListener("click", readCurrentPage);
  confirmReviewBtn?.addEventListener("click", () => {
    showStage(3);
    renderQuestionWeightList();
  });
  ocrTextInput?.addEventListener("input", () => {
    if (currentPage === 3) {
      renderQuestionWeightList();
    }
  });
  prevQuestionBtn?.addEventListener("click", () => {
    if (currentQuestionIndex > 0) {
      currentQuestionIndex -= 1;
      applyQuestionPage();
    }
  });
  nextQuestionBtn?.addEventListener("click", () => {
    if (currentQuestionIndex < currentQuestionGroups.length - 1) {
      currentQuestionIndex += 1;
      applyQuestionPage();
    }
  });
  prevPageBtn?.addEventListener("click", () => {
    if (currentBatchIndex > 0) switchToPage(currentBatchIndex - 1);
  });
  nextPageBtn?.addEventListener("click", () => {
    if (currentBatchIndex < assignmentPages.length - 1) switchToPage(currentBatchIndex + 1);
  });
  deleteQuestionBtn?.addEventListener("click", () => {
    if (!currentQuestionGroups.length) return;
    if (window.confirm(`确认删除第 ${currentQuestionIndex + 1} 题？此操作仅影响本次批改，不会修改原作业。`)) {
      deleteCurrentQuestion();
    }
  });
  ocrTextInput?.addEventListener("input", () => {
    const value = ocrTextInput.value;
    if (currentQuestionGroups.length) {
      const group = currentQuestionGroups[currentQuestionIndex];
      if (group) group.text = value;
    } else if (assignmentPages[currentBatchIndex]) {
      assignmentPages[currentBatchIndex].ocrText = value;
    }
    renderOcrPanel(value);
    renderStepPreviewFromText(value);
  });
  gradeForm?.addEventListener("submit", gradeHomework);
  $("resultPrevQuestionBtn")?.addEventListener("click", () => renderResultQuestion(currentResultQuestionIdx - 1));
  $("resultNextQuestionBtn")?.addEventListener("click", () => renderResultQuestion(currentResultQuestionIdx + 1));
  $("pauseGradeBtn")?.addEventListener("click", () => {
    if (gradingAbort) return;
    if (gradingPaused) {
      togglePauseGrading();
    } else if (confirm("暂停批改？完成当前题目后会停下，可点击「继续批改」恢复，或双击按钮终止。")) {
      togglePauseGrading();
    }
  });
  $("pauseGradeBtn")?.addEventListener("dblclick", () => {
    if (confirm("终止本次批改？将完成当前题目后结束，并保留已批改结果。")) {
      abortGrading();
    }
  });
  downloadSimpleReportBtn?.addEventListener("click", () => {
    if (!lastGradeResult) {
      setStatus("暂无可导出的批改报告。", "error");
      return;
    }
    downloadTextFile("math-grade-report-simple.md", buildSimpleReportMarkdown(lastGradeResult));
  });
  downloadDetailReportBtn?.addEventListener("click", () => {
    if (!cachedDetailReport.markdown) {
      setStatus("精确报告尚未就绪。", "error");
      return;
    }
    downloadTextFile("math-grade-detail-report.md", cachedDetailReport.markdown);
  });
  previewDetailReportBtn?.addEventListener("click", openDetailReportModal);
  regenerateDetailReportBtn?.addEventListener("click", regenerateDetailReport);
  closeDetailReportBtn?.addEventListener("click", closeDetailReportModal);
  detailReportMask?.addEventListener("click", closeDetailReportModal);
  downloadDetailFromModalBtn?.addEventListener("click", () => {
    if (!cachedDetailReport.markdown) return;
    downloadTextFile("math-grade-detail-report.md", cachedDetailReport.markdown);
  });
  favoriteBtn?.addEventListener("click", saveFavorite);
  newAssignmentBtn?.addEventListener("click", resetAssignment);
  accountTabs?.querySelectorAll(".account-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      if (!tab) return;
      switchAccountTab(tab);
      if (authToken) {
        if (tab === "ocr" || tab === "grading") loadHistory();
        if (tab === "favorites") loadFavorites();
        if (tab === "profile") loadDashboard();
        if (tab === "wrong-answers") loadWrongAnswersPage();
        if (tab === "my-classes") loadMyClasses();
      }
    });
  });
  refreshAccountBtn?.addEventListener("click", () => {
    if (!authToken) return;
    loadDashboard();
    loadHistory();
    loadFavorites();
    if (currentAccountTab === "wrong-answers") loadWrongAnswersPage();
  });
  loadAdminBtn?.addEventListener("click", loadAdminOverview);
  document.querySelectorAll("[data-admin-tab]").forEach((btn) => {
    btn.addEventListener("click", () => switchAdminTab(btn.dataset.adminTab));
  });
  $("adminKgNewBtn")?.addEventListener("click", () => adminOpenKgEditor(null));
  $("adminKgRefreshBtn")?.addEventListener("click", loadAdminKgNodes);
  $("closeClassEditorBtn")?.addEventListener("click", closeClassEditor);
  $("classEditorMask")?.addEventListener("click", closeClassEditor);
  $("classEditorSubmitBtn")?.addEventListener("click", submitClassEditor);
  $("closeClassReportBtn")?.addEventListener("click", closeClassReport);
  $("classReportMask")?.addEventListener("click", closeClassReport);
  $("myClassJoinBtn")?.addEventListener("click", joinMyClass);
  $("myClassInviteInput")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") joinMyClass();
  });
  $("wrongFilterStatus")?.addEventListener("change", loadWrongAnswers);
  $("wrongFilterKg")?.addEventListener("change", loadWrongAnswers);
  loginForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const username = authUsernameInput?.value.trim() || "";
    const password = authPasswordInput?.value.trim() || "";
    if (!username || !password) {
      setAuthError("请输入用户名和密码。");
      return;
    }
    if (authMode === "register") {
      if (username.length < 3) {
        setAuthError("用户名至少 3 位。");
        return;
      }
      if (password.length < 6) {
        setAuthError("密码至少 6 位。");
        return;
      }
    }
    const fd = new FormData();
    fd.append("username", username);
    fd.append("password", password);
    const endpoint = authMode === "register" ? "/api/auth/register" : "/api/auth/login";
    if (authMode === "register") {
      const selectedRole = document.querySelector('input[name="authRole"]:checked');
      fd.append("role", selectedRole?.value || "student");
    }
    if (loginBtn) loginBtn.disabled = true;
    try {
      const resp = await apiFetch(endpoint, { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) {
        setAuthError(data.detail || (authMode === "register" ? "注册失败。" : "用户名或密码错误。"));
        return;
      }
      setAuth(data.access_token, data.user);
      closeAuthModal();
    } catch (err) {
      setAuthError(err.message || "网络异常，请重试。");
    } finally {
      if (loginBtn) loginBtn.disabled = false;
    }
  });
  authModeSwitch?.addEventListener("click", () => {
    setAuthMode(authMode === "register" ? "login" : "register");
    setAuthError("");
  });
  logoutBtn?.addEventListener("click", () => {
    setAuth("", null);
    history.replaceState(null, "", "/");
    routePage();
  });
  adminLogoutBtn?.addEventListener("click", () => {
    setAuth("", null);
    history.replaceState(null, "", "/");
    routePage();
  });
}

bindEvents();
bindStepperNavigation();
renderBatchList();
renderOcrPanel("");
setDetailReportButtonState("unavailable");
showStage(1);
(async () => {
  await loadCurrentUser();
  routePage();
})();
