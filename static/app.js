const ocrForm = document.getElementById("ocrForm");
const gradeForm = document.getElementById("gradeForm");
const loginForm = document.getElementById("loginForm");
const registerBtn = document.getElementById("registerBtn");
const loginBtn = document.getElementById("loginBtn");
const logoutBtn = document.getElementById("logoutBtn");
const authUsernameInput = document.getElementById("authUsername");
const authPasswordInput = document.getElementById("authPassword");
const authRoleInput = document.getElementById("authRole");
const authInfo = document.getElementById("authInfo");

const imageInput = document.getElementById("image");
const imagePreview = document.getElementById("imagePreview");
const imagePlaceholder = document.getElementById("imagePlaceholder");
const manualTextInput = document.getElementById("manualText");
const useLlmCorrectionInput = document.getElementById("useLlmCorrection");
const returnLlmDebugInput = document.getElementById("returnLlmDebug");
const useVisionCorrectionInput = document.getElementById("useVisionCorrection");
const ocrTextInput = document.getElementById("ocrText");
const referenceSolutionInput = document.getElementById("referenceSolution");
const useLlmInput = document.getElementById("useLlm");
const gradeCurrentQuestionOnlyInput = document.getElementById("gradeCurrentQuestionOnly");

const ocrStatus = document.getElementById("ocrStatus");
const stepPreview = document.getElementById("stepPreview");
const ocrRendered = document.getElementById("ocrRendered");
const ocrMeta = document.getElementById("ocrMeta");
const segmentEditor = document.getElementById("segmentEditor");
const applySegmentsBtn = document.getElementById("applySegmentsBtn");
const questionGroupsEl = document.getElementById("questionGroups");
const questionPager = document.getElementById("questionPager");
const questionPageInfo = document.getElementById("questionPageInfo");
const prevQuestionBtn = document.getElementById("prevQuestionBtn");
const nextQuestionBtn = document.getElementById("nextQuestionBtn");

const resultSection = document.getElementById("resultSection");
const totalScoreEl = document.getElementById("totalScore");
const feedbackEl = document.getElementById("feedback");
const stepTable = document.getElementById("stepTable");
const gradeDiagWrap = document.getElementById("gradeDiagWrap");
const gradeDiag = document.getElementById("gradeDiag");

const ocrBtn = document.getElementById("ocrBtn");
const visionOnlyBtn = document.getElementById("visionOnlyBtn");
const gradeBtn = document.getElementById("gradeBtn");

let currentSegments = [];
let previewObjectUrl = "";
let currentQuestionGroups = [];
let currentQuestionIndex = 0;
let authToken = localStorage.getItem("auth_token") || "";
let authUser = null;

async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (authToken) headers.set("Authorization", `Bearer ${authToken}`);
  const resp = await fetch(url, { ...options, headers });
  return resp;
}

function renderAuthInfo() {
  if (!authInfo) return;
  if (!authUser) {
    authInfo.textContent = "未登录";
    return;
  }
  authInfo.textContent = `已登录：${authUser.username}（${authUser.role}）`;
}

function setAuth(token, user) {
  authToken = token || "";
  authUser = user || null;
  if (authToken) localStorage.setItem("auth_token", authToken);
  else localStorage.removeItem("auth_token");
  renderAuthInfo();
}

async function loadCurrentUser() {
  if (!authToken) {
    setAuth("", null);
    return;
  }
  try {
    const resp = await apiFetch("/api/auth/me");
    const data = await resp.json();
    if (!resp.ok) {
      setAuth("", null);
      return;
    }
    setAuth(authToken, data);
  } catch (_) {
    setAuth("", null);
  }
}

function requireLogin() {
  if (!authToken) {
    setStatus("请先登录后再使用识别与评分功能。", "error");
    return false;
  }
  return true;
}

function sanitizeUserMessage(text) {
  const raw = String(text || "");
  return raw
    .replace(/LLM_API_KEY/gi, "智能评分服务")
    .replace(/\/responses|\/chat\/completions/gi, "评分接口")
    .replace(/ReadTimeout/gi, "请求超时");
}

function setStatus(message, type = "") {
  ocrStatus.textContent = sanitizeUserMessage(message);
  ocrStatus.className = `status ${type}`.trim();
}

function renderMath(container, latex, displayMode = false) {
  container.innerHTML = "";
  const text = (latex || "").trim();
  if (!text) return;

  if (window.katex) {
    try {
      window.katex.render(text, container, { throwOnError: true, displayMode });
      return;
    } catch (_) {
      // fall through
    }
  }

  const fallback = document.createElement("span");
  fallback.className = "math-fallback";
  fallback.textContent = text;
  container.appendChild(fallback);
}

function renderOcrPanel(text) {
  ocrRendered.innerHTML = "";
  const lines = (text || "")
    .split(/\r?\n/)
    .map((v) => v.trim())
    .filter(Boolean);

  if (!lines.length) {
    ocrRendered.textContent = "等待 OCR 结果...";
    return;
  }

  for (const line of lines) {
    const lineEl = document.createElement("div");
    lineEl.className = "math-line";
    renderMath(lineEl, line, true);
    ocrRendered.appendChild(lineEl);
  }
}

function renderOcrMeta(segments, savedPath, stats, preprocessed, llmDebug, visionCorrection) {
  if (!ocrMeta) return;
  if (!segments || !segments.length) {
    ocrMeta.textContent = "暂无";
    return;
  }
  const lines = [];
  if (savedPath) lines.push(`保存路径: ${savedPath}`);
  if (stats) lines.push(`文本块: ${stats.text_count}，公式块: ${stats.formula_count}，噪声块: ${stats.noisy_count || 0}，总块数: ${stats.total}`);
  lines.push(`图像预处理: ${preprocessed ? "已启用" : "未启用"}`);
  if (visionCorrection && visionCorrection.enabled) {
    lines.push(`图文联合纠错: 已启用，尝试 ${visionCorrection.attempted || 0} 块，修正 ${visionCorrection.corrected_count || 0} 块`);
  }
  for (const seg of segments) {
    const bbox = Array.isArray(seg.bbox) ? seg.bbox.join(",") : "0,0,0,0";
    const txt = (seg.text || "").replace(/\s+/g, " ").slice(0, 70);
    const mark = seg.low_confidence ? " [低置信]" : "";
    const noisy = seg.noisy ? " [噪声]" : "";
    lines.push(
      `#${seg.index} [${seg.type}] score=${(seg.score || 0).toFixed(2)}${mark}${noisy} bbox=(${bbox}) text=${txt}`
    );
  }
  if (llmDebug) {
    lines.push("");
    lines.push("LLM调试-请求预览:");
    lines.push(llmDebug.request_preview || "");
    lines.push("LLM调试-响应预览:");
    lines.push(llmDebug.response_preview || "");
  }
  ocrMeta.textContent = lines.join("\n");
}

function renderQuestionGroups(groups) {
  if (!questionGroupsEl) return;
  if (!groups || !groups.length) {
    questionGroupsEl.textContent = "暂无";
    return;
  }
  const lines = [];
  for (const g of groups) {
    const preview = (g.text || "").replace(/\s+/g, " ").slice(0, 100);
    lines.push(`题号 ${g.question_no}: 块数 ${g.segments.length}，预览：${preview}`);
  }
  questionGroupsEl.textContent = lines.join("\n");
}

function applyQuestionPage() {
  if (!currentQuestionGroups.length) {
    if (questionPager) questionPager.classList.add("hidden");
    return;
  }
  const idx = Math.max(0, Math.min(currentQuestionIndex, currentQuestionGroups.length - 1));
  currentQuestionIndex = idx;
  const g = currentQuestionGroups[idx];
  ocrTextInput.value = (g?.text || "").trim();
  renderOcrPanel(ocrTextInput.value);
  renderStepPreviewFromText(ocrTextInput.value);
  if (questionPager) questionPager.classList.remove("hidden");
  if (questionPageInfo) questionPageInfo.textContent = `题目 ${idx + 1} / ${currentQuestionGroups.length}`;
  if (prevQuestionBtn) prevQuestionBtn.disabled = idx <= 0;
  if (nextQuestionBtn) nextQuestionBtn.disabled = idx >= currentQuestionGroups.length - 1;
}

function setQuestionGroups(groups) {
  currentQuestionGroups = Array.isArray(groups) ? groups.filter((g) => (g?.text || "").trim()) : [];
  currentQuestionIndex = 0;
  applyQuestionPage();
}

function renderSegmentEditor(segments) {
  if (!segmentEditor) return;
  segmentEditor.innerHTML = "";
  if (!segments || !segments.length) {
    segmentEditor.textContent = "暂无";
    return;
  }

  for (const seg of segments) {
    const row = document.createElement("div");
    row.className = `seg-row ${seg.low_confidence ? "low" : ""}`.trim();

    const head = document.createElement("div");
    head.className = "head";
    const bbox = Array.isArray(seg.bbox) ? seg.bbox.join(",") : "0,0,0,0";
    const noisy = seg.noisy ? " [噪声]" : "";
    head.textContent = `#${seg.index} [${seg.type}] score=${(seg.score || 0).toFixed(2)}${noisy} bbox=(${bbox})`;

    const ta = document.createElement("textarea");
    ta.value = seg.text || "";
    ta.dataset.index = String(seg.index);

    row.appendChild(head);
    row.appendChild(ta);
    segmentEditor.appendChild(row);
  }
}

function buildTextFromEditor() {
  const textareas = segmentEditor.querySelectorAll("textarea[data-index]");
  const pairs = [];
  for (const ta of textareas) {
    const idx = Number(ta.dataset.index || "0");
    const text = ta.value.trim();
    if (idx > 0) pairs.push({ idx, text });
  }
  pairs.sort((a, b) => a.idx - b.idx);
  return pairs.map((p) => p.text).filter(Boolean).join("\n");
}

function renderStepPreviewFromText(text) {
  const pseudoSteps = (text || "")
    .split(/\r?\n/)
    .map((v) => v.trim())
    .filter(Boolean)
    .map((line, i) => ({ index: i + 1, normalized: line }));
  renderStepPreview(pseudoSteps);
}

function renderStepPreview(steps) {
  stepPreview.innerHTML = "";
  if (!steps || !steps.length) return;
  for (const step of steps) {
    const li = document.createElement("li");
    renderMath(li, step.normalized, false);
    stepPreview.appendChild(li);
  }
}

function renderRows(steps, scores) {
  stepTable.innerHTML = "";
  const map = new Map(scores.map((s) => [s.index, s]));
  for (const step of steps) {
    const score = map.get(step.index);
    const row = document.createElement("tr");
    const idxTd = document.createElement("td");
    idxTd.textContent = String(step.index);

    const contentTd = document.createElement("td");
    contentTd.className = "math-cell";
    renderMath(contentTd, step.normalized, false);

    const scoreTd = document.createElement("td");
    scoreTd.textContent = score ? score.score.toFixed(2) : "-";

    const reasonTd = document.createElement("td");
    reasonTd.textContent = score ? score.reason : "无";

    row.appendChild(idxTd);
    row.appendChild(contentTd);
    row.appendChild(scoreTd);
    row.appendChild(reasonTd);
    stepTable.appendChild(row);
  }
}

function renderGradeDiag(meta) {
  if (!gradeDiagWrap || !gradeDiag) return;
  const llmUsed = Boolean(meta?.llm_used);
  const err = sanitizeUserMessage(meta?.llm_error || "");
  const diag = meta?.llm_diag || {};
  const attempts = Array.isArray(diag.attempts) ? diag.attempts : [];

  const lines = [];
  lines.push(`评分模式: ${llmUsed ? "大模型评分" : "规则评分(回退)"}`);
  if (err) lines.push(`错误: ${err}`);
  if (diag.timeout_sec) lines.push(`超时: ${diag.timeout_sec}s`);
  if (diag.base_url) lines.push("接口: 已配置");
  if (diag.model) lines.push("模型: 已配置");
  if (attempts.length) {
    lines.push("");
    lines.push("请求尝试:");
    for (const a of attempts) {
      const status = typeof a.status_code === "number" ? a.status_code : "-";
      const ms = typeof a.elapsed_ms === "number" ? `${a.elapsed_ms}ms` : "-";
        lines.push(`- ${a.path} | ok=${a.ok} | status=${status} | elapsed=${ms}`);
        if (a.response_preview) {
        lines.push(`  resp: ${sanitizeUserMessage(String(a.response_preview).replace(/\s+/g, " ").slice(0, 240))}`);
      }
    }
  }

  if (lines.length === 0) {
    gradeDiagWrap.classList.add("hidden");
    gradeDiag.textContent = "暂无";
    return;
  }

  gradeDiag.textContent = lines.join("\n");
  gradeDiagWrap.classList.remove("hidden");
  if (!llmUsed || err) gradeDiagWrap.open = true;
}

if (registerBtn) {
  registerBtn.addEventListener("click", async () => {
    const username = (authUsernameInput?.value || "").trim();
    const password = (authPasswordInput?.value || "").trim();
    const role = (authRoleInput?.value || "student").trim();
    if (!username || !password) {
      setStatus("请输入用户名和密码。", "error");
      return;
    }
    const fd = new FormData();
    fd.append("username", username);
    fd.append("password", password);
    fd.append("role", role);
    try {
      const resp = await apiFetch("/api/auth/register", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "注册失败");
      setAuth(data.access_token, data.user);
      setStatus("注册并登录成功。", "ok");
    } catch (err) {
      setStatus(err.message || "注册失败", "error");
    }
  });
}

if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = (authUsernameInput?.value || "").trim();
    const password = (authPasswordInput?.value || "").trim();
    if (!username || !password) {
      setStatus("请输入用户名和密码。", "error");
      return;
    }
    const fd = new FormData();
    fd.append("username", username);
    fd.append("password", password);
    try {
      const resp = await apiFetch("/api/auth/login", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "登录失败");
      setAuth(data.access_token, data.user);
      setStatus("登录成功。", "ok");
    } catch (err) {
      setStatus(err.message || "登录失败", "error");
    }
  });
}

if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    setAuth("", null);
    setStatus("已退出登录。", "ok");
  });
}

applySegmentsBtn.addEventListener("click", () => {
  if (!segmentEditor) return;
  const merged = buildTextFromEditor();
  ocrTextInput.value = merged;
  renderOcrPanel(merged);
  renderStepPreviewFromText(merged);
  setStatus("已应用修正。", "ok");
});

if (prevQuestionBtn) {
  prevQuestionBtn.addEventListener("click", () => {
    if (currentQuestionIndex > 0) {
      currentQuestionIndex -= 1;
      applyQuestionPage();
    }
  });
}

if (nextQuestionBtn) {
  nextQuestionBtn.addEventListener("click", () => {
    if (currentQuestionIndex < currentQuestionGroups.length - 1) {
      currentQuestionIndex += 1;
      applyQuestionPage();
    }
  });
}

function updateImagePreview(file) {
  if (!imagePreview || !imagePlaceholder) return;
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = "";
  if (!file) {
    imagePreview.style.display = "none";
    imagePreview.removeAttribute("src");
    imagePlaceholder.style.display = "flex";
    return;
  }
  previewObjectUrl = URL.createObjectURL(file);
  imagePreview.src = previewObjectUrl;
  imagePreview.style.display = "block";
  imagePlaceholder.style.display = "none";
}

visionOnlyBtn.addEventListener("click", async () => {
  if (!requireLogin()) return;
  const image = imageInput.files[0];
  if (!image) {
    setStatus("请先选择图片，再执行纯大模型OCR。", "error");
    return;
  }

  const fd = new FormData();
  fd.append("image", image);
  fd.append("return_llm_debug", returnLlmDebugInput.checked ? "true" : "false");

  visionOnlyBtn.disabled = true;
  setStatus("纯大模型OCR处理中...");

  try {
    const resp = await apiFetch("/api/ocr-vision-only", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "纯大模型OCR失败");

    currentSegments = data.segments || [];
    setQuestionGroups(data.question_groups || []);
    if (!currentQuestionGroups.length) {
      ocrTextInput.value = data.ocr_text || "";
      renderOcrPanel(data.ocr_text || "");
      renderStepPreview(data.steps || []);
    }
    renderOcrMeta(
      currentSegments,
      data.saved_path || "",
      data.segment_stats || null,
      Boolean(data.preprocessed),
      data.llm_debug || null,
      null
    );
    renderSegmentEditor(currentSegments);
    renderQuestionGroups(data.question_groups || []);

    const lowCount = currentSegments.filter((s) => s.low_confidence).length;
    const lowMsg = lowCount > 0 ? `，有 ${lowCount} 处建议检查` : "";
    const note = data.notes ? `（${data.notes}）` : "";
    setStatus(`识别完成${lowMsg}${note}`, "ok");
  } catch (err) {
    setStatus(err.message || "纯大模型OCR失败", "error");
  } finally {
    visionOnlyBtn.disabled = false;
  }
});

ocrForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!requireLogin()) return;
  const image = imageInput.files[0];
  const manualText = manualTextInput.value.trim();

  if (!image && !manualText) {
    setStatus("请上传图片或填写手工文本。", "error");
    return;
  }

  const fd = new FormData();
  if (image) fd.append("image", image);
  if (manualText) fd.append("extracted_text", manualText);
  fd.append("use_llm_correction", useLlmCorrectionInput.checked ? "true" : "false");
  fd.append("return_llm_debug", returnLlmDebugInput.checked ? "true" : "false");
  fd.append("use_vision_correction", useVisionCorrectionInput.checked ? "true" : "false");

  ocrBtn.disabled = true;
  setStatus("OCR 识别中...");

  try {
    const resp = await apiFetch("/api/ocr", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "OCR 请求失败");

    currentSegments = data.segments || [];
    setQuestionGroups(data.question_groups || []);
    if (!currentQuestionGroups.length) {
      ocrTextInput.value = data.ocr_text || "";
      renderOcrPanel(data.ocr_text || "");
      renderStepPreview(data.steps || []);
    }
    renderOcrMeta(
      currentSegments,
      data.saved_path || "",
      data.segment_stats || null,
      Boolean(data.preprocessed),
      data.llm_debug || null,
      data.vision_correction || null
    );
    renderSegmentEditor(currentSegments);
    renderQuestionGroups(data.question_groups || []);

    const lowCount = currentSegments.filter((s) => s.low_confidence).length;
    const lowMsg = lowCount > 0 ? `，有 ${lowCount} 处建议检查` : "";
    const llmMsg = data.corrected_by_llm ? "，已自动纠错" : "";
    const note = data.correction_note ? `（${data.correction_note}）` : "";
    setStatus(`OCR 完成${lowMsg}${llmMsg}${note}`, "ok");
  } catch (err) {
    setStatus(err.message || "OCR 失败", "error");
  } finally {
    ocrBtn.disabled = false;
  }
});

ocrTextInput.addEventListener("input", () => {
  renderOcrPanel(ocrTextInput.value);
  renderStepPreviewFromText(ocrTextInput.value);
});

window.addEventListener("load", () => {
  loadCurrentUser();
  updateImagePreview(imageInput.files[0]);
  if (ocrTextInput.value.trim()) {
    renderOcrPanel(ocrTextInput.value);
    renderStepPreviewFromText(ocrTextInput.value);
  }
  if (questionPager) questionPager.classList.add("hidden");
});

imageInput.addEventListener("change", () => {
  updateImagePreview(imageInput.files[0]);
});

gradeForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!requireLogin()) return;

  const ocrText = ocrTextInput.value.trim();
  const referenceSolution = referenceSolutionInput.value.trim();
  const useLlm = useLlmInput.checked;

  if (!ocrText) {
    setStatus("请先执行 OCR，或在 OCR 结果框中输入文本。", "error");
    return;
  }
  let textForGrade = ocrText;
  if (gradeCurrentQuestionOnlyInput?.checked && currentQuestionGroups.length) {
    textForGrade = (currentQuestionGroups[currentQuestionIndex]?.text || "").trim() || ocrText;
  }

  const fd = new FormData();
  fd.append("extracted_text", textForGrade);
  if (referenceSolution) fd.append("reference_solution", referenceSolution);
  fd.append("use_llm", useLlm ? "true" : "false");

  gradeBtn.disabled = true;

  try {
    const resp = await apiFetch("/api/grade", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "评分失败");

    totalScoreEl.textContent = `${data.total_score.toFixed(2)} / 100`;
    feedbackEl.textContent = data.feedback;
    renderRows(data.steps, data.step_scores);
    renderGradeDiag(data.grading_meta || {});
    resultSection.classList.remove("hidden");
    const llmUsed = Boolean(data.grading_meta?.llm_used);
    const modeText = llmUsed ? "大模型评分" : "规则评分";
    const err = data.grading_meta?.llm_error ? `（${sanitizeUserMessage(data.grading_meta.llm_error)}）` : "";
    setStatus(`评分完成：${modeText}${err}`, "ok");
  } catch (err) {
    setStatus(err.message || "评分失败", "error");
  } finally {
    gradeBtn.disabled = false;
  }
});
