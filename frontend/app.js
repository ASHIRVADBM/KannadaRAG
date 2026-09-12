/* ───────────────────────────────────────────────────────────────────────────
   Karnataka Heritage QA — client

   Three behaviours here are deliberate rather than decorative:

   1. Every answer renders its sources with document, page, similarity score
      and an OCR flag. A heritage answer that cannot be traced to a page is not
      verifiable, and a passage recovered by OCR carries recognition risk the
      reader should be able to see.

   2. Abstention is rendered as a distinct, calm state — not an error. The
      system declining to answer when the corpus lacks the information is
      correct behaviour and the interface should not make it look like a fault.

   3. The Kannada-ratio meter previews the server's language guard before the
      request is sent, so a user typing in English learns why the query will be
      refused rather than being refused without explanation.
   ─────────────────────────────────────────────────────────────────────────── */

(() => {
  "use strict";

  const API = (location.protocol === "file:" ? "http://127.0.0.1:8000" : "") + "/api";
  const KANNADA = /[ಀ-೿]/g;
  const LANG_THRESHOLD = 0.5;   // must match RetrievalConfig / text.is_predominantly_kannada
  const HISTORY_KEY = "khqa.history.v2";
  const THEME_KEY = "khqa.theme";

  const $ = (id) => document.getElementById(id);

  const el = {
    thread: $("thread"), welcome: $("welcome"), query: $("query"),
    send: $("send"), history: $("history"), status: $("status"),
    meta: $("systemMeta"), modelBadge: $("modelBadge"),
    sidebar: $("sidebar"), scrim: $("scrim"),
    langMeter: $("langMeter"), langFill: $("langFill"), langText: $("langText"),
    keyboard: $("keyboard"), tpl: $("tpl-message"),
  };

  let busy = false;
  let history = load(HISTORY_KEY, []);

  /* ─────────────────────────────── storage ─────────────────────────────── */

  function load(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch { return fallback; }
  }

  function save(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ }
  }

  /* ──────────────────────────────── theme ──────────────────────────────── */

  const savedTheme = load(THEME_KEY, null);
  if (savedTheme) {
    document.documentElement.dataset.theme = savedTheme;
  } else if (window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    document.documentElement.dataset.theme = "dark";
  }

  $("themeToggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    save(THEME_KEY, next);
  });

  /* ──────────────────────────────── health ─────────────────────────────── */

  async function checkHealth() {
    const dot = el.status.querySelector(".status__dot");
    const text = el.status.querySelector(".status__text");
    try {
      const response = await fetch(`${API}/health`);
      if (!response.ok) throw new Error(response.statusText);
      const data = await response.json();

      dot.dataset.state = "ok";
      text.textContent = "ಸಿದ್ಧವಾಗಿದೆ";
      el.modelBadge.textContent = data.model || "—";
      el.meta.innerHTML = `
        <dt>chunks</dt><dd>${data.n_chunks.toLocaleString("en-IN")}</dd>
        <dt>encoder</dt><dd title="${esc(data.embedding_model)}">${esc(shortName(data.embedding_model))}</dd>
        <dt>index</dt><dd>${esc(data.index_fingerprint || "—")}</dd>`;
    } catch {
      dot.dataset.state = "error";
      text.textContent = "ಸರ್ವರ್ ಲಭ್ಯವಿಲ್ಲ";
      el.meta.innerHTML = `<dt>hint</dt><dd>start the API server</dd>`;
    }
  }

  const shortName = (s) => (s || "").split("/").pop() || "—";

  /* ────────────────────────────── rendering ────────────────────────────── */

  function esc(value) {
    const div = document.createElement("div");
    div.textContent = value ?? "";
    return div.innerHTML;
  }

  function addMessage(role, text, extras = {}) {
    el.welcome?.setAttribute("hidden", "");

    const node = el.tpl.content.cloneNode(true);
    const article = node.querySelector(".msg");
    article.classList.add(role === "user" ? "msg--user" : "msg--bot");
    if (extras.status && extras.status !== "ok") {
      article.classList.add(`msg--${extras.status.replace(/_/g, "-")}`);
    }

    const body = node.querySelector(".msg__text");
    body.textContent = text;
    body.lang = "kn";

    if (extras.sources?.length) {
      node.querySelector(".msg__sources").appendChild(renderSources(extras.sources));
    }

    if (extras.timing) {
      node.querySelector(".msg__foot").innerHTML = `
        <span>retrieval ${extras.timing.retrieval.toFixed(3)}s</span>
        <span>generation ${extras.timing.generation.toFixed(3)}s</span>
        <span>total ${extras.timing.total.toFixed(3)}s</span>
        ${extras.model ? `<span>${esc(extras.model)}</span>` : ""}`;
    }

    el.thread.appendChild(node);
    el.thread.scrollTop = el.thread.scrollHeight;
    return article;
  }

  function renderSources(sources) {
    const wrap = document.createElement("div");
    wrap.className = "sources";

    const head = document.createElement("div");
    head.className = "sources__head";
    head.textContent = `ಆಧಾರಗಳು · ${sources.length} ${sources.length === 1 ? "passage" : "passages"}`;
    wrap.appendChild(head);

    sources.forEach((source, i) => {
      const details = document.createElement("details");
      details.className = "source";

      const summary = document.createElement("summary");
      summary.className = "source__head";
      summary.innerHTML = `
        <span class="source__n">${i + 1}</span>
        <span class="source__cite">${esc(source.citation)}</span>
        ${source.used_ocr ? '<span class="source__ocr" title="Recovered by OCR">OCR</span>' : ""}
        <span class="source__score" title="cosine similarity">${source.score.toFixed(3)}</span>
        <span class="source__chev">▶</span>`;

      const body = document.createElement("div");
      body.className = "source__body";
      body.lang = "kn";
      body.textContent = source.excerpt || "—";

      details.append(summary, body);
      wrap.appendChild(details);
    });

    return wrap;
  }

  /* ────────────────────────────── language ─────────────────────────────── */

  function updateLangMeter() {
    const value = el.query.value.replace(/\s/g, "");
    if (!value) { el.langMeter.hidden = true; return; }

    const ratio = (value.match(KANNADA) || []).length / value.length;
    const ok = ratio >= LANG_THRESHOLD;

    el.langMeter.hidden = false;
    el.langMeter.classList.toggle("is-ok", ok);
    el.langFill.style.width = `${Math.round(ratio * 100)}%`;
    el.langText.textContent = ok
      ? "ಕನ್ನಡ ಪ್ರಶ್ನೆ"
      : "ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ (ಕನಿಷ್ಠ ೫೦%)";
  }

  /* ──────────────────────────────── asking ─────────────────────────────── */

  async function ask(question) {
    question = (question || "").trim();
    if (!question || busy) return;

    addMessage("user", question);
    el.query.value = "";
    autoGrow();
    updateLangMeter();

    busy = true;
    el.send.classList.add("is-busy");
    el.send.disabled = true;

    try {
      const response = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      addMessage("bot", data.answer, {
        status: data.status,
        sources: data.sources,
        model: data.model,
        timing: {
          retrieval: data.retrieval_latency_s,
          generation: data.generation_latency_s,
          total: data.total_latency_s,
        },
      });

      remember(question);
    } catch (error) {
      addMessage("bot", `ಸರ್ವರ್ ಸಂಪರ್ಕದಲ್ಲಿ ದೋಷ: ${error.message}`, { status: "error" });
    } finally {
      busy = false;
      el.send.classList.remove("is-busy");
      el.send.disabled = false;
      el.query.focus();
    }
  }

  /* ────────────────────────────── history ──────────────────────────────── */

  function remember(question) {
    history = [question, ...history.filter((q) => q !== question)].slice(0, 40);
    save(HISTORY_KEY, history);
    renderHistory();
  }

  function renderHistory() {
    if (!history.length) {
      el.history.innerHTML = '<p class="history__empty" lang="kn">ಇನ್ನೂ ಯಾವುದೇ ಪ್ರಶ್ನೆಗಳಿಲ್ಲ</p>';
      return;
    }
    el.history.innerHTML = "";
    history.forEach((question) => {
      const button = document.createElement("button");
      button.className = "history__item";
      button.lang = "kn";
      button.textContent = question;
      button.title = question;
      button.addEventListener("click", () => {
        el.query.value = question;
        autoGrow();
        updateLangMeter();
        el.query.focus();
        closeSidebar();
      });
      el.history.appendChild(button);
    });
  }

  /* ──────────────────────────────── UI glue ────────────────────────────── */

  function autoGrow() {
    el.query.style.height = "auto";
    el.query.style.height = `${Math.min(el.query.scrollHeight, 190)}px`;
  }

  const openSidebar = () => { el.sidebar.classList.add("is-open"); el.scrim.hidden = false; };
  const closeSidebar = () => { el.sidebar.classList.remove("is-open"); el.scrim.hidden = true; };

  el.query.addEventListener("input", () => { autoGrow(); updateLangMeter(); });
  el.query.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask(el.query.value);
    }
  });

  el.send.addEventListener("click", () => ask(el.query.value));

  $("samples").addEventListener("click", (event) => {
    const sample = event.target.closest(".sample");
    if (sample) ask(sample.dataset.q);
  });

  $("newChat").addEventListener("click", () => {
    el.thread.querySelectorAll(".msg").forEach((n) => n.remove());
    el.welcome?.removeAttribute("hidden");
    closeSidebar();
    el.query.focus();
  });

  $("clearHistory").addEventListener("click", () => {
    history = [];
    save(HISTORY_KEY, history);
    renderHistory();
  });

  $("openSidebar").addEventListener("click", openSidebar);
  $("closeSidebar").addEventListener("click", closeSidebar);
  el.scrim.addEventListener("click", closeSidebar);

  $("toggleKeyboard").addEventListener("click", (event) => {
    const button = event.currentTarget;
    const open = button.getAttribute("aria-pressed") === "true";
    button.setAttribute("aria-pressed", String(!open));
    el.keyboard.hidden = open;
    if (!open) window.KannadaKeyboard?.mount(el.keyboard, el.query);
  });

  /* ───────────────────────────────── boot ──────────────────────────────── */

  renderHistory();
  checkHealth();
  setInterval(checkHealth, 30_000);
  el.query.focus();
})();
