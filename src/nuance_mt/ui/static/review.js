let current = null;

const fieldKeys = [
  "source_text",
  "src_span",
  "src_char_start",
  "src_char_end",
  "nuance_type",
  "tgt_span",
  "en_frame",
  "target_frame",
  "explanation",
  "evidence",
];

const els = {
  lang: document.getElementById("lang"),
  reviewer: document.getElementById("reviewer"),
  queue: document.getElementById("queue"),
  card: document.getElementById("card"),
  empty: document.getElementById("empty"),
  meta: document.getElementById("meta"),
  sentence: document.getElementById("sentence"),
  warn: document.getElementById("span-warning"),
  source: document.getElementById("source_text"),
  span: document.getElementById("src_span"),
  start: document.getElementById("src_char_start"),
  end: document.getElementById("src_char_end"),
};

els.lang.addEventListener("change", loadNext);
document.getElementById("accept").addEventListener("click", () => decide("accept"));
document.getElementById("edit").addEventListener("click", () => decide("edit"));
document.getElementById("reject").addEventListener("click", () => decide("reject"));
document.getElementById("skip").addEventListener("click", loadNext);
document.getElementById("locate").addEventListener("click", () => {
  locateSpan({ alertIfMissing: true });
  refreshPreview();
});

els.source.addEventListener("input", () => {
  locateSpan({ alertIfMissing: false });
  refreshPreview();
});
els.span.addEventListener("input", () => {
  locateSpan({ alertIfMissing: false });
  refreshPreview();
});
els.start.addEventListener("input", refreshPreview);
els.end.addEventListener("input", refreshPreview);

document.addEventListener("keydown", (ev) => {
  if (!current) return;
  const tag = (ev.target && ev.target.tagName) || "";
  if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;
  const k = ev.key.toLowerCase();
  if (k === "a") decide("accept");
  if (k === "e") decide("edit");
  if (k === "r") decide("reject");
  if (k === "s") loadNext();
});

function locateSpan({ alertIfMissing }) {
  const text = els.source.value;
  const needle = els.span.value.trim();
  if (!needle) return false;
  const idx = text.toLowerCase().indexOf(needle.toLowerCase());
  if (idx < 0) {
    els.warn.hidden = false;
    els.warn.textContent = `“${needle}” not found in the English sentence. Fix the sentence or the span.`;
    if (alertIfMissing) alert(els.warn.textContent);
    return false;
  }
  els.start.value = String(idx);
  els.end.value = String(idx + needle.length);
  // preserve original casing from the sentence
  els.span.value = text.slice(idx, idx + needle.length);
  els.warn.hidden = true;
  return true;
}

function refreshPreview() {
  const text = els.source.value;
  let s = Number(els.start.value);
  let e = Number(els.end.value);
  if (!Number.isFinite(s) || !Number.isFinite(e) || s < 0 || e > text.length || e <= s) {
    els.sentence.textContent = text || "(empty sentence)";
    els.warn.hidden = false;
    els.warn.textContent = "Span offsets are invalid for this sentence.";
    return;
  }
  const before = text.slice(0, s);
  const mid = text.slice(s, e);
  const after = text.slice(e);
  els.sentence.innerHTML = `${escapeHtml(before)}<mark class="nuance">${escapeHtml(mid)}</mark>${escapeHtml(after)}`;
  if (mid.toLowerCase() !== els.span.value.trim().toLowerCase()) {
    els.warn.hidden = false;
    els.warn.textContent = "Span text and offsets disagree — click “Find span in sentence”.";
  } else {
    els.warn.hidden = true;
  }
}

function fieldsChanged() {
  if (!current) return false;
  for (const key of fieldKeys) {
    const node = document.getElementById(key);
    let val = node.value;
    if (key === "src_char_start" || key === "src_char_end") val = Number(val);
    const orig = current[key];
    if (String(val ?? "") !== String(orig ?? "")) return true;
  }
  return false;
}

function collectFields() {
  const fields = {};
  for (const key of fieldKeys) {
    const node = document.getElementById(key);
    let val = node.value;
    if (key === "src_char_start" || key === "src_char_end") val = Number(val);
    fields[key] = val;
  }
  // keep span text synced to offsets when possible
  const text = fields.source_text || "";
  const s = fields.src_char_start;
  const e = fields.src_char_end;
  if (Number.isFinite(s) && Number.isFinite(e) && s >= 0 && e <= text.length && e > s) {
    fields.src_span = text.slice(s, e);
  }
  return fields;
}

async function loadNext() {
  const lang = els.lang.value;
  const res = await fetch(`/api/review/next?lang_pair=${encodeURIComponent(lang)}`);
  const data = await res.json();
  els.queue.textContent = `${data.remaining} medium row(s) left in queue`;
  if (!data.item) {
    els.card.hidden = true;
    els.empty.hidden = false;
    current = null;
    return;
  }
  els.empty.hidden = true;
  els.card.hidden = false;
  current = data.item;
  els.meta.textContent = `${current.id} · ${current.lang_pair} · ${current.annotator || "?"} · split=${current.split}`;
  for (const key of fieldKeys) {
    const node = document.getElementById(key);
    if (node) node.value = current[key] ?? "";
  }
  locateSpan({ alertIfMissing: false });
  refreshPreview();
}

async function decide(action) {
  if (!current) return;
  // If user edited anything but hit Accept, promote to edit automatically
  if (action === "accept" && fieldsChanged()) action = "edit";
  if (action === "edit" || action === "accept") {
    const ok = locateSpan({ alertIfMissing: action === "edit" });
    if (!ok && action === "edit") return;
    refreshPreview();
  }
  const fields = action === "edit" ? collectFields() : {};
  const res = await fetch("/api/review/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: current.id,
      action,
      reviewer: els.reviewer.value || "human",
      fields,
    }),
  });
  if (!res.ok) {
    alert(await res.text());
    return;
  }
  await loadNext();
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

loadNext();
