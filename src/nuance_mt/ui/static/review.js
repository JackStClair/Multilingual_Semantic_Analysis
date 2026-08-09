let current = null;

const els = {
  lang: document.getElementById("lang"),
  reviewer: document.getElementById("reviewer"),
  queue: document.getElementById("queue"),
  card: document.getElementById("card"),
  empty: document.getElementById("empty"),
  meta: document.getElementById("meta"),
  sentence: document.getElementById("sentence"),
  fields: [
    "src_span",
    "src_char_start",
    "src_char_end",
    "nuance_type",
    "tgt_span",
    "en_frame",
    "target_frame",
    "explanation",
    "evidence",
  ],
};

els.lang.addEventListener("change", loadNext);
document.getElementById("accept").addEventListener("click", () => decide("accept"));
document.getElementById("edit").addEventListener("click", () => decide("edit"));
document.getElementById("reject").addEventListener("click", () => decide("reject"));
document.getElementById("skip").addEventListener("click", loadNext);

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
  const s = current.src_char_start;
  const e = current.src_char_end;
  const before = current.source_text.slice(0, s);
  const mid = current.source_text.slice(s, e);
  const after = current.source_text.slice(e);
  els.sentence.innerHTML = `${escapeHtml(before)}<mark class="nuance">${escapeHtml(mid)}</mark>${escapeHtml(after)}`;
  els.meta.textContent = `${current.id} · ${current.lang_pair} · ${current.annotator || "?"} · split=${current.split}`;
  for (const key of els.fields) {
    const node = document.getElementById(key);
    if (node) node.value = current[key] ?? "";
  }
}

async function decide(action) {
  if (!current) return;
  const fields = {};
  for (const key of els.fields) {
    const node = document.getElementById(key);
    let val = node.value;
    if (key === "src_char_start" || key === "src_char_end") val = Number(val);
    fields[key] = val;
  }
  const res = await fetch("/api/review/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: current.id,
      action,
      reviewer: els.reviewer.value || "human",
      fields: action === "edit" ? fields : {},
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
