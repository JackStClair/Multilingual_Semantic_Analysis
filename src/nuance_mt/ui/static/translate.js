const sourceEl = document.getElementById("source");
const targetEl = document.getElementById("target");
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");
const outEl = document.getElementById("out");
const sourceHtml = document.getElementById("source-html");
const translationHtml = document.getElementById("translation-html");
const notesEl = document.getElementById("notes");
const notesMeta = document.getElementById("notes-meta");

const EXAMPLES = {
  de: "After years abroad, he still longed for his homeland in a way that maps never captured.",
  ja: "Her dependence on her mentor was not weakness; it was a culturally legible intimacy.",
};

if (!sourceEl.value) {
  sourceEl.value = EXAMPLES[targetEl.value] || EXAMPLES.de;
}
targetEl.addEventListener("change", () => {
  if (!sourceEl.value.trim() || Object.values(EXAMPLES).includes(sourceEl.value)) {
    sourceEl.value = EXAMPLES[targetEl.value] || "";
  }
});

runBtn.addEventListener("click", async () => {
  const text = sourceEl.value.trim();
  if (!text) return;
  runBtn.disabled = true;
  statusEl.hidden = false;
  statusEl.textContent = "Running translation + nuance detection (first run may download models)…";
  outEl.hidden = true;
  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, target: targetEl.value }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    sourceHtml.innerHTML = data.source_html;
    translationHtml.innerHTML = data.translation_html || escapeHtml(data.translation);
    notesMeta.textContent = data.hits.length
      ? "Shown only when the cultural term (or English loan) appears in the translation."
      : "No translation-grounded nuance hits for this output.";
    notesEl.innerHTML = data.hits.length
      ? data.hits
          .map(
            (h, i) => `
        <article>
          <h3>[${i}] ${escapeHtml(h.tgt_span || h.text)} — ${escapeHtml(h.nuance_type)}</h3>
          <ul>
            <li><strong>English trigger:</strong> ${escapeHtml(h.text || "—")}</li>
            <li><strong>English frame:</strong> ${escapeHtml(h.en_frame || "—")}</li>
            <li><strong>Target frame:</strong> ${escapeHtml(h.target_frame || "—")}</li>
          </ul>
          <p>${escapeHtml(h.explanation || "No explanation in store.")}</p>
          <p class="meta">Evidence: ${escapeHtml(h.evidence || "—")}</p>
        </article>`
          )
          .join("")
      : "<p>No cultural terms from the detector/lexicon appeared in this translation. Try <em>homeland</em>, <em>dependence</em>, or <em>reading the room</em>.</p>";
    outEl.hidden = false;
    statusEl.textContent = `Done · ${data.hits.length} translation-grounded nuance hit(s)`;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message || err}`;
  } finally {
    runBtn.disabled = false;
  }
});

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
