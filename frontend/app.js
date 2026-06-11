"use strict";

const langSel = document.getElementById("lang");
const wordInput = document.getElementById("word");
const goBtn = document.getElementById("go");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const resultWordEl = document.getElementById("result-word");
const phonemesEl = document.getElementById("phonemes");
const audioRow = document.getElementById("audio-row");
const playBtn = document.getElementById("play");
const heatmapBlock = document.getElementById("heatmap-block");
const heatmapEl = document.getElementById("heatmap");

let languages = [];   // [{code, name, test_word_acc, speakable}]
let lastWord = null;  // word of the result currently shown (for audio)

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.hidden = true;
}

async function apiError(resp) {
  try {
    const body = await resp.json();
    if (body && body.detail) return body.detail;
  } catch (e) { /* non-JSON body */ }
  return `request failed (HTTP ${resp.status})`;
}

async function loadLanguages() {
  try {
    const resp = await fetch("/api/languages");
    if (!resp.ok) throw new Error(await apiError(resp));
    languages = await resp.json();
  } catch (e) {
    showError(`Could not load languages: ${e.message}. Is the API running?`);
    return;
  }
  langSel.innerHTML = "";
  for (const l of languages) {
    const opt = document.createElement("option");
    opt.value = l.code;
    const acc = (l.test_word_acc * 100).toFixed(1);
    opt.textContent = `${l.name} — ${acc}% model accuracy`;
    langSel.appendChild(opt);
  }
}

function renderPhonemes(phonemes) {
  phonemesEl.innerHTML = "";
  for (const p of phonemes) {
    const span = document.createElement("span");
    span.className = "phoneme";
    span.textContent = p;
    phonemesEl.appendChild(span);
  }
}

function renderHeatmap(chars, phonemes, attention) {
  if (!attention || attention.length === 0) {
    heatmapBlock.hidden = true;
    return;
  }
  const table = document.createElement("table");
  table.className = "heatmap-table";

  // Header row: input characters on the x-axis.
  const head = document.createElement("tr");
  head.appendChild(document.createElement("th")); // corner
  for (const ch of chars) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = ch;
    head.appendChild(th);
  }
  table.appendChild(head);

  // One row per output phoneme (y-axis), cell opacity = attention weight.
  attention.forEach((row, i) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    th.className = "row-label";
    th.textContent = phonemes[i] ?? "";
    tr.appendChild(th);
    const heat = getComputedStyle(document.documentElement)
      .getPropertyValue("--heat").trim() || "11, 122, 92";
    row.forEach((w, j) => {
      const td = document.createElement("td");
      td.className = "cell";
      td.style.background = `rgba(${heat}, ${Math.min(1, w).toFixed(3)})`;
      td.title = `${phonemes[i] ?? ""} ← ${chars[j] ?? ""}: ${w.toFixed(3)}`;
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });

  heatmapEl.innerHTML = "";
  heatmapEl.appendChild(table);
  heatmapBlock.hidden = false;
}

async function phonemize() {
  clearError();
  const word = wordInput.value.trim();
  const lang = langSel.value;
  if (!word) {
    showError("Type a word first.");
    return;
  }
  goBtn.disabled = true;
  try {
    const params = new URLSearchParams({ word, lang });
    const resp = await fetch(`/api/phonemize?${params}`);
    if (!resp.ok) {
      showError(await apiError(resp));
      resultEl.hidden = true;
      return;
    }
    const data = await resp.json();
    lastWord = data.word;
    resultWordEl.textContent = data.word;
    renderPhonemes(data.phonemes);
    renderHeatmap(data.chars, data.phonemes, data.attention);

    const langInfo = languages.find((l) => l.code === lang);
    audioRow.hidden = !(langInfo && langInfo.speakable);
    resultEl.hidden = false;
  } catch (e) {
    showError(`Network error: ${e.message}`);
    resultEl.hidden = true;
  } finally {
    goBtn.disabled = false;
  }
}

async function playReference() {
  if (!lastWord) return;
  playBtn.disabled = true;
  try {
    const params = new URLSearchParams({ word: lastWord, lang: langSel.value });
    const resp = await fetch(`/api/speak?${params}`);
    if (!resp.ok) {
      showError(await apiError(resp));
      return;
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.addEventListener("ended", () => URL.revokeObjectURL(url));
    await audio.play();
  } catch (e) {
    showError(`Could not play audio: ${e.message}`);
  } finally {
    playBtn.disabled = false;
  }
}

goBtn.addEventListener("click", phonemize);
wordInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") phonemize();
});
playBtn.addEventListener("click", playReference);

loadLanguages();
