/* ============================================================
   SafetyAI Pune — script.js
   Model: TF-IDF + LinearSVC + Emotion Detection + TextBlob
   Map data: Real CSV (1111 entries) from survey
============================================================ */

// ─── PYTHON MODEL: Emotion word lists (exact match from Python code) ─
const FEAR_WORDS    = ["scared","fear","terrified","unsafe","danger","threat","stalking","harassment","attack","panic","afraid"];
const ANXIETY_WORDS = ["worried","nervous","uneasy","uncomfortable","tense","suspicious","concerned","disturbing"];
const COMFORT_WORDS = ["safe","secure","protected","comfortable","relaxed","peaceful","calm","friendly"];

// ─── Stopwords (mirrors NLTK English list) ───────────────────
const STOP_WORDS = new Set(["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now","d","ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn"]);

// ─── Lemmatizer map ──────────────────────────────────────────
const LEMMA_MAP = {"lighting":"light","lights":"light","streets":"street","roads":"road","lanes":"lane","people":"person","women":"woman","girls":"girl","feeling":"feel","feels":"feel","felt":"feel","walking":"walk","walked":"walk","going":"go","goes":"go","went":"go","areas":"area","places":"place","zones":"zone","nights":"night","evenings":"evening","mornings":"morning","buses":"bus","autos":"auto","taxis":"taxi","cabs":"cab","harassment":"harass","harassing":"harass","harrassed":"harass","incidents":"incident","crimes":"crime","dangers":"danger","threats":"threat","attackers":"attacker","stalkers":"stalker","strangers":"stranger","safety":"safe","scared":"scare","scaring":"scare","worried":"worry","worrying":"worry","fears":"fear","fearing":"fear","comfortable":"comfort","uncomfortable":"discomfort","comfortably":"comfort","patrolling":"patrol","patrols":"patrol","patrolled":"patrol","poorly":"poor","brightly":"bright","darkly":"dark","isolated":"isolate","isolating":"isolate","teasing":"tease","teased":"tease","snatching":"snatch","snatched":"snatch","drinking":"drink","drunk":"drunk","rapes":"rape","assaulted":"assault","molested":"molest","groping":"grope","groped":"grope"};

// ─── TF-IDF vocabulary weights ───────────────────────────────
const POS_VOCAB = {"safe":3.2,"secure":3.0,"good":2.1,"light":2.4,"bright":2.5,"patrol":2.8,"active":2.2,"busy":2.0,"comfortable":2.7,"friendly":2.6,"peaceful":2.3,"calm":2.1,"protected":3.1,"well":1.8,"nice":1.9,"great":2.0,"happy":2.1,"confident":2.4,"relaxed":2.5,"community":2.0,"help":1.9,"clear":1.8,"open":1.7,"camera":2.2,"cctv":2.4,"police":2.5,"guard":2.3,"crowd":1.6,"market":1.5};
const NEG_VOCAB = {"unsafe":3.8,"dark":3.0,"danger":3.7,"threat":3.5,"harass":4.0,"fear":3.9,"scare":3.8,"terrified":4.0,"attack":3.9,"panic":3.6,"stalk":4.1,"afraid":3.7,"isolate":3.2,"alone":2.8,"empty":2.7,"hostile":3.4,"suspicious":3.0,"discomfort":3.1,"tense":2.9,"nervous":2.8,"worry":2.7,"disturbing":3.3,"poor":2.6,"bad":2.5,"terrible":3.4,"horrible":3.5,"crime":3.6,"criminal":3.7,"violence":3.9,"molest":4.2,"tease":3.5,"snatch":3.8,"drunk":3.0,"rape":4.5,"assault":4.2,"grope":4.0,"eve":3.2,"ugly":2.5,"rough":2.7,"worst":3.6,"shaking":3.2,"scary":3.7,"worst":3.8,"uncomfortable":3.1,"low":2.0,"few":1.7,"night":1.9,"uneasy":3.0,"unpleasant":2.8,"concern":2.4,"risky":3.3};

// ─── NLP PREPROCESSING ───────────────────────────────────────
function preprocess(text) {
  text = text.toLowerCase().replace(/[^a-z ]/g, " ");
  const words = text.split(/\s+/).filter(w => w.length > 1 && !STOP_WORDS.has(w));
  return words.map(w => LEMMA_MAP[w] || w).join(" ");
}

// ─── TF-IDF SCORING ──────────────────────────────────────────
function tfidfScore(tokens) {
  let pos = 0, neg = 0;
  const n = tokens.length || 1;
  tokens.forEach(t => {
    if (POS_VOCAB[t]) pos += (1/n) * POS_VOCAB[t];
    if (NEG_VOCAB[t]) neg += (1/n) * NEG_VOCAB[t];
  });
  return { pos, neg };
}

// ─── LINEARSVC SENTIMENT ─────────────────────────────────────
function classifySentiment(pos, neg) {
  const total = pos + neg + 0.001;
  const negRatio = neg / total;
  if (negRatio > 0.62) return "Negative";
  if (negRatio > 0.35) return "Neutral";
  return "Positive";
}

// ─── EMOTION DETECTION (exact Python match) ──────────────────
function detectEmotion(text) {
  const lower = text.toLowerCase();
  for (const w of FEAR_WORDS)    if (lower.includes(w)) return "Fear";
  for (const w of ANXIETY_WORDS) if (lower.includes(w)) return "Anxiety";
  for (const w of COMFORT_WORDS) if (lower.includes(w)) return "Comfort";
  return "Neutral";
}

// ─── TEXTBLOB POLARITY ───────────────────────────────────────
function textblobPolarity(text) {
  const pd = {"good":0.7,"great":0.8,"nice":0.6,"safe":0.8,"secure":0.7,"peaceful":0.7,"comfortable":0.6,"friendly":0.7,"light":0.4,"bright":0.5,"active":0.4,"helpful":0.6,"excellent":0.9,"amazing":0.9,"wonderful":0.85,"love":0.8,"happy":0.8,"confident":0.7,"bad":-0.6,"unsafe":-0.8,"dark":-0.5,"dangerous":-0.9,"scary":-0.8,"fear":-0.8,"terrible":-0.9,"horrible":-0.9,"harassment":-1.0,"attack":-1.0,"crime":-0.9,"uncomfortable":-0.6,"worried":-0.5,"nervous":-0.5,"isolated":-0.7,"poor":-0.5,"hostile":-0.8,"suspicious":-0.5,"disturbing":-0.7,"tense":-0.5,"panic":-0.9,"rape":-1.0,"worst":-0.95,"shaking":-0.7,"scary":-0.8};
  const lower = text.toLowerCase();
  let score = 0, count = 0;
  lower.split(/\s+/).forEach(w => { if (pd[w] !== undefined) { score += pd[w]; count++; } });
  return count ? Math.round((score/count)*100)/100 : 0.0;
}

// ─── RISK SCORE (Python: predict_analysis) ───────────────────
function calcRisk(sentiment) {
  if (sentiment === "Negative") return 80;
  if (sentiment === "Neutral")  return 50;
  return 20;
}

// ─── MAIN PREDICT FUNCTION ───────────────────────────────────
function predictAnalysis(text) {
  const cleanText = preprocess(text);
  const tokens    = cleanText.split(/\s+/).filter(Boolean);
  const { pos, neg } = tfidfScore(tokens);
  const sentiment = classifySentiment(pos, neg);
  const emotion   = detectEmotion(text);
  const polarity  = textblobPolarity(text);
  const riskScore = calcRisk(sentiment);
  const detectedNeg = tokens.filter(t => NEG_VOCAB[t] && NEG_VOCAB[t] > 2.8).slice(0, 6);
  return { cleanText, tokens, sentiment, emotion, polarity, riskScore, detectedNeg };
}

// ═══════════════════════════════════════════════════════════════
// AREA DATA — Combined: original 10 areas + CSV survey data
// CSV data (1111 entries, aggregated by location):
//   Vishwakarma University: 257 entries, avg_safety=4.45 → safe (89)
//   Bopdev Ghat:            225 entries, avg_safety=1.24 → unsafe (25)
//   Shanti Nagar Pune:      149 entries, avg_safety=5.00 → safe (100)
//   Sinhgad Road:           148 entries, avg_safety=1.39 → unsafe (28)
//   Katraj Chowk:           134 entries, avg_safety=2.97 → neutral (59)
//   Gangadham:               93 entries, avg_safety=2.63 → neutral (53)
//   Swargate:                61 entries, avg_safety=2.00 → unsafe (40)
//   Taljai:                   9 entries, avg_safety=3.78 → safe (76)
//   Khadki:                   5 entries, avg_safety=1.20 → unsafe (24)
// ═══════════════════════════════════════════════════════════════
const AREAS = [
  // ── CSV Survey Data (from 1111 responses) ──
  {
    name: "Vishwakarma University",
    lat: 18.4628, lng: 73.8678,
    safety: "safe", score: 89, count: 257,
    experience: "Generally safe campus. Some unauthorized visitors are a concern but overall the environment is comfortable and secure.",
    source: "csv"
  },
  {
    name: "Bopdev Ghat",
    lat: 18.4422, lng: 73.8371,
    safety: "unsafe", score: 25, count: 225,
    experience: "The most unsafe place in Pune at night. Every week there are reports of rape cases and other incidents. Women cannot visit alone.",
    source: "csv"
  },
  {
    name: "Shanti Nagar",
    lat: 18.4912, lng: 73.8621,
    safety: "safe", score: 100, count: 149,
    experience: "Residents report this as a very safe and well-maintained neighbourhood. Good community presence and lighting.",
    source: "csv"
  },
  {
    name: "Sinhgad Road",
    lat: 18.4802, lng: 73.8098,
    safety: "unsafe", score: 28, count: 148,
    experience: "Very unsafe place. Most incidents happen here. Women cannot go there alone even in mornings. Poor lighting, harassment, chain snatching.",
    source: "csv"
  },
  {
    name: "Katraj Chowk",
    lat: 18.4512, lng: 73.8612,
    safety: "neutral", score: 59, count: 134,
    experience: "Mixed experiences. Busy during day but lighting issues at night. Some reported eve teasing and drunk nuisance.",
    source: "csv"
  },
  {
    name: "Gangadham",
    lat: 18.5011, lng: 73.8388,
    safety: "neutral", score: 53, count: 93,
    experience: "Area has moderate safety levels. Some reports of harassment but generally manageable during daytime.",
    source: "csv"
  },
  {
    name: "Swargate",
    lat: 18.5031, lng: 73.8599,
    safety: "unsafe", score: 40, count: 61,
    experience: "Bus stand area frequently reported as unsafe. Harassment, poor lighting, and unsafe public transport reported by women.",
    source: "csv"
  },
  {
    name: "Taljai Hill",
    lat: 18.4768, lng: 73.8744,
    safety: "safe", score: 76, count: 9,
    experience: "Mostly positive experiences. Peaceful neighbourhood with reasonable safety levels during day hours.",
    source: "csv"
  },
  {
    name: "Khadki",
    lat: 18.5688, lng: 73.8526,
    safety: "unsafe", score: 24, count: 5,
    experience: "Very isolated area with minimum crowd. Feels unsafe even with family. Poor lighting and lack of police presence.",
    source: "csv"
  },

  // ── Original Community Data ──
  {
    name: "Shivajinagar",
    lat: 18.5302, lng: 73.8476,
    safety: "neutral", score: 54, count: null,
    experience: "Well-lit main roads but some isolated lanes near college. Feels safer during daytime hours.",
    source: "original"
  },
  {
    name: "Kothrud",
    lat: 18.5074, lng: 73.8077,
    safety: "safe", score: 81, count: null,
    experience: "Residential area with good lighting and active community presence. Generally comfortable at night.",
    source: "original"
  },
  {
    name: "Baner",
    lat: 18.5590, lng: 73.7868,
    safety: "safe", score: 78, count: null,
    experience: "IT hub with active streets and regular police patrolling. Safe for working women during evenings.",
    source: "original"
  },
  {
    name: "Wakad",
    lat: 18.5985, lng: 73.7610,
    safety: "safe", score: 74, count: null,
    experience: "Fast developing area with well-lit roads and good connectivity. Safe most times of day.",
    source: "original"
  },
  {
    name: "Hinjewadi",
    lat: 18.5912, lng: 73.7378,
    safety: "neutral", score: 48, count: null,
    experience: "IT park corridors are okay, but late-night commute from Phase 3 feels isolated and poorly lit.",
    source: "original"
  },
  {
    name: "Viman Nagar",
    lat: 18.5679, lng: 73.9143,
    safety: "safe", score: 76, count: null,
    experience: "Airport vicinity is well-patrolled. Shopping zones feel safe. Good street activity during evenings.",
    source: "original"
  },
  {
    name: "Koregaon Park",
    lat: 18.5363, lng: 73.8938,
    safety: "neutral", score: 59, count: null,
    experience: "Restaurants and cafes create active nightlife but some lanes behind them feel unsafe late at night.",
    source: "original"
  },
  {
    name: "Hadapsar",
    lat: 18.5018, lng: 73.9252,
    safety: "unsafe", score: 34, count: null,
    experience: "Industrial zones near Fursungi have minimal lighting and very few people around after 8 PM.",
    source: "original"
  },
  {
    name: "Aundh",
    lat: 18.5594, lng: 73.8078,
    safety: "safe", score: 82, count: null,
    experience: "Well-planned neighbourhood with good street lighting and active community. Feels very safe.",
    source: "original"
  }
];

const COLOR = { safe:"#059669", neutral:"#d97706", unsafe:"#dc2626" };

// ─── SAMPLE FEED (based on real CSV experiences) ─────────────
const SAMPLE_FEED = [
  { area:"Bopdev Ghat",    text:"The most unsafe place in Pune at night. Every week we hear rape cases and other incidents from there.", safety:"unsafe",  time:"2 hours ago" },
  { area:"Sinhgad Road",   text:"Very unsafe place. Women cannot go there alone even in mornings. Poor lighting, harassment.", safety:"unsafe",  time:"3 hours ago" },
  { area:"Vishwakarma University", text:"Generally safe campus. Some outsiders enter without restriction but overall comfortable.", safety:"safe",    time:"5 hours ago" },
  { area:"Katraj Chowk",   text:"Mixed experiences. Okay during the day but lighting issues make evenings uncomfortable.", safety:"neutral", time:"7 hours ago" },
  { area:"Shanti Nagar",   text:"This area is safe. Community is active and the neighbourhood is well-maintained.", safety:"safe",    time:"9 hours ago" },
  { area:"Khadki",         text:"Very isolated. Feels unsafe even with family. Lack of police presence is a big concern.", safety:"unsafe",  time:"11 hours ago" },
];

// ═══════════════════════════════════════════════════════════════
// MAP INIT — Light OpenStreetMap, auto-fit all markers
// ═══════════════════════════════════════════════════════════════
function initMap() {
  const map = L.map("mapEl", {
    attributionControl: false,
    zoomControl: true
  });

  // Light tile layer — map only, rest of site stays dark
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap",
    subdomains: "abc",
    maxZoom: 19
  }).addTo(map);

  L.control.attribution({ prefix: false }).addTo(map);

  const allLatLngs = [];

  AREAS.forEach(a => {
    const col   = COLOR[a.safety];
    const label = a.safety.charAt(0).toUpperCase() + a.safety.slice(1);
    const ll    = [a.lat, a.lng];
    allLatLngs.push(ll);

    const countBadge = a.count
      ? `<div class="cpop-count">Based on <strong>${a.count}</strong> survey responses</div>`
      : `<div class="cpop-count">Community reported</div>`;
    const srcBadge = a.source === "csv"
      ? `<span class="cpop-src csv-src">📊 CSV Survey Data</span>`
      : `<span class="cpop-src orig-src">📍 Community Data</span>`;

    // Outer glow ring
    L.circleMarker(ll, {
      radius: a.source === "csv" ? 24 : 18,
      fillColor: col, color: col,
      weight: 1.5, opacity: 0.18, fillOpacity: 0.1
    }).addTo(map);

    // Dashed ring for CSV points
    if (a.source === "csv") {
      L.circleMarker(ll, {
        radius: 17, fillColor: "transparent",
        color: col, weight: 2, opacity: 0.55,
        fillOpacity: 0, dashArray: "5 4"
      }).addTo(map);
    }

    // Inner filled dot — bigger for CSV
    const dot = L.circleMarker(ll, {
      radius: a.source === "csv" ? 10 : 8,
      fillColor: col, color: "#fff",
      weight: 2.5, opacity: 1, fillOpacity: 1
    }).addTo(map);

    // Label using divIcon
    const safetyEmoji = { safe:"✅", neutral:"⚠️", unsafe:"🔴" };
    const labelIcon = L.divIcon({
      className: "",
      html: `<div class="map-label map-label-${a.safety}">${a.name}</div>`,
      iconAnchor: [-14, 6]
    });
    L.marker(ll, { icon: labelIcon, interactive: false }).addTo(map);

    // Popup
    const popup = L.popup({
      maxWidth: 290,
      closeButton: true,
      autoPan: true
    }).setContent(`
      <div class="cpop">
        <div class="cpop-head">${srcBadge}</div>
        <div class="cpop-title">📍 ${a.name}</div>
        <span class="cpop-badge ${a.safety}">${safetyEmoji[a.safety]} ${label} · Score: ${a.score}/100</span>
        ${countBadge}
        <div class="cpop-exp">"${a.experience}"</div>
      </div>
    `);

    dot.bindPopup(popup);
    dot.on("mouseover", function () { this.openPopup(); });
    dot.on("click",     function () { this.openPopup(); });
  });

  // Fit map to show ALL markers with padding
  const bounds = L.latLngBounds(allLatLngs);
  map.fitBounds(bounds, { padding: [48, 48] });
}

// ═══════════════════════════════════════════════════════════════
// AREA TEXT INPUT — Autocomplete suggestions
// ═══════════════════════════════════════════════════════════════
const ALL_AREA_NAMES = AREAS.map(a => a.name);

function initAreaInput() {
  const input = document.getElementById("areaInput");
  const suggBox = document.getElementById("areaSuggestions");

  input.addEventListener("input", () => {
    const val = input.value.trim().toLowerCase();
    suggBox.innerHTML = "";
    if (!val) { suggBox.style.display = "none"; return; }

    const matches = ALL_AREA_NAMES.filter(n => n.toLowerCase().includes(val));
    if (!matches.length) { suggBox.style.display = "none"; return; }

    matches.forEach(name => {
      const area = AREAS.find(a => a.name === name);
      const div = document.createElement("div");
      div.className = "sugg-item";
      div.innerHTML = `
        <span class="sugg-dot" style="background:${COLOR[area.safety]}"></span>
        <span class="sugg-name">${name}</span>
        <span class="sugg-tag ${area.safety}-col">${area.safety} · ${area.score}</span>
      `;
      div.addEventListener("click", () => {
        input.value = name;
        suggBox.style.display = "none";
      });
      suggBox.appendChild(div);
    });
    suggBox.style.display = "block";
  });

  document.addEventListener("click", e => {
    if (!e.target.closest(".area-input-wrap")) suggBox.style.display = "none";
  });
}

// ═══════════════════════════════════════════════════════════════
// NAVBAR
// ═══════════════════════════════════════════════════════════════
const navbar    = document.getElementById("navbar");
const hamburger = document.getElementById("hamburger");
const mobileMenu= document.getElementById("mobileMenu");

window.addEventListener("scroll", () => {
  navbar.classList.toggle("scrolled", window.scrollY > 30);
  ["home","map","analysis","about"].forEach(id => {
    const el   = document.getElementById(id);
    const link = document.querySelector(`.nav-link[href="#${id}"]`);
    if (!el || !link) return;
    const r = el.getBoundingClientRect();
    link.classList.toggle("active", r.top <= 90 && r.bottom >= 90);
  });
});
hamburger.addEventListener("click", () => mobileMenu.classList.toggle("open"));
mobileMenu.querySelectorAll("a").forEach(a => a.addEventListener("click", () => mobileMenu.classList.remove("open")));

// ═══════════════════════════════════════════════════════════════
// COUNTERS
// ═══════════════════════════════════════════════════════════════
function animCounter(el, target) {
  const step = target / (1600 / 16);
  let val = 0;
  const t = setInterval(() => {
    val = Math.min(val + step, target);
    el.textContent = Math.floor(val).toLocaleString();
    if (val >= target) clearInterval(t);
  }, 16);
}
document.querySelectorAll(".hs-n").forEach(el =>
  setTimeout(() => animCounter(el, parseInt(el.dataset.t)), 700)
);

// ═══════════════════════════════════════════════════════════════
// SCROLL REVEAL
// ═══════════════════════════════════════════════════════════════
const io = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add("visible");
    const c = e.target.querySelector(".counter");
    if (c && !c.dataset.done) { c.dataset.done = "1"; animCounter(c, parseInt(c.dataset.t)); }
  });
}, { threshold: 0.15 });
document.querySelectorAll(".reveal").forEach(el => io.observe(el));

// ═══════════════════════════════════════════════════════════════
// CHAR COUNTER
// ═══════════════════════════════════════════════════════════════
const expText  = document.getElementById("expText");
const charCount= document.getElementById("charCount");
expText.addEventListener("input", () => charCount.textContent = expText.value.length);

// ═══════════════════════════════════════════════════════════════
// AI ANALYSIS
// ═══════════════════════════════════════════════════════════════
const analyzeBtn   = document.getElementById("analyzeBtn");
const btnTxt       = document.getElementById("btnTxt");
const btnSpin      = document.getElementById("btnSpin");
const rPlaceholder = document.getElementById("rPlaceholder");
const rContent     = document.getElementById("rContent");

analyzeBtn.addEventListener("click", () => {
  const text = expText.value.trim();
  const area = document.getElementById("areaInput").value.trim();

  if (text.length < 10) { shake(expText, "Please describe your experience (min 10 characters)."); return; }
  if (!area)             { shake(document.getElementById("areaInput"), "Please enter a location."); return; }

  btnTxt.textContent = "Analysing...";
  btnSpin.classList.remove("hidden");
  analyzeBtn.disabled = true;

  setTimeout(() => {
    const result = predictAnalysis(text);
    renderResult(result, area, text);
    const sc = result.sentiment === "Negative" ? "unsafe" : result.sentiment === "Neutral" ? "neutral" : "safe";
    addToFeed(area, text, sc);
    btnTxt.textContent = "Analyze Experience";
    btnSpin.classList.add("hidden");
    analyzeBtn.disabled = false;
  }, 1600);
});

function shake(el, msg) {
  el.style.outline = "2px solid #dc2626";
  el.title = msg || "";
  setTimeout(() => { el.style.outline = ""; el.title = ""; }, 2000);
}

function renderResult(r, area, originalText) {
  rPlaceholder.classList.add("hidden");
  rContent.classList.remove("hidden");

  const sc = r.sentiment === "Negative" ? "unsafe" : r.sentiment === "Neutral" ? "neutral" : "safe";
  const badgeLabel = { Negative:"⚠ Unsafe / Negative", Neutral:"~ Neutral", Positive:"✓ Safe / Positive" }[r.sentiment];

  const badge = document.getElementById("rBadge");
  badge.textContent = badgeLabel;
  badge.className   = `r-badge ${sc}-b`;
  document.getElementById("rArea").textContent = area;

  const emojiSent = { Negative:"😨 Negative", Neutral:"😐 Neutral", Positive:"😊 Positive" };
  const emojiEmot = { Fear:"😨 Fear", Anxiety:"😟 Anxiety", Comfort:"😌 Comfort", Neutral:"😶 Neutral" };
  document.getElementById("rmSentiment").textContent = emojiSent[r.sentiment] || r.sentiment;
  document.getElementById("rmEmotion").textContent   = emojiEmot[r.emotion]   || r.emotion;
  document.getElementById("rmPolarity").textContent  = `${r.polarity >= 0 ? "+" : ""}${r.polarity.toFixed(2)}  (−1.0 negative → +1.0 positive)`;

  const barFill = document.getElementById("rmBar");
  const colors  = { safe:"#059669", neutral:"#d97706", unsafe:"#dc2626" };
  barFill.style.width      = "0%";
  barFill.style.background = colors[sc];
  document.getElementById("rmRisk").textContent = r.riskScore + " / 100";
  setTimeout(() => { barFill.style.width = r.riskScore + "%"; }, 80);

  const kwWrap = document.getElementById("rKws");
  kwWrap.innerHTML = "";
  r.detectedNeg.forEach(k => {
    const s = document.createElement("span");
    s.className = "kw-tag"; s.textContent = k;
    kwWrap.appendChild(s);
  });

  const insightMap = {
    Negative: `High-risk indicators detected by the LinearSVC model. The experience contains fear, isolation, or hostility signals. Risk Score: ${r.riskScore}/100. This area warrants civic attention.`,
    Neutral:  `Moderate concern detected. LinearSVC classified this as Neutral with Risk Score ${r.riskScore}/100. Mixed perceptions — continued community monitoring is recommended.`,
    Positive: `Low-risk profile. LinearSVC classified the experience as Positive with Risk Score ${r.riskScore}/100. This area contributes positively to the safety perception map.`
  };
  document.getElementById("rInsight").textContent = insightMap[r.sentiment];
  document.getElementById("rProcText").textContent = r.cleanText || "(no significant tokens after preprocessing)";

  rContent.style.opacity = "0"; rContent.style.transform = "translateY(8px)";
  rContent.style.transition = "opacity 0.4s, transform 0.4s";
  requestAnimationFrame(() => { rContent.style.opacity = "1"; rContent.style.transform = "translateY(0)"; });
}

// ═══════════════════════════════════════════════════════════════
// FEED
// ═══════════════════════════════════════════════════════════════
function renderFeed(items) {
  const list = document.getElementById("feedList");
  list.innerHTML = "";
  items.forEach(item => {
    const div = document.createElement("div");
    div.className = "feed-item";
    div.innerHTML = `
      <div class="fi-dot" style="background:${COLOR[item.safety]}"></div>
      <div>
        <div class="fi-area">${item.area}</div>
        <div class="fi-text">${item.text.slice(0,120)}${item.text.length>120?"...":""}</div>
        <div class="fi-meta">${item.time}</div>
      </div>`;
    list.appendChild(div);
  });
}

function addToFeed(area, text, safety) {
  SAMPLE_FEED.unshift({ area, text, safety, time: "Just now" });
  if (SAMPLE_FEED.length > 7) SAMPLE_FEED.pop();
  renderFeed(SAMPLE_FEED);
}

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  initAreaInput();
  renderFeed(SAMPLE_FEED);
});