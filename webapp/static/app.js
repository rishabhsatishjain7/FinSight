// FinSight dashboard — vanilla JS, no build step, no external dependencies.
// Everything renders from webapp/main.py's JSON; this file only shapes and
// displays it. Charts are hand-drawn inline SVG / styled divs rather than a
// charting library, so the dashboard works offline with zero network
// dependencies beyond the API itself.

const state = {
  companies: [],
  detailCache: new Map(), // ticker -> detail JSON
  openTicker: null,
  query: "",
  sector: "",
};

// Mirrors reporting/charts.py::HIGHER_IS_BETTER so trend coloring is
// consistent between the PDF reports and this dashboard.
const HIGHER_IS_BETTER = {
  current_ratio: true,
  quick_ratio: true,
  cash_ratio: true,
  net_margin: true,
  gross_margin: true,
  operating_margin: true,
  return_on_assets: true,
  return_on_equity: true,
  interest_coverage: true,
  free_cash_flow: true,
  fcf_margin: true,
  debt_to_equity: false,
  debt_to_assets: false,
  liabilities_to_assets: false,
  long_term_debt_to_equity: false,
  days_sales_outstanding: false,
  days_inventory_outstanding: false,
};

function isImproving(ratioName, first, last) {
  const higherIsBetter = HIGHER_IS_BETTER[ratioName] ?? true;
  const rising = last > first;
  return higherIsBetter ? rising : !rising;
}

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url} -> ${resp.status}`);
  return resp.json();
}

function fmtPct(v) {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtNum(v, digits = 3) {
  if (v === null || v === undefined) return "—";
  return v.toFixed(digits);
}

function riskLabel(band) {
  return { high: "High risk", elevated: "Elevated", low: "Low risk" }[band] || "Not scored";
}

// --- Summary strip -------------------------------------------------------

function renderSummary(summary) {
  const el = document.getElementById("summary-strip");
  el.innerHTML = `
    <span class="pill pill--high"><span class="pill__dot"></span>${summary.high} <span class="pill__count">high</span></span>
    <span class="pill pill--elevated"><span class="pill__dot"></span>${summary.elevated} <span class="pill__count">elevated</span></span>
    <span class="pill pill--low"><span class="pill__dot"></span>${summary.low} <span class="pill__count">low</span></span>
    <span class="pill">${summary.scored} / ${summary.total} <span class="pill__count"></span>scored</span>
  `;

  const sectorSelect = document.getElementById("sector-filter");
  for (const sector of summary.sectors) {
    const opt = document.createElement("option");
    opt.value = sector;
    opt.textContent = sector.charAt(0).toUpperCase() + sector.slice(1);
    sectorSelect.appendChild(opt);
  }
}

// --- Company table ---------------------------------------------------------

function matchesFilters(company) {
  const q = state.query.trim().toLowerCase();
  const matchesQuery =
    !q ||
    company.ticker.toLowerCase().includes(q) ||
    company.name.toLowerCase().includes(q) ||
    company.sector.toLowerCase().includes(q);
  const matchesSector = !state.sector || company.sector === state.sector;
  return matchesQuery && matchesSector;
}

function renderTable() {
  const tbody = document.getElementById("company-rows");
  tbody.innerHTML = "";

  const visible = state.companies.filter(matchesFilters);
  document.getElementById("empty-state").hidden = state.companies.length > 0;

  for (const c of visible) {
    const row = document.createElement("tr");
    row.className = "company-row";
    row.dataset.ticker = c.ticker;
    const probClass = c.distress_probability === null ? "prob-value is-null" : "prob-value";
    const yearClass = c.fiscal_year === null ? "fiscal-year is-null" : "fiscal-year";
    const badgeClass = c.risk_band ? `risk-badge risk-badge--${c.risk_band}` : "risk-badge risk-badge--none";

    row.innerHTML = `
      <td><span class="expand-chevron">▸</span><span class="ticker">${c.ticker}</span></td>
      <td class="company-name">${c.name}</td>
      <td class="sector-tag">${c.sector}</td>
      <td><span class="${yearClass}">${c.fiscal_year ?? "—"}</span></td>
      <td><span class="${probClass}">${fmtPct(c.distress_probability)}</span></td>
      <td><span class="${badgeClass}">${riskLabel(c.risk_band)}</span></td>
    `;
    row.addEventListener("click", () => toggleDetail(c.ticker));
    tbody.appendChild(row);
  }
}

// --- Detail panel ----------------------------------------------------------

async function toggleDetail(ticker) {
  const allRows = document.querySelectorAll("tr.company-row, tr.detail-row");
  const targetRow = document.querySelector(`tr.company-row[data-ticker="${ticker}"]`);
  const existingDetail = document.querySelector(`tr.detail-row[data-ticker="${ticker}"]`);

  // Collapse whatever's currently open (including re-clicking the same row).
  if (state.openTicker) {
    const openRow = document.querySelector(`tr.company-row[data-ticker="${state.openTicker}"]`);
    const openDetail = document.querySelector(`tr.detail-row[data-ticker="${state.openTicker}"]`);
    if (openRow) openRow.classList.remove("is-open");
    if (openDetail) openDetail.remove();
  }

  if (state.openTicker === ticker) {
    state.openTicker = null;
    return;
  }

  state.openTicker = ticker;
  targetRow.classList.add("is-open");

  let detail = state.detailCache.get(ticker);
  if (!detail) {
    try {
      detail = await fetchJSON(`/api/companies/${ticker}`);
      state.detailCache.set(ticker, detail);
    } catch (err) {
      detail = { error: true };
    }
  }

  const template = document.getElementById("detail-template");
  const clone = template.content.cloneNode(true);
  const detailRow = clone.querySelector("tr.detail-row");
  detailRow.dataset.ticker = ticker;
  detailRow.hidden = false;

  if (detail.error) {
    clone.querySelector("[data-field='narrative']").textContent =
      "Couldn't load detail for this company.";
  } else {
    renderDetail(clone, detail);
  }

  targetRow.after(clone);
}

function renderDetail(root, detail) {
  const narrativeEl = root.querySelector("[data-field='narrative']");
  narrativeEl.textContent = detail.narrative || "No narrative generated for this period yet.";

  renderDrivers(root.querySelector("[data-field='drivers']"), detail.shap_contributions || []);
  renderOutliers(root.querySelector("[data-field='outliers'] tbody"), detail.outliers || []);
  renderTrends(root.querySelector("[data-field='trends']"), detail.ratio_history || {});

  const reportEl = root.querySelector("[data-field='report']");
  if (detail.report_available) {
    reportEl.innerHTML = `<a class="report-link" href="${detail.report_url}" target="_blank" rel="noopener">Download PDF report</a>`;
  }
}

function renderDrivers(container, contributions) {
  if (!contributions.length) {
    container.innerHTML = '<p class="detail__empty">No model score available for this period.</p>';
    return;
  }
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.shap_value)), 1e-9);

  container.innerHTML = contributions
    .map((c) => {
      const pct = (Math.abs(c.shap_value) / maxAbs) * 50; // half-track max, diverges from center
      const dir = c.shap_value >= 0 ? "up" : "down";
      return `
        <div class="driver">
          <span class="driver__label" title="${c.feature}">${c.feature}</span>
          <span class="driver__track">
            <span class="driver__fill driver__fill--${dir}" style="width:${pct}%"></span>
          </span>
          <span class="driver__value">${c.shap_value >= 0 ? "+" : ""}${c.shap_value.toFixed(3)}</span>
        </div>`;
    })
    .join("");
}

function renderOutliers(tbody, outliers) {
  if (!outliers.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="detail__empty">No ratios deviate &ge;1.5&sigma; from sector norms.</td></tr>';
    return;
  }
  tbody.innerHTML = outliers
    .map((o) => {
      // The sign of a z-score alone doesn't say whether an outlier is good
      // or bad -- that depends on which direction is healthy for THIS
      // ratio. A current_ratio well ABOVE sector norm (z > 0) is a
      // strength, not a concern; a debt_to_equity well above norm (z > 0)
      // is the opposite. Route through the same HIGHER_IS_BETTER table the
      // trend-chart coloring uses, so an above-norm value on a "higher is
      // better" ratio reads as favorable, not alarming, and vice versa.
      const higherIsBetter = HIGHER_IS_BETTER[o.ratio] ?? true;
      const favorable = higherIsBetter ? o.z_score >= 0 : o.z_score < 0;
      const cls = favorable ? "z-favorable" : "z-concerning";
      return `<tr><td>${o.ratio}</td><td class="${cls}">${o.z_score >= 0 ? "+" : ""}${o.z_score.toFixed(2)}</td><td>${fmtNum(o.raw_value)}</td></tr>`;
    })
    .join("");
}

function renderTrends(container, ratioHistory) {
  const entries = Object.entries(ratioHistory);
  if (!entries.length) {
    container.innerHTML = '<p class="detail__empty">Not enough history for trend charts.</p>';
    return;
  }

  container.innerHTML = entries
    .map(([name, series]) => {
      const years = Object.keys(series).map(Number).sort((a, b) => a - b);
      const values = years.map((y) => series[y]);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min || 1;

      const w = 240;
      const h = 36;
      const pad = 3;
      const points = values
        .map((v, i) => {
          const x = (i / (values.length - 1 || 1)) * (w - pad * 2) + pad;
          const y = h - pad - ((v - min) / span) * (h - pad * 2);
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");

      const improving = isImproving(name, values[0], values[values.length - 1]);
      const color = improving ? "var(--risk-low)" : "var(--risk-high)";
      const lastPoint = points.split(" ").pop();

      return `
        <div class="trend-panel">
          <div class="trend-panel__label">${name}</div>
          <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
            <polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.6" />
            <circle cx="${lastPoint.split(",")[0]}" cy="${lastPoint.split(",")[1]}" r="2.2" fill="${color}" />
          </svg>
        </div>`;
    })
    .join("");
}

// --- Wiring ------------------------------------------------------------

async function init() {
  try {
    const [summary, companiesResp] = await Promise.all([
      fetchJSON("/api/summary"),
      fetchJSON("/api/companies"),
    ]);
    renderSummary(summary);
    state.companies = companiesResp.companies;
    renderTable();

    // Deep-link support: ?company=TICKER auto-expands that row on load,
    // so a specific company's detail can be shared/bookmarked directly.
    const params = new URLSearchParams(window.location.search);
    const deepLinkTicker = params.get("company");
    if (deepLinkTicker && state.companies.some((c) => c.ticker === deepLinkTicker.toUpperCase())) {
      toggleDetail(deepLinkTicker.toUpperCase());
    }
  } catch (err) {
    document.getElementById("empty-state").hidden = false;
    document.querySelector(".empty-state__title").textContent = "Couldn't reach the API";
    document.querySelector(".empty-state__body").textContent =
      "Make sure the backend is running: uvicorn webapp.main:app --reload";
  }

  document.getElementById("search").addEventListener("input", (e) => {
    state.query = e.target.value;
    renderTable();
  });
  document.getElementById("sector-filter").addEventListener("change", (e) => {
    state.sector = e.target.value;
    renderTable();
  });
}

init();
