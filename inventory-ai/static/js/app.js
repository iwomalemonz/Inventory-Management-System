/* Inventory simulator front end.
 * The server computes the whole simulation up front (/api/simulate);
 * this file plays it back at exactly 5 seconds per simulated day. */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---- state --------------------------------------------------------------
  let cfg = null;            // /api/bootstrap
  let products = [];
  let cap = 100;
  let dayMs = 5000;
  let stock = [];
  let sim = null;
  let running = false;
  let dayIdx = 0, evIdx = 0, dayStart = 0, raf = 0, hiddenAt = 0;
  let forecast = null;
  let totals = emptyTotals();
  let stockDirty = false;
  let forecastTimer = 0, forecastToken = 0;
  let phase = "idle";       // idle | loading | running | done
  let runToken = 0;         // bumps on reset so a late server reply is ignored

  let stockChart, outlookChart, totalChart;

  function emptyTotals() { return { sold: 0, restocked: 0, lost: 0, value: 0 }; }

  // ---- colour: red (empty) -> amber (half) -> green (full) --------------
  const RED = [214, 69, 69], AMBER = [240, 180, 41], GREEN = [46, 158, 91];
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  function stockColor(frac) {
    const f = Math.max(0, Math.min(1, frac));
    const c = f < 0.5 ? mix(RED, AMBER, f / 0.5) : mix(AMBER, GREEN, (f - 0.5) / 0.5);
    return `rgb(${c.join(",")})`;
  }

  // ---- charts -------------------------------------------------------------
  const valueLabels = {
    id: "valueLabels",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.font = '600 12px "Instrument Sans", system-ui, sans-serif';
      ctx.fillStyle = css("--ink");
      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      meta.data.forEach((bar, i) => {
        ctx.fillText(String(Math.round(chart.data.datasets[0].data[i])), bar.x, bar.y - 10);
      });
      ctx.restore();
    },
  };

  function buildCharts() {
    Chart.defaults.font.family = '"Instrument Sans", system-ui, sans-serif';
    Chart.defaults.color = css("--muted");
    const anim = reduceMotion ? false : { duration: 320 };

    stockChart = new Chart($("#stockChart"), {
      type: "bar",
      data: {
        labels: products.map((p) => [p.tier, p.category]),
        datasets: [{
          data: stock.slice(),
          backgroundColor: stock.map((s) => stockColor(s / cap)),
          borderRadius: 4,
          barPercentage: 0.78,
          categoryPercentage: 0.9,
        }],
      },
      options: {
        maintainAspectRatio: false,
        animation: anim,
        layout: { padding: { top: 6 } },
        scales: {
          y: {
            min: 0, max: cap + 14,
            grid: { color: css("--line") },
            border: { display: false },
            ticks: { stepSize: Math.round(cap / 4), callback: (v) => (v <= cap ? v : "") },
          },
          x: { grid: { display: false }, border: { display: false }, ticks: { color: css("--ink"), font: { size: 13 }, maxRotation: 0 } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: (items) => products[items[0].dataIndex].name, label: (c) => `${Math.round(c.parsed.y)} of ${cap} units` } },
        },
      },
      plugins: [valueLabels],
    });

    outlookChart = new Chart($("#outlookChart"), {
      type: "bar",
      data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderColor: css("--ink"), borderWidth: [], borderRadius: 3 }] },
      options: {
        maintainAspectRatio: false,
        animation: anim,
        scales: {
          x: { grid: { display: false }, border: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 12, font: { size: 11 } } },
          y: { beginAtZero: true, grid: { color: css("--line") }, border: { display: false }, title: { display: true, text: "units / day", font: { size: 11 } } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => {
                const d = forecast && forecast.days[items[0].dataIndex];
                return d ? `${d.weekday}, ${d.date_label}` : "";
              },
              label: (item) => {
                const d = forecast.days[item.dataIndex];
                return [`${Math.round(item.parsed.y)} units expected`, d.headline];
              },
            },
          },
        },
      },
    });

    totalChart = new Chart($("#totalChart"), {
      type: "line",
      data: { labels: ["Start"], datasets: [{
        data: [100], borderColor: css("--ink"), backgroundColor: "rgba(21,34,28,0.07)",
        fill: true, tension: 0.25, pointRadius: 2.5, pointBackgroundColor: css("--ink"), borderWidth: 2,
      }] },
      options: {
        maintainAspectRatio: false,
        animation: anim,
        scales: {
          x: { grid: { display: false }, border: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 14, font: { size: 11 } } },
          y: { min: 0, max: 100, grid: { color: css("--line") }, border: { display: false }, ticks: { callback: (v) => v + "%", stepSize: 25 } },
        },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${Math.round(c.parsed.y)}% of full stock` } } },
      },
    });
  }

  function refreshStockChart() {
    const ds = stockChart.data.datasets[0];
    ds.data = stock.slice();
    ds.backgroundColor = stock.map((s) => stockColor(s / cap));
    stockChart.update();
    stockDirty = false;
  }

  const totalPct = () => (stock.reduce((a, b) => a + b, 0) / (cap * products.length)) * 100;

  // ---- KPIs ---------------------------------------------------------------
  const fmt = (n) => Math.round(n).toLocaleString("en-US");
  function renderTotals() {
    $("#kSold").textContent = fmt(totals.sold);
    $("#kRestocked").textContent = fmt(totals.restocked);
    const lost = $("#kLost");
    lost.textContent = fmt(totals.lost);
    lost.classList.toggle("bad", totals.lost > 0);
    $("#kValue").textContent = fmt(totals.value);
    const asked = totals.sold + totals.lost;
    $("#kService").textContent = asked ? Math.round((totals.sold / asked) * 100) + "%" : "100%";
  }

  // ---- log ----------------------------------------------------------------
  const logEl = $("#log");
  function addLog(kind, text) {
    const line = document.createElement("div");
    line.className = "line " + kind;
    line.textContent = text;
    const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 30;
    logEl.appendChild(line);
    while (logEl.children.length > 300) logEl.removeChild(logEl.firstChild);
    if (atBottom) logEl.scrollTop = logEl.scrollHeight;   // follow new lines unless the user scrolled up
  }

  // ---- banner -------------------------------------------------------------
  function setBanner(main, sub, text, level) {
    $("#dateMain").textContent = main;
    $("#dayOf").textContent = sub;
    const el = $("#demand");
    el.textContent = text;
    el.className = "demand " + level;
  }

  function idleBanner() {
    if (forecast && forecast.days.length) {
      const d = forecast.days[0];
      setBanner(`${d.weekday}, ${d.date_label}`, "Choose days and press Simulate", d.headline, d.level);
    }
  }

  // ---- events -------------------------------------------------------------
  function applyEvent(e) {
    const name = products[e.p].name;
    if (e.type === "sale") {
      stock[e.p] = Math.max(0, stock[e.p] - e.qty);
      totals.sold += e.qty;
      totals.value += e.qty * products[e.p].price;
      addLog("sale", `-${e.qty} x ${name}`);
    } else if (e.type === "restock") {
      stock[e.p] = Math.min(cap, stock[e.p] + e.qty);
      totals.restocked += e.qty;
      addLog("restock", `+${e.qty} ${name}`);
    } else if (e.type === "stockout") {
      totals.lost += e.qty;
      addLog("out", `Out of stock: ${name} (${e.qty} missed)`);
    }
    stockDirty = true;
  }

  // ---- playback -----------------------------------------------------------
  function startDay(i, now) {
    dayIdx = i; evIdx = 0; dayStart = now;
    const d = sim.days[i];
    setBanner(`${d.weekday}, ${d.date_label}`, `Day ${d.day} of ${sim.days.length}`, d.headline, d.level);
    highlightOutlook(i);
    totalChart.data.labels.push(`${d.date_label.replace(/ \d{4}$/, "")}`);
    totalChart.data.datasets[0].data.push(totalPct());
    totalChart.update();
  }

  function tick(now) {
    if (!running) return;
    const day = sim.days[dayIdx];
    const frac = (now - dayStart) / dayMs;

    while (evIdx < day.events.length && day.events[evIdx].t <= frac) applyEvent(day.events[evIdx++]);
    if (stockDirty) {
      refreshStockChart();
      renderTotals();
      const data = totalChart.data.datasets[0].data;
      data[data.length - 1] = totalPct();
      totalChart.update("none");
    }

    if (frac >= 1) {
      while (evIdx < day.events.length) applyEvent(day.events[evIdx++]);
      stock = day.end_stock.slice();        // re-sync with the server's numbers
      refreshStockChart();
      renderTotals();
      const data = totalChart.data.datasets[0].data;
      data[data.length - 1] = totalPct();
      totalChart.update("none");
      if (dayIdx + 1 < sim.days.length) {
        startDay(dayIdx + 1, dayStart + dayMs);   // keep exactly 5 s per day
      } else {
        finish();
        return;
      }
    }
    raf = requestAnimationFrame(tick);
  }

  function finish() {
    running = false;
    phase = "done";
    setControls(false);
    highlightOutlook(-1);
    const t = totals;
    addLog("note", `Done: ${fmt(t.sold)} sold, ${fmt(t.restocked)} restocked, ${fmt(t.lost)} missed`);
    setBanner($("#dateMain").textContent, `Simulation complete (${sim.days.length} days)`, "Complete", "done");
  }

  async function startSim() {
    if (phase === "loading" || phase === "running") { resetAll(); return; }   // button doubles as "Stop"
    resetView();
    setControls(true);
    phase = "loading";
    const myRun = ++runToken;
    let data;
    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          days: +$("#days").value,
          smart: $("#smart").checked,
          seed: Math.floor(Math.random() * 1e9),
        }),
      });
      if (!res.ok) throw new Error("Server returned " + res.status);
      data = await res.json();
    } catch (err) {
      if (myRun !== runToken) return;
      phase = "idle";
      setControls(false);
      addLog("out", "Could not start: " + err.message);
      return;
    }
    if (myRun !== runToken) return;               // stopped while waiting
    sim = data;
    running = true;
    phase = "running";
    const now = performance.now();
    startDay(0, now);
    raf = requestAnimationFrame(tick);
  }

  function resetView() {
    cancelAnimationFrame(raf);
    running = false;
    phase = "idle";
    runToken++;
    stock = products.map(() => cap);
    totals = emptyTotals();
    logEl.innerHTML = "";
    totalChart.data.labels = ["Start"];
    totalChart.data.datasets[0].data = [100];
    totalChart.update("none");
    refreshStockChart();
    renderTotals();
    highlightOutlook(-1);
    idleBanner();
  }

  function resetAll() {
    resetView();
    setControls(false);
  }

  function setControls(isRunning) {
    const btn = $("#simulateBtn");
    btn.textContent = isRunning ? "Stop" : "Simulate";
    btn.classList.toggle("running", isRunning);
    $("#days").disabled = isRunning;
    $("#smart").disabled = isRunning;
    $("#resetBtn").disabled = false;
  }

  // keep timing honest when the tab is hidden (browsers pause animation frames)
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { hiddenAt = performance.now(); return; }
    if (running && hiddenAt) dayStart += performance.now() - hiddenAt;
    hiddenAt = 0;
  });

  // ---- forecast panels ----------------------------------------------------
  const LEVEL_COLOR = { high: "#F0B429", normal: "#C3CCC8", low: "#7FA6CC" };

  function highlightOutlook(i) {
    if (!outlookChart) return;
    const ds = outlookChart.data.datasets[0];
    ds.borderWidth = (forecast ? forecast.days : []).map((_, k) => (k === i ? 2 : 0));
    outlookChart.update("none");
  }

  function renderForecast() {
    const f = forecast;
    const ds = outlookChart.data.datasets[0];
    outlookChart.data.labels = f.days.map((d) => d.date_label.replace(/ \d{4}$/, ""));
    ds.data = f.days.map((d) => d.units);
    ds.backgroundColor = f.days.map((d) => LEVEL_COLOR[d.level]);
    ds.borderWidth = f.days.map(() => 0);
    outlookChart.update();

    $("#festivalLine").textContent = f.festivals.length
      ? "Festivals in range: " + f.festivals.map((x) => `${x.name} (${x.label})`).join(", ")
      : "No major festivals in this range.";

    const policy = $("#smart").checked ? "forecast-based" : "basic 30% threshold";
    $("#riskSub").textContent =
      `Chance each product sells out at least once in ${f.days.length} day${f.days.length > 1 ? "s" : ""} ` +
      `(${policy} restocking, ${cfg.risk_runs} simulated runs).`;

    const list = $("#riskList");
    list.innerHTML = "";
    f.risk.forEach((r) => {
      const pct = Math.round(r.risk * 100);
      const color = r.risk < 0.15 ? "var(--green)" : r.risk < 0.5 ? "var(--amber)" : "var(--red)";
      const li = document.createElement("li");
      const runway = r.runway_days ? `empty in about ${r.runway_days} days if never restocked` : "lasts 60+ days without restocking";
      li.innerHTML =
        `<span class="r-name"></span><span class="r-pct">${pct}%</span>` +
        `<span class="r-meta">${runway}</span>` +
        `<span class="r-bar"><i style="width:${Math.max(pct, 2)}%;background:${color}"></i></span>`;
      li.querySelector(".r-name").textContent = r.name;
      list.appendChild(li);
    });
    if (phase === "idle") idleBanner();
  }

  async function loadForecast() {
    const token = ++forecastToken;
    try {
      const res = await fetch(`/api/forecast?days=${$("#days").value}&smart=${$("#smart").checked ? 1 : 0}`);
      if (!res.ok) throw new Error("Server returned " + res.status);
      const data = await res.json();
      if (token !== forecastToken) return;       // a newer request replaced this one
      forecast = data;
      renderForecast();
    } catch (err) {
      $("#riskSub").textContent = "Could not load the forecast: " + err.message;
    }
  }

  function queueForecast() {
    clearTimeout(forecastTimer);
    forecastTimer = setTimeout(loadForecast, 250);
  }

  // ---- model notes --------------------------------------------------------
  function renderHow() {
    const m = cfg.metrics;
    const nb = m.naive_bayes, rf = m.random_forest;
    $("#howBody").innerHTML = `
      <p>Two models learn from ${m.train_days} days of sales and are tested on the next ${m.test_days} days they have not seen.</p>
      <dl>
        <dt>Naive Bayes</dt>
        <dd>Predicts whether a day will be a high-demand day for a product. Accuracy ${nb.accuracy} (always guessing "no" scores ${nb.majority_baseline}), AUC ${nb.auc} (0.5 means no better than chance).</dd>
        <dt>Random Forest</dt>
        <dd>Predicts units sold per day. Average error ${rf.mae} units versus ${rf.baseline_mae} for a plain average; R&sup2; ${rf.r2}.</dd>
      </dl>
      <p>${cfg.calendar_priors
        ? "This dataset shows almost no weekday, month or holiday pattern, so the models add little on their own. Their weight in the forecast depends on how much they beat the plain average on the test days. The weekend, payday, season and festival lifts come from editable assumptions in <code>inventory/calendar_features.py</code>, applied on top."
        : "Calendar assumptions are switched off, so the forecast uses the models alone."}</p>
      <p>Restocking: "forecast-based" orders when stock plus deliveries on the way would not cover the forecast for the next few days. "Basic" orders when stock drops below 30%. Deliveries take 1 to 4 days.</p>`;
  }

  // ---- start up -----------------------------------------------------------
  async function init() {
    const res = await fetch("/api/bootstrap");
    cfg = await res.json();
    products = cfg.products;
    cap = cfg.capacity;
    dayMs = cfg.seconds_per_day * 1000;
    stock = products.map(() => cap);

    const slider = $("#days");
    slider.min = cfg.min_days;
    slider.max = cfg.max_days;
    const updateLabel = () => {
      const n = +slider.value;
      $("#daysValue").textContent = n;
      $("#daysSeconds").textContent = `(${n * cfg.seconds_per_day} seconds)`;
    };
    slider.addEventListener("input", () => { updateLabel(); queueForecast(); });
    $("#smart").addEventListener("change", queueForecast);
    $("#simulateBtn").addEventListener("click", startSim);
    $("#resetBtn").addEventListener("click", resetAll);
    updateLabel();

    buildCharts();
    renderTotals();
    renderHow();
    await loadForecast();
  }

  init().catch((err) => {
    $("#dateMain").textContent = "Could not load";
    $("#dayOf").textContent = String(err);
  });
})();
