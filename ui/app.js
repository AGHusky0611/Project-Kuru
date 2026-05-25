const API_BASE = window.KURU_API_BASE || "";

let equityChart;
let signalChart;

function formatNumber(value) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 }).format(value);
}

function formatPct(value) {
  return `${formatNumber(value)}%`;
}

async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}`);
  }
  return res.json();
}

function updateKpis(stats) {
  document.getElementById("kpiTotal").textContent = stats.total_trades;
  document.getElementById("kpiWinRate").textContent = formatPct(stats.win_rate);
  document.getElementById("kpiRoi").textContent = formatPct(stats.roi_pct);
  document.getElementById("kpiPnl").textContent = formatNumber(stats.total_pnl);
}

function renderTrades(trades, search) {
  const tbody = document.getElementById("tradesTable");
  tbody.innerHTML = "";
  const term = search.toLowerCase();

  trades
    .filter((trade) => {
      if (!term) return true;
      return JSON.stringify(trade).toLowerCase().includes(term);
    })
    .slice()
    .reverse()
    .forEach((trade) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${trade.exit_time || "--"}</td>
        <td>${trade.side || "--"}</td>
        <td>${formatNumber(trade.entry_price || 0)}</td>
        <td>${formatNumber(trade.exit_price || 0)}</td>
        <td>${formatNumber(trade.pnl || 0)}</td>
        <td>${formatPct(trade.pnl_pct || 0)}</td>
      `;
      tbody.appendChild(row);
    });
}

function renderEquityChart(equity) {
  const ctx = document.getElementById("equityChart");
  const labels = equity.map((point) => point.timestamp || "");
  const data = equity.map((point) => point.equity || 0);

  if (equityChart) {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = data;
    equityChart.update();
    return;
  }

  equityChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Equity",
          data,
          borderColor: "#00d19a",
          backgroundColor: "rgba(0, 209, 154, 0.2)",
          tension: 0.3,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: "#94a3b8" } },
        y: { ticks: { color: "#94a3b8" } },
      },
      plugins: {
        legend: { labels: { color: "#e5ebff" } },
      },
    },
  });
}

function renderSignalChart(signals) {
  const ctx = document.getElementById("signalChart");
  const labels = signals.map((signal) => signal.timestamp || "");
  const data = signals.map((signal) => signal.confidence || 0);

  if (signalChart) {
    signalChart.data.labels = labels;
    signalChart.data.datasets[0].data = data;
    signalChart.update();
    return;
  }

  signalChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Confidence",
          data,
          backgroundColor: "rgba(255, 184, 77, 0.6)",
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: "#94a3b8" } },
        y: { ticks: { color: "#94a3b8" } },
      },
      plugins: {
        legend: { labels: { color: "#e5ebff" } },
      },
    },
  });
}

async function refreshDashboard() {
  const [stats, trades, equity, signals] = await Promise.all([
    fetchJson("/stats"),
    fetchJson("/trades"),
    fetchJson("/equity"),
    fetchJson("/signals"),
  ]);

  updateKpis(stats);
  renderTrades(trades.trades || [], document.getElementById("searchInput").value || "");
  renderEquityChart(equity.equity || []);
  renderSignalChart(signals.signals || []);

  document.getElementById("lastUpdated").textContent = `Last updated: ${new Date().toLocaleString()}`;
}

function setupExportButtons() {
  document.getElementById("exportCsvBtn").addEventListener("click", () => {
    window.open(`${API_BASE}/trades.csv`, "_blank");
  });

  document.getElementById("exportJsonBtn").addEventListener("click", () => {
    window.open(`${API_BASE}/trades`, "_blank");
  });
}

function setupSearch() {
  document.getElementById("searchInput").addEventListener("input", async () => {
    const trades = await fetchJson("/trades");
    renderTrades(trades.trades || [], document.getElementById("searchInput").value || "");
  });
}

document.getElementById("refreshBtn").addEventListener("click", refreshDashboard);

setupExportButtons();
setupSearch();
refreshDashboard();
