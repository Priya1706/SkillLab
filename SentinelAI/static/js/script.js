/* SentinelAI Frontend Interactivity & Plotly Visualizations */

document.addEventListener("DOMContentLoaded", () => {
  // Mobile sidebar toggle
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebar = document.getElementById("sidebar");
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  // Upload dropzone enhancement on index.html
  const dropZone = document.getElementById("csvDropZone");
  const fileInput = document.getElementById("csvFileInput");
  const fileNameLabel = document.getElementById("selectedFileName");
  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0 && fileNameLabel) {
        fileNameLabel.textContent = `Selected: ${fileInput.files[0].name}`;
      }
    });
    ["dragenter", "dragover"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
      });
    });
    dropZone.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        if (fileNameLabel) {
          fileNameLabel.textContent = `Selected: ${e.dataTransfer.files[0].name}`;
        }
      }
    });
  }

  // Dashboard Plotly Visualizations
  if (document.getElementById("chartAnomalyDist")) {
    initDashboardCharts();
  }

  // Components interactive table
  if (document.getElementById("componentsTableBody")) {
    initComponentsTable();
  }

  // Reports batch export filter updater
  const reportBatchSelect = document.getElementById("reportBatchSelect");
  if (reportBatchSelect) {
    const csvBtn = document.getElementById("exportCsvBtn");
    const pdfBtn = document.getElementById("exportPdfBtn");
    reportBatchSelect.addEventListener("change", () => {
      const batch = encodeURIComponent(reportBatchSelect.value);
      const q = batch ? `?batch_id=${batch}` : "";
      if (csvBtn) csvBtn.href = `/export/csv${q}`;
      if (pdfBtn) pdfBtn.href = `/export/pdf${q}`;
    });
  }
});

const PLOTLY_CONFIG = { responsive: true, displaylogo: false };
const BASE_LAYOUT = {
  font: { family: "Inter, Segoe UI, sans-serif", color: "#334155", size: 12 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "#ffffff",
  margin: { t: 28, r: 20, b: 46, l: 54 },
};

async function initDashboardCharts() {
  try {
    const res = await fetch("/api/dashboard-data");
    const data = await res.json();

    // 1. Component Anomaly Distribution (Donut)
    const ac = data.anomaly_counts;
    Plotly.newPlot(
      "chartAnomalyDist",
      [
        {
          type: "pie",
          hole: 0.56,
          labels: ["NORMAL", "WARNING", "ANOMALOUS"],
          values: [ac.NORMAL, ac.WARNING, ac.ANOMALOUS],
          marker: { colors: ["#059669", "#D97706", "#DC2626"] },
          textinfo: "label+percent",
          hovertemplate: "<b>%{label}</b><br>Components: %{value}<br>Share: %{percent}<extra></extra>",
        },
      ],
      {
        ...BASE_LAYOUT,
        margin: { t: 20, r: 20, b: 20, l: 20 },
        showlegend: true,
        legend: { orientation: "h", y: -0.05, x: 0.5, xanchor: "center" },
      },
      PLOTLY_CONFIG
    );

    // 2. Leakage Current Distribution
    const lbr = data.leakage_by_risk;
    const leakTraces = [
      { name: "LOW RISK", vals: lbr["LOW RISK"] || [], color: "#059669" },
      { name: "MEDIUM RISK", vals: lbr["MEDIUM RISK"] || [], color: "#D97706" },
      { name: "HIGH RISK", vals: lbr["HIGH RISK"] || [], color: "#DC2626" },
    ].map((g) => ({
      type: "histogram",
      x: g.vals,
      name: g.name,
      opacity: 0.68,
      marker: { color: g.color },
      nbinsx: 28,
    }));

    Plotly.newPlot(
      "chartLeakageDist",
      leakTraces,
      {
        ...BASE_LAYOUT,
        barmode: "overlay",
        xaxis: { title: "Leakage Current (nA)", gridcolor: "#f1f5f9" },
        yaxis: { title: "Component Count", gridcolor: "#f1f5f9" },
        shapes: [
          {
            type: "line",
            x0: 250,
            x1: 250,
            y0: 0,
            y1: 1,
            yref: "paper",
            line: { color: "#0F172A", width: 2, dash: "dash" },
          },
        ],
        annotations: [
          {
            x: 250,
            y: 0.96,
            yref: "paper",
            text: "Datasheet Limit (250 nA)",
            showarrow: false,
            xanchor: "right",
            font: { size: 11, color: "#0F172A" },
          },
        ],
        legend: { orientation: "h", y: 1.12, x: 0 },
      },
      PLOTLY_CONFIG
    );

    // 3. Component Drift Trends
    const driftPlotTraces = [];
    data.drift_traces.forEach((tr) => {
      driftPlotTraces.push({
        type: "scatter",
        mode: "lines+markers",
        x: tr.x,
        y: tr.y,
        name: `${tr.component_id} (${tr.risk_status})`,
        line: { color: tr.color, width: 1.4, dash: "dot" },
        marker: { size: 5 },
        opacity: 0.55,
        showlegend: false,
        hovertemplate: `<b>${tr.component_id}</b><br>Hour: %{x}h<br>Value: %{y:.2f}<extra></extra>`,
      });
    });
    data.mean_trajectories.forEach((mt) => {
      driftPlotTraces.push({
        type: "scatter",
        mode: "lines+markers",
        x: mt.x,
        y: mt.y,
        name: mt.name,
        line: { color: mt.color, width: 3.5 },
        marker: { size: 8 },
      });
    });

    Plotly.newPlot(
      "chartDriftTrends",
      driftPlotTraces,
      {
        ...BASE_LAYOUT,
        xaxis: {
          title: "Burn-In Duration (Hours)",
          tickvals: [0, 24, 96, 168],
          gridcolor: "#f1f5f9",
        },
        yaxis: { title: "Parametric Value", gridcolor: "#f1f5f9" },
        legend: { orientation: "h", y: 1.14, x: 0 },
      },
      PLOTLY_CONFIG
    );

    // 4. Actual vs Predicted Value_168h
    const avp = data.actual_vs_pred;
    if (avp.available) {
      const colorMap = {
        "LOW RISK": "#059669",
        "MEDIUM RISK": "#D97706",
        "HIGH RISK": "#DC2626",
      };
      const pointColors = avp.risk_status.map((s) => colorMap[s] || "#2563EB");
      Plotly.newPlot(
        "chartActualVsPred",
        [
          {
            type: "scatter",
            mode: "markers",
            x: avp.actual,
            y: avp.predicted,
            text: avp.component_ids,
            marker: { color: pointColors, size: 7, opacity: 0.78 },
            hovertemplate:
              "<b>%{text}</b><br>Actual 168h: %{x:.2f}<br>Predicted 168h: %{y:.2f}<extra></extra>",
            name: "Components",
          },
          {
            type: "scatter",
            mode: "lines",
            x: [avp.min_val, avp.max_val],
            y: [avp.min_val, avp.max_val],
            line: { color: "#0F172A", width: 2, dash: "dash" },
            name: "Ideal y = x",
          },
        ],
        {
          ...BASE_LAYOUT,
          xaxis: { title: "Actual Measured Value_168h", gridcolor: "#f1f5f9" },
          yaxis: { title: "Predicted Value_168h (RF)", gridcolor: "#f1f5f9" },
          legend: { orientation: "h", y: 1.12, x: 0 },
        },
        PLOTLY_CONFIG
      );
    } else {
      document.getElementById("chartActualVsPred").innerHTML =
        '<div class="text-muted text-center py-5">Actual Value_168h ground truth is not included in this uploaded CSV (inference-only mode).</div>';
    }

    // 5. Risk Distribution Across Batches
    const br = data.batch_risk;
    Plotly.newPlot(
      "chartBatchRisk",
      [
        {
          type: "bar",
          name: "NORMAL",
          x: br.batches,
          y: br.normal,
          marker: { color: "#059669" },
        },
        {
          type: "bar",
          name: "WARNING",
          x: br.batches,
          y: br.warning,
          marker: { color: "#D97706" },
        },
        {
          type: "bar",
          name: "ANOMALOUS",
          x: br.batches,
          y: br.anomalous,
          marker: { color: "#DC2626" },
        },
      ],
      {
        ...BASE_LAYOUT,
        barmode: "stack",
        xaxis: { title: "Manufacturing / Screening Batch", gridcolor: "#f1f5f9" },
        yaxis: { title: "Component Count", gridcolor: "#f1f5f9" },
        legend: { orientation: "h", y: 1.12, x: 0 },
      },
      PLOTLY_CONFIG
    );
  } catch (err) {
    console.error("Error rendering dashboard charts:", err);
  }
}

/* Components Interactive Table */
function initComponentsTable() {
  const state = {
    page: 1,
    per_page: 25,
    sort_by: "anomaly_score",
    sort_dir: "desc",
    search: "",
    batch_id: "",
    anomaly_status: "",
    risk_status: "",
    static_pass: "",
  };

  const tbody = document.getElementById("componentsTableBody");
  const searchInput = document.getElementById("filterSearch");
  const batchSel = document.getElementById("filterBatch");
  const anomalySel = document.getElementById("filterAnomaly");
  const riskSel = document.getElementById("filterRisk");
  const staticSel = document.getElementById("filterStatic");
  const paginationInfo = document.getElementById("paginationInfo");
  const prevBtn = document.getElementById("prevPageBtn");
  const nextBtn = document.getElementById("nextPageBtn");
  const resetFiltersBtn = document.getElementById("resetFiltersBtn");

  let debounceTimer = null;

  async function loadTable() {
    const params = new URLSearchParams(state);
    tbody.innerHTML =
      '<tr><td colspan="9" class="text-center text-muted py-4">Loading screening telemetry...</td></tr>';
    try {
      const res = await fetch(`/api/components?${params.toString()}`);
      const data = await res.json();

      if (!data.items.length) {
        tbody.innerHTML =
          '<tr><td colspan="9" class="text-center text-muted py-4">No components match the active filters.</td></tr>';
      } else {
        tbody.innerHTML = data.items
          .map((item) => {
            const riskClass =
              item.risk_status === "HIGH RISK"
                ? "pill-high"
                : item.risk_status === "MEDIUM RISK"
                ? "pill-medium"
                : "pill-low";
            const anomClass =
              item.anomaly_status === "ANOMALOUS"
                ? "pill-anomalous"
                : item.anomaly_status === "WARNING"
                ? "pill-warning"
                : "pill-normal";
            const staticBadge = item.static_pass
              ? '<span class="badge text-bg-light border text-success">PASS</span>'
              : '<span class="badge text-bg-light border text-danger">FAIL</span>';

            return `
              <tr>
                <td class="mono fw-semibold">${item.component_id}</td>
                <td class="mono">${item.batch_id}</td>
                <td class="mono">${item.leakage_current_na.toFixed(1)} nA</td>
                <td class="mono">${item.value_0h.toFixed(2)}</td>
                <td class="mono">${item.predicted_value_168h.toFixed(2)}</td>
                <td class="mono fw-semibold">${item.anomaly_score.toFixed(1)}</td>
                <td><span class="status-pill ${anomClass}">${item.anomaly_status}</span></td>
                <td><span class="status-pill ${riskClass}">${item.risk_status}</span> ${staticBadge}</td>
                <td>
                  <a href="/analysis/${encodeURIComponent(item.component_id)}" class="btn btn-sm btn-outline-primary">
                    <i class="bi bi-search me-1"></i>Details
                  </a>
                </td>
              </tr>
            `;
          })
          .join("");
      }

      paginationInfo.textContent = `Showing page ${data.page} of ${data.total_pages} (${data.total} matching components)`;
      prevBtn.disabled = data.page <= 1;
      nextBtn.disabled = data.page >= data.total_pages;
      state.page = data.page;
    } catch (err) {
      tbody.innerHTML =
        '<tr><td colspan="9" class="text-center text-danger py-4">Failed to load component records.</td></tr>';
    }
  }

  if (searchInput) {
    searchInput.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        state.search = searchInput.value.trim();
        state.page = 1;
        loadTable();
      }, 220);
    });
  }

  [
    [batchSel, "batch_id"],
    [anomalySel, "anomaly_status"],
    [riskSel, "risk_status"],
    [staticSel, "static_pass"],
  ].forEach(([el, key]) => {
    if (el) {
      el.addEventListener("change", () => {
        state[key] = el.value;
        state.page = 1;
        loadTable();
      });
    }
  });

  document.querySelectorAll(".sortable-th").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.getAttribute("data-sort");
      if (state.sort_by === col) {
        state.sort_dir = state.sort_dir === "asc" ? "desc" : "asc";
      } else {
        state.sort_by = col;
        state.sort_dir = "desc";
      }
      state.page = 1;
      loadTable();
    });
  });

  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (state.page > 1) {
        state.page -= 1;
        loadTable();
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      state.page += 1;
      loadTable();
    });
  }

  if (resetFiltersBtn) {
    resetFiltersBtn.addEventListener("click", () => {
      state.search = "";
      state.batch_id = "";
      state.anomaly_status = "";
      state.risk_status = "";
      state.static_pass = "";
      state.page = 1;
      if (searchInput) searchInput.value = "";
      if (batchSel) batchSel.value = "";
      if (anomalySel) anomalySel.value = "";
      if (riskSel) riskSel.value = "";
      if (staticSel) staticSel.value = "";
      loadTable();
    });
  }

  loadTable();
}
