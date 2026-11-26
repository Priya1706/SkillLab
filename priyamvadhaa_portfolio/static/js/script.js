const menuToggle = document.getElementById("menuToggle");
const navLinks = document.getElementById("navLinks");
if (menuToggle && navLinks) {
  menuToggle.addEventListener("click", () => {
    const isOpen = navLinks.classList.toggle("open");
    menuToggle.setAttribute("aria-expanded", String(isOpen));
    menuToggle.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
  });
  navLinks.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
    navLinks.classList.remove("open");
    menuToggle.setAttribute("aria-expanded", "false");
  }));
}
const year = document.getElementById("year");
if (year) year.textContent = new Date().getFullYear();

/* Sequencing trace: the letters of the name become bases (char code mod 4). */
const BASES = ["A", "C", "G", "T"];
const COLORS = { A: "#1f9d6a", C: "#2f6df0", G: "#1b2338", T: "#e5484d" };
const NAME = "PRIYAMVADHAAHOSKERE";
const NS = "http://www.w3.org/2000/svg";

function sequence(n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(BASES[NAME.charCodeAt(i % NAME.length) % 4]);
  return out;
}
function rand(i, k) { const x = Math.sin(i * 12.9898 + k * 78.233) * 43758.5453; return x - Math.floor(x); }

function buildTrace(el, animate) {
  const W = Math.max(300, Math.round(el.clientWidth)), H = 170, base = 140;
  const n = Math.max(8, Math.min(32, Math.floor(W / 40)));
  const seq = sequence(n), step = W / n;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H + 30}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-hidden", "true");
  BASES.forEach((b, ci) => {
    const peaks = seq.map((s, i) => ({
      x: step * (i + .5) + (rand(i, ci) - .5) * 4,
      h: s === b ? 90 + rand(i, 7) * 40 : (rand(i, ci + 3) > .55 ? 8 + rand(i, ci + 9) * 22 : 0),
      w: step * .2
    }));
    let d = "";
    for (let x = 0; x <= W; x += 3) {
      let y = 0;
      for (const p of peaks) y += p.h * Math.exp(-((x - p.x) ** 2) / (2 * p.w * p.w));
      d += (x ? "L" : "M") + x + " " + (base - y).toFixed(1);
    }
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("class", "line");
    path.setAttribute("stroke", COLORS[b]);
    path.setAttribute("pathLength", "1");
    svg.appendChild(path);
  });
  const g = document.createElementNS(NS, "g");
  g.setAttribute("class", "bases");
  seq.forEach((s, i) => {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", step * (i + .5));
    t.setAttribute("y", H + 14);
    t.setAttribute("fill", COLORS[s]);
    t.textContent = s;
    g.appendChild(t);
  });
  svg.appendChild(g);
  el.replaceChildren(svg);
  el.classList.toggle("draw", animate);
}

/* Decorative double helix for the featured project. */
function buildStrand(el) {
  const W = 520, H = 300, n = 22, svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
  const cy = H / 2, amp = 70, step = W / (n - 1);
  const pts = (ph) => Array.from({ length: n }, (_, i) => [i * step, cy + amp * Math.sin(i * .55 + ph)]);
  const a = pts(0), b = pts(Math.PI);
  const seq = sequence(n);
  for (let i = 0; i < n; i++) {
    const l = document.createElementNS(NS, "line");
    l.setAttribute("x1", a[i][0]); l.setAttribute("y1", a[i][1]);
    l.setAttribute("x2", b[i][0]); l.setAttribute("y2", b[i][1]);
    l.setAttribute("stroke", COLORS[seq[i]] === "#1b2338" ? "#8aa0d0" : COLORS[seq[i]]);
    l.setAttribute("stroke-width", "3"); l.setAttribute("stroke-linecap", "round"); l.setAttribute("opacity", ".85");
    svg.appendChild(l);
  }
  [a, b].forEach((p) => {
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", p.map((q, i) => (i ? "L" : "M") + q[0] + " " + q[1].toFixed(1)).join(""));
    path.setAttribute("fill", "none"); path.setAttribute("stroke", "#dbe3f5");
    path.setAttribute("stroke-width", "3"); path.setAttribute("stroke-linejoin", "round");
    svg.appendChild(path);
  });
  el.replaceChildren(svg);
}

const trace = document.getElementById("trace");
if (trace) {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  buildTrace(trace, !reduce);
  let w = trace.clientWidth, t;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => { if (trace.clientWidth !== w) { w = trace.clientWidth; buildTrace(trace, false); } }, 150);
  });
}
const strand = document.getElementById("strand");
if (strand) buildStrand(strand);
