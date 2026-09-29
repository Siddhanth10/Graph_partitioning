const API_URL = (window.GRAPH_API_URL || "http://localhost:5000").replace(/\/$/, "");

const $ = (id) => document.getElementById(id);
const nodeCount = $("nodeCount");
const edgesInput = $("edges");
const algorithm = $("algorithm");
const errorBox = $("errorBox");
const results = $("results");
const emptyState = $("emptyState");
const partitionBtn = $("partitionBtn");

function setStatus(online, text) {
  $("statusDot").style.background = online ? "#21d4a5" : "#7d8998";
  $("statusText").textContent = text;
}

async function checkApi() {
  try {
    const response = await fetch(API_URL + "/api/health");
    if (!response.ok) throw new Error();
    setStatus(true, "API connected");
  } catch {
    setStatus(false, "API offline");
  }
}

function parseEdges() {
  const lines = edgesInput.value.split(/\n|,/).map(x => x.trim()).filter(Boolean);
  const edges = [];
  for (const line of lines) {
    const match = line.match(/^(\d+)\s*[-:]\s*(\d+)(?:\s*:\s*(\d+(?:\.\d+)?))?$/);
    if (!match) throw new Error("Invalid edge: " + line + ". Use 0-1 or 0-1:2.");
    edges.push({source: Number(match[1]), target: Number(match[2]), weight: match[3] ? Number(match[3]) : 1});
  }
  return edges;
}

function loadSample() {
  nodeCount.value = 10;
  edgesInput.value = ["0-1","0-2","0-3","1-2","1-4","2-3","2-5","3-5","4-5","4-6","5-7","6-7","6-8","7-8","7-9","8-9","2-7","3-6"].join("\n");
  algorithm.value = "spectral";
  hideError();
}

function randomEdges() {
  const n = Math.max(2, Math.min(500, Number(nodeCount.value) || 10));
  const probability = n <= 25 ? 0.22 : 0.08;
  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) if (Math.random() < probability) edges.push(i + "-" + j);
  }
  if (!edges.length) edges.push("0-1");
  edgesInput.value = edges.join("\n");
  hideError();
}

function showError(message) { errorBox.textContent = message; errorBox.classList.remove("hidden"); }
function hideError() { errorBox.classList.add("hidden"); }

function renderNodes(containerId, nodes) {
  const container = $(containerId);
  container.innerHTML = "";
  nodes.forEach(node => {
    const chip = document.createElement("span");
    chip.className = "node-chip";
    chip.textContent = node;
    container.appendChild(chip);
  });
}

function drawGraph(data) {
  const svg = $("graphSvg");
  svg.innerHTML = "";
  const width = 700, height = 330;
  const nodes = Array.from({length: data.node_count}, (_, i) => i);
  const positions = {};
  const radius = Math.max(105, Math.min(145, data.node_count * 3.5));
  const cx = width / 2, cy = height / 2;

  nodes.forEach((node, i) => {
    const angle = (-Math.PI / 2) + (2 * Math.PI * i / nodes.length);
    positions[node] = {x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle)};
  });

  data.edges.forEach(edge => {
    const a = positions[edge.source], b = positions[edge.target];
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
    line.setAttribute("class", edge.cut ? "edge-line edge-cut" : "edge-line");
    svg.appendChild(line);
  });

  const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
  nodes.forEach(node => {
    const label = data.partition["0"].includes(node) ? 0 : 1;
    const p = positions[node];
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", p.x); circle.setAttribute("cy", p.y); circle.setAttribute("r", 14);
    circle.setAttribute("fill", label === 0 ? "#8b7fff" : "#25d4a5");
    circle.setAttribute("class", "node-circle");
    group.appendChild(circle);
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", p.x); text.setAttribute("y", p.y);
    text.setAttribute("class", "node-label");
    text.textContent = node;
    group.appendChild(text);
  });
  svg.appendChild(group);
}

async function runPartition() {
  hideError();
  partitionBtn.disabled = true;
  partitionBtn.textContent = "Partitioning…";
  try {
    const count = Number(nodeCount.value);
    const edges = parseEdges();
    if (!Number.isInteger(count) || count < 2 || count > 500) throw new Error("Number of nodes must be an integer from 2 to 500.");

    const response = await fetch(API_URL + "/api/partition", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({node_count: count, edges, algorithm: algorithm.value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Partitioning failed.");

    $("algorithmBadge").textContent = algorithm.options[algorithm.selectedIndex].text.split(" — ")[0];
    $("cutSize").textContent = data.metrics.cut_size;
    $("balance").textContent = data.metrics.balance.toFixed(3);
    $("normalizedCut").textContent = data.metrics.normalized_cut.toFixed(3);
    $("edgeCount").textContent = data.edge_count;
    renderNodes("partitionA", data.partition["0"]);
    renderNodes("partitionB", data.partition["1"]);
    $("sizeA").textContent = data.partition["0"].length + " nodes";
    $("sizeB").textContent = data.partition["1"].length + " nodes";
    drawGraph(data);
    emptyState.classList.add("hidden");
    results.classList.remove("hidden");
  } catch (error) {
    showError(error.message);
  } finally {
    partitionBtn.disabled = false;
    partitionBtn.innerHTML = "Partition Network <span>→</span>";
  }
}

$("sampleBtn").addEventListener("click", loadSample);
$("randomBtn").addEventListener("click", randomEdges);
$("partitionBtn").addEventListener("click", runPartition);
checkApi();
