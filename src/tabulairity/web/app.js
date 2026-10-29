const $ = (id) => document.getElementById(id);
const esc = (v) =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fmt = (v) => (typeof v === "string" ? v : JSON.stringify(v, null, 2));
const state = {
  id: null,
  doc: null,
  network: null,
  selection: null,
  view: "graph",
  inventory: {},
  run: null,
  cursor: 0,
  offset: 0,
  polling: false,
  statuses: {},
  transform: { x: 0, y: 0, k: 1 },
  traces: 0,
  graphFrame: null,
  seenDrafts: new Set(),
  fxName: null,
  history: [],
  historyIndex: -1,
  savedDoc: null,
  runActive: false,
  lastAssistant: null,
  appliedRuns: new Set(),
  fxDrafts: new Map(),
  files: [],
  eventBacklog: false,
  transcriptPaused: false,
  runDefinition: null,
};
const titles = {
  graph: ["", "CDEN editor", ""],
  inputs: ["", "Inputs", "Files and variables available to your CDENs."],
  atoms: ["", "Atoms", "Small workflows with named inputs and outputs."],
  models: ["", "Models", "Routes, local servers and model roles."],
  fx: ["", "Functions", "Python chatFX modules."],
  runs: ["", "Runs", "Results, execution records and inference counts."],
  review: ["", "Review", "Flags and proposed revisions."],
};
const builtinFx = [
  "pass",
  "insertJson",
  "mergeJson",
  "getYN",
  "expandIterable",
  "null",
  "isYes",
  "isNo",
];
const fxLabel = (name) =>
  ({
    pass: "Keep response text",
    insertJson: "Read JSON",
    getYN: "Read yes / no",
    expandIterable: "Process each item",
    null: "Always continue",
    isYes: "If yes",
    isNo: "If no",
    mergeJson: "Merge fields into run variables",
  })[name] || name;
const net = () => state.doc?.nets[state.network];
const base = () => "/api/projects/" + state.id;
const fxNames = () => [
  ...new Set([...builtinFx, ...Object.values(state.inventory).flat()]),
];
function toast(message, error = false) {
  $("toast").textContent = message;
  $("toast").style.background = error ? "#91394a" : "#4c2d63";
  $("toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(
    () => ($("toast").hidden = true),
    error ? 9000 : 4000,
  );
}
function historySnapshot() {
  return {
    doc: JSON.stringify(state.doc),
    network: state.network,
    selection: structuredClone(state.selection),
  };
}
function updateHistoryControls() {
  $("undo-edit").disabled = state.historyIndex <= 0;
  $("redo-edit").disabled = state.historyIndex >= state.history.length - 1;
}
function updateSaveState() {
  const changed = state.doc && JSON.stringify(state.doc) !== state.savedDoc;
  const fxChanged = [...state.fxDrafts.keys()].some((key) =>
    key.startsWith(state.id + "/"),
  );
  $("save-state").textContent = changed
    ? "Unsaved edits"
    : fxChanged
      ? "Function unsaved"
      : "Saved";
}
function resetHistory() {
  ensurePositions();
  state.history = [historySnapshot()];
  state.historyIndex = 0;
  state.lastEdit = null;
  state.savedDoc = JSON.stringify(state.doc);
  updateHistoryControls();
  updateSaveState();
}
function dirty() {
  const snapshot = historySnapshot();
  if (state.history[state.historyIndex]?.doc !== snapshot.doc) {
    const el = document.activeElement;
    const typing = el?.matches(
      "textarea, input[type=text], input[type=number]",
    );
    const merge =
      typing &&
      state.lastEdit?.id === el.id &&
      Date.now() - state.lastEdit.time < 800 &&
      state.historyIndex > 0;
    state.history.splice(state.historyIndex + 1);
    if (merge) state.history[state.historyIndex] = snapshot;
    else {
      state.history.push(snapshot);
      state.historyIndex++;
    }
    if (state.history.length > 50) {
      state.history.shift();
      state.historyIndex--;
    }
    state.lastEdit = typing ? { id: el.id, time: Date.now() } : null;
  }
  updateSaveState();
  updateHistoryControls();
  $("validation-report").hidden = true;
}
function restoreEdit(delta) {
  const index = state.historyIndex + delta;
  if (index < 0 || index >= state.history.length) return;
  state.historyIndex = index;
  const snapshot = state.history[index];
  state.doc = JSON.parse(snapshot.doc);
  state.network = state.doc.nets[snapshot.network]
    ? snapshot.network
    : state.doc.entry;
  state.selection = structuredClone(snapshot.selection);
  state.lastEdit = null;
  updateSaveState();
  renderAll();
  updateHistoryControls();
}
$("undo-edit").onclick = () => restoreEdit(-1);
$("redo-edit").onclick = () => restoreEdit(1);
window.addEventListener("beforeunload", (event) => {
  if (
    state.doc &&
    (JSON.stringify(state.doc) !== state.savedDoc || state.fxDrafts.size)
  ) {
    event.preventDefault();
    event.returnValue = "";
  }
});
document.addEventListener("keydown", (event) => {
  if (!(event.ctrlKey || event.metaKey)) return;
  const editing = event.target.matches(
    "input, textarea, select, [contenteditable]",
  );
  if (event.key.toLowerCase() === "s") {
    event.preventDefault();
    protect(save)();
  }
  if (!editing && event.key.toLowerCase() === "z") {
    event.preventDefault();
    restoreEdit(event.shiftKey ? 1 : -1);
  }
});
$("shortcuts").onclick = () =>
  inspect(
    "Keyboard shortcuts",
    "Ctrl / Cmd + S: Save project\nCtrl / Cmd + Z: Undo graph edit\nCtrl / Cmd + Shift + Z: Redo\nTab: Move between controls and graph nodes\nEnter / Space: Select a node or edge\nUse the inspector's node list to select without the canvas.",
  );
async function api(path, method = "GET", body) {
  const r = await fetch(path, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-Tabulairity": "workbench",
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  const value = await r.json();
  if (!r.ok) throw new Error(value.error || r.statusText);
  return value;
}
function protect(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (e) {
      toast(e.message, true);
    }
  };
}
function options(items, selected, label = (v) => v) {
  return items
    .map(
      (v) =>
        `<option value="${esc(v)}" ${v === selected ? "selected" : ""}>${esc(label(v))}</option>`,
    )
    .join("");
}
function modal(title, html, submit, label = "Continue") {
  $("dialog-title").textContent = title;
  $("dialog-body").innerHTML = html;
  $("dialog-error").hidden = true;
  $("dialog-submit").textContent = label;
  $("dialog-submit").hidden = !submit;
  $("dialog-cancel").textContent = submit ? "Cancel" : "Close";
  $("dialog-form").onsubmit = async (event) => {
    event.preventDefault();
    if (!submit) return $("dialog").close();
    $("dialog-submit").disabled = true;
    try {
      await submit();
      $("dialog").close();
    } catch (e) {
      $("dialog-error").textContent = e.message;
      $("dialog-error").hidden = false;
    } finally {
      $("dialog-submit").disabled = false;
    }
  };
  $("dialog").showModal();
}
$("dialog-close").onclick = $("dialog-cancel").onclick = () =>
  $("dialog").close();
function inspect(title, value) {
  modal(title, "<pre>" + esc(fmt(value)) + "</pre>", null);
}
async function projectList(selected = state.id) {
  const projects = await api("/api/projects");
  $("project-select").innerHTML =
    '<option value="">Choose a project</option>' +
    options(
      projects.map((p) => p.id),
      selected,
      (id) => projects.find((p) => p.id === id).title,
    );
  return projects;
}
async function loadProject(id) {
  if (!id) return;
  if (
    state.id &&
    state.id !== id &&
    JSON.stringify(state.doc) !== state.savedDoc
  )
    await save();
  const info = await api("/api/projects/" + id);
  state.id = id;
  state.doc = info.project;
  state.inventory = info.fx_inventory;
  state.files = info.files;
  state.fxName = null;
  resetTranscript();
  state.network = state.doc.entry;
  state.selection = null;
  state.run = null;
  state.cursor = 0;
  state.offset = 0;
  state.statuses = {};
  state.transform = { x: 0, y: 0, k: 1 };
  $("empty").hidden = true;
  $("workspace").hidden = false;
  $("save-state").textContent = "Saved";
  resetHistory();
  renderAll();
  await projectList(id);
  await refreshRuns();
}
async function save() {
  if (!state.doc) throw new Error("Create or choose a project first");
  const invalid = document.querySelector("[data-invalid-json]");
  if (invalid) throw new Error("Correct the invalid JSON before saving.");
  await api(base(), "PUT", { project: state.doc });
  state.savedDoc = JSON.stringify(state.doc);
  state.lastEdit = null;
  updateSaveState();
}
function newProject() {
  modal(
    "Create a project",
    '<label>Project name<input id="new-project-title" type="text" value="My research project" required></label><p class="muted small">Includes runnable samples. No model needed to try them.</p>',
    async () => {
      const result = await api("/api/projects", "POST", {
        title: $("new-project-title").value,
      });
      await loadProject(result.id);
    },
    "Create project",
  );
}
$("new-project").onclick = $("empty-new").onclick = newProject;
$("project-select").onchange = protect(async (e) => {
  try {
    await loadProject(e.target.value);
  } catch (error) {
    e.target.value = state.id || "";
    throw error;
  }
});
$("save-project").onclick = protect(save);
$("export-project").onclick = protect(async () => {
  await save();
  location.href = base() + "/export";
});
function showView(view) {
  if (!state.doc) return toast("The workspace is still opening.");
  state.view = view;
  document
    .querySelectorAll(".view")
    .forEach((el) => (el.hidden = el.id !== "view-" + view));
  document.querySelectorAll("[data-view]").forEach((el) => {
    el.classList.toggle("active", el.dataset.view === view);
    el.setAttribute(
      "aria-current",
      el.dataset.view === view ? "page" : "false",
    );
  });
  const [eyebrow, title, desc] = titles[view];
  $("view-eyebrow").textContent = eyebrow;
  $("view-title").textContent = title;
  $("view-description").textContent = desc;
  $("assistant-open").textContent =
    view === "fx" ? "Draft function" : "Draft prompt";
  if (view === "graph") renderGraph();
  if (view === "inputs") protect(renderInputs)();
  if (view === "runs")
    protect(async () => {
      await refreshRuns();
      await pollRun();
    })();
  if (view === "review") protect(renderReview)();
  if (view === "fx") protect(renderFx)();
  if (view === "atoms") renderAtoms();
}
document
  .querySelectorAll("[data-view]")
  .forEach((button) => (button.onclick = () => showView(button.dataset.view)));
function renderAll() {
  renderNetworks();
  renderRoutes();
  renderRoles();
  renderRunSettings();
  renderFiles();
  renderWorkflowSelectors();
  renderInspector();
  renderAtoms();
  showView(state.view);
}
function renderNetworks() {
  const pools = [
    ["user", "User"],
    ["samples", "Samples"],
    ["internal", "Internal · workbench tools"],
  ];
  const visibleNetwork = (name) =>
    !state.doc.nets[name].atom_owner || name === state.network;
  $("network-select").innerHTML = pools
    .map(
      ([pool, label]) =>
        '<optgroup label="' +
        esc(label) +
        '">' +
        options(
          Object.keys(state.doc.nets).filter(
            (n) =>
              visibleNetwork(n) && (state.doc.nets[n].pool || "user") === pool,
          ),
          state.network,
          (name) => (state.doc.nets[name].title || name) + " (" + name + ")",
        ) +
        "</optgroup>",
    )
    .join("");
  $("internal-warning").hidden = net().pool !== "internal";
  $("network-description").textContent = net().description || "";
  $("use-example").hidden = !net().example_input;
  const library = $("cden-pools");
  library.replaceChildren();
  for (const [pool, label] of pools) {
    const names = Object.keys(state.doc.nets).filter(
      (name) =>
        visibleNetwork(name) && (state.doc.nets[name].pool || "user") === pool,
    );
    const section = document.createElement("section");
    section.dataset.pool = pool;
    const heading = document.createElement("h2");
    heading.textContent = label + " · " + names.length;
    section.append(heading);
    const list = document.createElement("div");
    list.className = "pool-list";
    for (const name of names) {
      const network = state.doc.nets[name];
      const button = document.createElement("button");
      button.className = "pool-network";
      button.dataset.network = name;
      button.setAttribute("aria-pressed", String(name === state.network));
      const title = document.createElement("strong");
      title.textContent = network.title || name;
      const description = document.createElement("span");
      description.textContent = network.description || name;
      button.append(title, description);
      button.onclick = () => {
        selectNetwork(name);
        $("cden-library").open = false;
      };
      list.append(button);
    }
    if (!names.length) {
      const empty = document.createElement("p");
      empty.className = "muted small";
      empty.textContent = pool === "user" ? "No user CDENs yet." : "No CDENs.";
      list.append(empty);
    }
    section.append(list);
    library.append(section);
  }
  $("set-entry").textContent =
    state.network === state.doc.entry ? "Entry CDEN" : "Set as entry";
  $("set-entry").disabled = state.network === state.doc.entry;
}

function selectNetwork(name) {
  cancelAnimationFrame(state.graphFrame);
  state.network = name;
  state.selection = null;
  state.lastEdit = null;
  $("validation-report").hidden = true;
  state.transform = { x: 0, y: 0, k: 1 };
  renderNetworks();
  renderGraph();
  renderInspector();
  fitGraph();
}
$("library-search").oninput = (e) => {
  const query = e.target.value.toLowerCase();
  document.querySelectorAll(".pool-network").forEach((button) => {
    button.hidden = !(button.textContent + button.dataset.network)
      .toLowerCase()
      .includes(query);
  });
};
$("network-select").onchange = (e) => selectNetwork(e.target.value);
$("add-bundled").onclick = protect(async () => {
  await save();
  const result = await api(base() + "/bundled", "POST");
  state.doc = result.project;
  renderAll();
  toast(
    result.added.length
      ? "Added " + result.added.length + " bundled CDENs."
      : "Bundled CDENs are already available.",
  );
});
$("use-example").onclick = protect(async () => {
  state.doc.entry = state.network;
  state.doc.input = net().example_input;
  state.doc.output_node = net().example_output_node || "";
  await save();
  renderAll();
  showView("inputs");
  toast("Sample inputs selected. Open Runs & results to execute this example.");
});
$("set-entry").onclick = () => {
  state.doc.entry = state.network;
  state.doc.output_node = "";
  dirty();
  renderNetworks();
  renderRunSettings();
};
function checkedName(value) {
  const name = value.trim();
  if (
    !/^[A-Za-z0-9][A-Za-z0-9_. -]{0,100}$/.test(name) ||
    [".", ".."].includes(name)
  )
    throw new Error(
      "Use a name starting with a letter or number, up to 101 characters.",
    );
  return name;
}
$("new-network").onclick = () =>
  modal(
    "New CDEN",
    '<label>Name<input id="new-network-name" type="text" required value="new_network"></label><label>Starting point<select id="new-network-template"><option value="blank">Blank graph</option><option value="prompt">One model step</option></select></label><p class="small muted">Add nodes to build manually, or use Draft CDEN to describe a workflow.</p>',
    () => {
      const name = checkedName($("new-network-name").value);
      if (state.doc.nets[name])
        throw new Error("A CDEN already has that name.");
      const nodes = [
        {
          key: "Start",
          fx: "pass",
          prompt: "recall: ready",
          persona: "",
          tokens: 1,
          x: 180,
          y: 150,
        },
      ];
      const edges = [];
      if ($("new-network-template").value === "prompt") {
        nodes.push({
          key: "extract",
          fx: "pass",
          prompt: "Summarize [text] in one sentence.",
          persona: "Use only the supplied text.",
          model: state.doc.settings.assistant,
          tokens: 512,
          x: 400,
          y: 280,
        });
        edges.push({
          source: "Start",
          target: "extract",
          fx: "null",
          prompt: "",
        });
      }
      state.doc.nets[name] = { pool: "user", nodes, edges };
      state.network = name;
      state.selection = { type: "node", key: nodes.at(-1).key };
      dirty();
      renderAll();
      fitGraph();
    },
    "Create CDEN",
  );
$("duplicate-network").onclick = () =>
  modal(
    "Duplicate CDEN",
    '<label>New name<input id="duplicate-name" type="text" value="' +
      esc(state.network + "_copy") +
      '" required></label>',
    () => {
      const name = checkedName($("duplicate-name").value);
      if (state.doc.nets[name]) throw new Error("Choose a new CDEN name");
      state.doc.nets[name] = structuredClone(net());
      state.doc.nets[name].pool = "user";
      state.network = name;
      state.selection = null;
      dirty();
      renderNetworks();
      renderGraph();
      renderInspector();
    },
    "Duplicate to User pool",
  );
$("duplicate-internal").onclick = () => $("duplicate-network").click();

$("add-node").onclick = () =>
  modal(
    "Add a node",
    '<label>Node name<input id="new-node-name" type="text" required value="extract_value"></label>',
    () => {
      const name = $("new-node-name").value.trim();
      if (
        !name ||
        ["success", "errors"].includes(name) ||
        net().nodes.some((n) => n.key === name)
      )
        throw new Error("Choose a unique node name");
      const source =
        state.selection?.type === "node" ? state.selection.key : "Start";
      const parent = net().nodes.find((n) => n.key === source);
      net().nodes.push({
        key: name,
        x: parent ? parent.x + 170 : 250,
        y: parent ? parent.y + 100 : 250,
        fx: "pass",
        prompt: "[text]",
        persona:
          "Extract only the requested information from the supplied text.",
        tokens: 2000,
        model: state.doc.settings.assistant,
      });
      if (parent)
        net().edges.push({ source, target: name, fx: "null", prompt: "" });
      state.selection = { type: "node", key: name };
      dirty();
      renderGraph();
      renderInspector();
      fitGraph();
    },
    "Add node",
  );
$("add-edge").onclick = () =>
  modal(
    "Connect two nodes",
    `<label>From<select id="edge-from">${options(
      net().nodes.map((n) => n.key),
      state.selection?.key,
    )}</select></label><label>To<select id="edge-to">${options(
      net().nodes.map((n) => n.key),
      net().nodes.at(-1)?.key,
    )}</select></label><label>Condition<select id="edge-fx">${options(
      fxNames().filter((n) => n !== "expandIterable"),
      "null",
    )}</select></label><p class="small muted">null opens the edge. isYes and isNo branch on the upstream response.</p>`,
    () => {
      const source = $("edge-from").value,
        target = $("edge-to").value;
      if (source === target) throw new Error("Choose two different nodes.");
      if (net().edges.some((e) => e.source === source && e.target === target))
        throw new Error("Those nodes are already connected.");
      const pending = [target],
        seen = new Set();
      while (pending.length) {
        const key = pending.pop();
        if (key === source)
          throw new Error("This connection would create a cycle.");
        if (seen.has(key)) continue;
        seen.add(key);
        net()
          .edges.filter((e) => e.source === key)
          .forEach((e) => pending.push(e.target));
      }
      net().edges.push({ source, target, fx: $("edge-fx").value, prompt: "" });
      dirty();
      renderGraph();
      renderInspector();
    },
    "Add edge",
  );
$("weld-net").onclick = () =>
  modal(
    "Join CDENs",
    `<p>Connect a downstream CDEN to an existing upstream node. The downstream Start and any colliding keys receive new names.</p><label>Upstream CDEN<select id="weld-up">${options(Object.keys(state.doc.nets), state.network)}</select></label><label>Upstream node<select id="weld-node">${options(
      net().nodes.map((n) => n.key),
      state.selection?.key,
    )}</select></label><label>Downstream CDEN<select id="weld-down">${options(Object.keys(state.doc.nets).filter((n) => n !== state.network))}</select></label><label>New Start key (optional)<input id="weld-key" type="text"></label>`,
    async () => {
      await save();
      const r = await api(base() + "/weld", "POST", {
        upstream: $("weld-up").value,
        downstream: $("weld-down").value,
        node: $("weld-node").value,
        start_key: $("weld-key").value,
      });
      state.doc = r.project;
      state.selection = null;
      renderAll();
      toast("Joined CDENs. Renamed keys: " + JSON.stringify(r.rename_map));
    },
    "Join CDENs",
  );
$("dialog-body").addEventListener("change", (e) => {
  if (e.target.id === "weld-up")
    $("weld-node").innerHTML = options(
      state.doc.nets[e.target.value].nodes.map((n) => n.key),
    );
});
function atomStepCount(atom) {
  return Object.values(atom.networks)
    .flatMap((n) => n.nodes)
    .filter((n) => n.persona && !n.prompt?.startsWith("recall:")).length;
}
function atomCost(atom) {
  const nested = Object.values(atom.networks).some((n) =>
    n.nodes.some((x) => x.fx === "expandIterable"),
  );
  return `${atomStepCount(atom)} model steps${nested ? " · repeated for nested items" : ""}`;
}
function renderAtoms() {
  if (!state.doc) return;
  const list = $("atoms-list");
  list.replaceChildren();
  for (const [id, atom] of Object.entries(state.doc.atoms || {})) {
    const card = document.createElement("article");
    card.className = "card atom-card";
    card.dataset.atom = id;
    const title = document.createElement("h2");
    title.textContent = atom.title;
    const desc = document.createElement("p");
    desc.textContent = atom.description || "Reusable CDEN";
    const cost = document.createElement("p");
    cost.className = "atom-cost";
    cost.textContent = atomCost(atom);
    const ports = document.createElement("p");
    ports.className = "small muted";
    ports.textContent = `Inputs: ${
      Object.entries(atom.inputs)
        .map(([name, spec]) => spec.label || name.replace(/_/g, " "))
        .join(", ") || "none"
    } → Outputs: ${Object.keys(atom.outputs)
      .map((name) => name.replace(/_/g, " "))
      .join(", ")}`;
    const actions = document.createElement("div");
    actions.className = "section-row";
    for (const [label, action] of [
      ["Try atom", () => tryAtom(id)],
      ["Insert into CDEN", () => insertAtomDialog(id)],
      ["Inspect steps", () => inspectAtom(id)],
      [
        "Export atom",
        () =>
          download(
            id + ".json",
            JSON.stringify(atom, null, 2),
            "application/json",
          ),
      ],
    ]) {
      const button = document.createElement("button");
      button.textContent = label;
      button.onclick = protect(action);
      actions.append(button);
    }
    card.append(title, desc, cost, ports, actions);
    list.append(card);
  }
  if (!list.childElementCount)
    list.innerHTML =
      '<p class="muted">No atoms yet. Save a CDEN as an atom, import one, or add the bundled examples from the CDEN browser.</p>';
}
function inspectAtom(id) {
  const atom = state.doc.atoms[id];
  const stages = Object.entries(atom.networks)
    .map(
      ([name, network]) =>
        `<h3>${esc(name)}</h3>` +
        network.nodes
          .map(
            (node) =>
              `<details><summary>${esc(node.key)} · ${node.persona ? "Model step" : "Data step"}</summary><p class="small">${esc(node.fx || "pass")}</p><pre>${esc(node.prompt || "")}</pre><p>${esc(node.persona || "")}</p></details>`,
          )
          .join(""),
    )
    .join("");
  modal(
    atom.title,
    `<p>${esc(atomCost(atom))}. Actual requests and cache hits appear in Runs.</p>${atom.authoring_request ? `<details><summary>Original request</summary><p>${esc(atom.authoring_request)}</p></details>` : ""}${stages}`,
    () => {},
    "Close",
  );
}
function tryAtom(id) {
  if (
    atomStepCount(state.doc.atoms[id]) &&
    !state.doc.routes.some(
      (route) => route.model === state.doc.settings.assistant,
    )
  ) {
    showView("models");
    toast("Choose a model before trying this atom.");
    return;
  }
  const atom = state.doc.atoms[id],
    ports = Object.entries(atom.inputs);
  modal(
    "Try " + atom.title,
    `<p>${esc(atomCost(atom))}. The trial is saved in Runs with its inputs and full transcript.</p>${ports.map(([name, spec], i) => `<label>${esc(spec.label || name)}<textarea id="atom-try-${i}" rows="6" required placeholder="${esc(spec.description || name)}"></textarea></label>`).join("")}`,
    async () => {
      const values = Object.fromEntries(
        ports.map(([name], i) => [name, $("atom-try-" + i).value]),
      );
      if (Object.values(values).some((value) => !value.trim()))
        throw new Error("Fill in each atom input");
      await save();
      await startJob("atom_test", { atom: id, values });
    },
    "Run trial",
  );
}
async function insertAtomDialog(id) {
  const atom = state.doc.atoms[id],
    ports = Object.entries(atom.inputs);
  let columns = [];
  try {
    columns = Object.keys((await api(base() + "/input")).rows[0] || {});
  } catch (_) {}
  const after =
    state.selection?.type === "node" ? state.selection.key : "Start";
  const defaultVariable =
    after !== "Start"
      ? after
      : columns.includes("text")
        ? "text"
        : columns[0] || "text";
  modal(
    "Insert " + atom.title,
    `<p>Add to <strong>${esc(net().title || state.network)}</strong> · ${esc(atomCost(atom))}.</p>
    <label>Step name<input id="atom-instance" type="text" value="${esc((/^[A-Za-z]/.test(id) ? id : "atom_" + id).replace(/[^A-Za-z0-9_]/g, "_").slice(0, 60))}" required pattern="[A-Za-z][A-Za-z0-9_]{0,59}" title="Start with a letter; use letters, numbers and underscores"></label>
    <label>Run after<select id="atom-after">${options(
      displayedGraph().nodes.map((n) => n.key),
      after,
    )}</select></label>
    <datalist id="atom-variables">${[...new Set([...columns, ...displayedGraph().nodes.map((n) => n.key)])].map((key) => `<option value="${esc(key)}"></option>`).join("")}</datalist>
    ${ports.map(([name, spec], i) => `<fieldset class="atom-binding"><legend>${esc(spec.label || name)}</legend><p class="small muted">${esc(spec.description || "")}</p><label>Take input from<select id="atom-mode-${i}">${options(["variable", "literal"], "variable", (v) => (v === "variable" ? "Input column or earlier step" : "Fixed text"))}</select></label><label>Column / step name, or fixed text<textarea id="atom-value-${i}" rows="2">${esc(defaultVariable)}</textarea></label></fieldset>`).join("")}
    <p class="small muted">Connect the inserted step to the next step when needed. Existing branches stay in place.</p>`,
    async () => {
      const bindings = Object.fromEntries(
        ports.map(([name], i) => [
          name,
          $("atom-mode-" + i).value === "variable"
            ? { mode: "variable", variable: $("atom-value-" + i).value.trim() }
            : { mode: "literal", value: $("atom-value-" + i).value },
        ]),
      );
      const result = await api(base() + "/atom-insert", "POST", {
        project: state.doc,
        network: state.network,
        atom: id,
        after: $("atom-after").value,
        instance: $("atom-instance").value,
        bindings,
      });
      state.doc = result.project;
      state.selection = { type: "node", key: result.instance };
      dirty();
      renderAll();
      showView("graph");
      fitGraph();
      toast(
        `Added ${result.instance}. Use [${result.instance}] in a later prompt.`,
      );
    },
    "Insert atom",
  );
}
$("add-atom").onclick = () => {
  const ids = Object.keys(state.doc.atoms || {});
  if (!ids.length) {
    showView("atoms");
    return;
  }
  modal(
    "Choose an atom",
    `<label>Atom<select id="choose-atom">${options(ids, ids[0], (id) => state.doc.atoms[id].title)}</select></label>`,
    async () => {
      const id = $("choose-atom").value;
      setTimeout(() => protect(() => insertAtomDialog(id))(), 0);
    },
    "Choose",
  );
};
function saveAtomDialog() {
  modal(
    "Save CDEN as an atom",
    `<p>Capture <strong>${esc(net().title || state.network)}</strong> and its nested CDENs. Later edits to this graph will not change saved copies.</p><label>Name<input id="atom-save-name" type="text" value="${esc(state.network)}" required></label><label>Title<input id="atom-save-title" type="text" value="${esc(net().title || state.network)}" required></label><label>Return the value from<select id="atom-save-output">${options(
      net().nodes.map((n) => n.key),
      state.selection?.key || net().nodes.at(-1)?.key,
    )}</select></label><p class="small muted">Input columns are detected from bracket references. Graphs with variable-writing functions or dynamic references need an isolation adapter before they can become atoms.</p>`,
    async () => {
      const id = checkedName($("atom-save-name").value);
      if (state.doc.atoms?.[id]) throw new Error("Choose a new atom name");
      const result = await api(base() + "/atom-save", "POST", {
        project: state.doc,
        network: state.network,
        title: $("atom-save-title").value,
        output: $("atom-save-output").value,
      });
      (state.doc.atoms ||= {})[id] = result.atom;
      dirty();
      renderAtoms();
      showView("atoms");
    },
    "Save atom",
  );
}
$("save-atom").onclick = saveAtomDialog;
$("atom-from-current").onclick = saveAtomDialog;
$("atom-file").onchange = protect(async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  await upload(file, "atom");
  renderAtoms();
  event.target.value = "";
});
function renderAtomInspector(group) {
  const panel = $("inspector"),
    bindings = Object.entries(group.bindings);
  panel.innerHTML = `<div class="eyebrow">ATOM</div><h2>${esc(group.title)}</h2><p><code>[${esc(group.key)}]</code> returns ${Object.keys(group.outputs).length === 1 ? "the result" : "named output fields"}.</p><p class="atom-cost">${group.model_steps} model steps${group.networks.length ? " plus nested item steps" : ""}</p>
    <button id="atom-toggle">${group.collapsed ? "Show internal steps" : "Collapse internal steps"}</button>
    ${bindings.map(([name, binding], i) => `<fieldset class="atom-binding"><legend>${esc(group.input_labels?.[name] || name.replace(/_/g, " "))}</legend><label>Take input from<select id="binding-mode-${i}">${options(["variable", "literal"], binding.mode, (v) => (v === "variable" ? "Input column or earlier step" : "Fixed text"))}</select></label><label>Value<textarea id="binding-value-${i}" rows="3">${esc(binding.mode === "variable" ? binding.variable : typeof binding.value === "string" ? binding.value : JSON.stringify(binding.value))}</textarea></label></fieldset>`).join("")}
    <label>Model for these steps<select id="atom-model">${options([...new Set(["[assistant_model]", ...state.doc.routes.map((r) => r.model), group.model || "[assistant_model]"])], group.model || "[assistant_model]", (name) => (name === "[assistant_model]" ? "Project assistant" : name))}</select></label>
    <div class="section-row"><button id="atom-connect">Connect…</button><button id="atom-remove" class="danger">Remove atom</button></div><details><summary>Advanced</summary><button id="atom-ungroup">Ungroup into ordinary steps</button><p class="small muted">Keeps the graph and its result. Removes the atom's combined controls.</p></details><details><summary>Saved definition</summary><p class="small">Version ${esc(group.version)}. This insertion is an independent copy.</p><code class="atom-hash">${esc(group.definition_sha256)}</code></details>`;
  $("atom-toggle").onclick = () => {
    group.collapsed = !group.collapsed;
    dirty();
    renderGraph();
    renderAtomInspector(group);
    fitGraph();
  };
  for (const [i, [name]] of bindings.entries()) {
    const update = () => {
      const value = $("binding-value-" + i).value;
      const binding =
        $("binding-mode-" + i).value === "variable"
          ? { mode: "variable", variable: value.trim() }
          : { mode: "literal", value };
      group.bindings[name] = binding;
      const input = net().nodes.find((n) => n.key === group.node_map[name]);
      if (!input)
        return toast(
          "The atom input step was removed. Undo that edit or ungroup the atom.",
          true,
        );
      input.fx_params = binding;
      input.prompt =
        binding.mode === "variable"
          ? `recall: [${binding.variable}]`
          : "recall:";
      dirty();
      renderGraph();
    };
    $("binding-mode-" + i).onchange = update;
    $("binding-value-" + i).oninput = update;
  }
  $("atom-model").onchange = (event) => {
    group.model = event.target.value;
    for (const node of [
      ...net().nodes.filter((n) => group.members.includes(n.key)),
      ...group.networks.flatMap((name) => state.doc.nets[name]?.nodes || []),
    ])
      if (node.persona && !node.prompt?.startsWith("recall:"))
        node.model = group.model;
    dirty();
    renderGraph();
  };
  $("atom-connect").onclick = () => $("add-edge").click();
  $("atom-ungroup").onclick = () => {
    net().atom_instances = net().atom_instances.filter(
      (g) => g.key !== group.key,
    );
    dirty();
    renderAll();
    renderGraph();
    fitGraph();
  };
  $("atom-remove").onclick = () => {
    const members = new Set(group.members);
    net().nodes = net().nodes.filter((n) => !members.has(n.key));
    net().edges = net().edges.filter(
      (e) => !members.has(e.source) && !members.has(e.target),
    );
    net().forced_dependencies = (net().forced_dependencies || []).filter(
      ([a, b]) => !members.has(a) && !members.has(b),
    );
    net().atom_instances = net().atom_instances.filter(
      (g) => g.key !== group.key,
    );
    if (state.doc.entry === state.network && members.has(state.doc.output_node))
      state.doc.output_node = "";
    const candidates = new Set(group.networks);
    const used = new Set(
      Object.entries(state.doc.nets)
        .filter(([name]) => !candidates.has(name))
        .flatMap(([, n]) => n.nodes.map((x) => x.fx_params?.chatNet)),
    );
    if (![...candidates].some((name) => used.has(name)))
      for (const name of candidates) delete state.doc.nets[name];
    state.selection = null;
    dirty();
    renderAll();
    renderGraph();
  };
}
function summary() {
  const n = net();
  const steps = n.nodes.filter(
    (node) => node.persona && !node.prompt?.startsWith("recall:"),
  ).length;
  return `${displayedGraph().nodes.length} visible steps · ${(n.atom_instances || []).length} atoms · ${steps} model steps · ${n.nodes.filter((node) => node.fx === "expandIterable").length} item loops · ${n.edges.filter((e) => e.fx && e.fx !== "null").length} conditions`;
}
function graphIssues(network = net()) {
  const issues = [],
    keys = new Set(network.nodes.map((n) => n.key));
  if (!keys.has("Start")) issues.push("Add a Start node.");
  const reached = new Set(),
    pending = keys.has("Start") ? ["Start"] : [];
  while (pending.length) {
    const key = pending.pop();
    if (reached.has(key)) continue;
    reached.add(key);
    network.edges
      .filter((e) => e.source === key)
      .forEach((e) => pending.push(e.target));
  }
  const detached = [...keys].filter((k) => !reached.has(k));
  if (detached.length) issues.push("Connect to Start: " + detached.join(", "));
  for (const node of network.nodes) {
    if (node.persona && !node.prompt?.trim())
      issues.push(node.key + ": add a prompt.");
    if (
      node.fx === "expandIterable" &&
      !state.doc.nets[node.fx_params?.chatNet]
    )
      issues.push(node.key + ": choose a nested CDEN.");
  }
  return issues;
}
function renderHealth() {
  const issues = graphIssues();
  $("graph-health").classList.toggle("has-errors", !!issues.length);
  $("graph-health").textContent = issues.length
    ? issues.join(" · ")
    : "Connected graph · Check validates inputs and nested CDENs";
}
async function checkGraph(entry = state.network) {
  const doc = structuredClone(state.doc);
  if (doc.entry !== entry) doc.output_node = "";
  doc.entry = entry;
  $("validate-graph").disabled = true;
  let report;
  try {
    report = await api(base() + "/validate", "POST", { project: doc });
  } finally {
    $("validate-graph").disabled = false;
  }
  const panel = $("validation-report");
  panel.hidden = false;
  panel.classList.toggle("valid", report.valid);
  panel.innerHTML =
    "<strong>" +
    (report.valid ? "Ready to run" : "Resolve before running") +
    "</strong><ul>" +
    [...report.errors, ...report.warnings]
      .map((issue) => "<li>" + esc(issue.message) + "</li>")
      .join("") +
    "</ul>";
  return report;
}
$("validate-graph").onclick = protect(() => checkGraph());
$("graph-run").onclick = protect(async () => {
  $("run-status").textContent = "preparing";
  const report = await checkGraph();
  if (!report.valid) return;
  if (state.doc.entry !== state.network)
    state.doc.output_node = net().example_output_node || "";
  state.doc.entry = state.network;
  dirty();
  renderRunSettings();
  await save();
  await startJob("run", { trust_fx: $("run-trust-fx").checked });
});
function svg(tag, attrs = {}, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}
function ensurePositions() {
  const width = $("graph").clientWidth || 700,
    height = $("graph").clientHeight || 570;
  for (const network of Object.values(state.doc.nets))
    network.nodes.forEach((node, i) => {
      if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) {
        const a = (i * 2 * Math.PI) / Math.max(1, network.nodes.length);
        node.x = width / 2 + Math.sin(a) * Math.min(width * 0.28, 200);
        node.y = height / 2 - Math.cos(a) * 160;
      }
    });
}
function atomGroup(key) {
  return (net().atom_instances || []).find((g) => g.key === key);
}
function displayedGraph(network = net()) {
  const hidden = new Map();
  for (const group of network.atom_instances || [])
    if (group.collapsed)
      for (const key of group.members)
        if (key !== group.key) hidden.set(key, group.key);
  return {
    nodes: network.nodes.filter((n) => !hidden.has(n.key)),
    edges: network.edges
      .map((e, index) => ({
        ...e,
        source: hidden.get(e.source) || e.source,
        target: hidden.get(e.target) || e.target,
        originalIndex: index,
      }))
      .filter((e) => e.source !== e.target),
  };
}
function renderGraph() {
  if (!state.doc || state.view !== "graph") return;
  ensurePositions();
  const world = $("graph-world");
  const focusedNode = document.activeElement
    ?.closest?.("[data-node]")
    ?.getAttribute("data-node");
  const focusedEdge = document.activeElement?.getAttribute?.("data-edge");
  world.replaceChildren();
  const t = state.transform;
  world.setAttribute("transform", `translate(${t.x},${t.y}) scale(${t.k})`);
  const display = displayedGraph();
  const positions = new Map(display.nodes.map((n) => [n.key, n]));
  display.edges.forEach((edge) => {
    const i = edge.originalIndex;
    const a = positions.get(edge.source),
      b = positions.get(edge.target);
    if (!a || !b) return;
    const dx = b.x - a.x,
      dy = b.y - a.y,
      d = Math.max(1, Math.hypot(dx, dy));
    const start = { x: a.x + (dx / d) * 43, y: a.y + (dy / d) * 30 },
      end = { x: b.x - (dx / d) * 48, y: b.y - (dy / d) * 34 };
    const line = svg("path", {
      d: `M${start.x},${start.y} L${end.x},${end.y}`,
      class:
        "edge" +
        (state.selection?.type === "edge" && state.selection.index === i
          ? " selected"
          : ""),
      "marker-end": "url(#arrow)",
    });
    const hit = svg("path", {
      d: line.getAttribute("d"),
      class: "edge-hit",
      tabindex: 0,
      role: "button",
      "aria-label": "Edit connection " + edge.source + " to " + edge.target,
      "data-edge": i,
    });
    line.onclick = (e) => {
      e.stopPropagation();
      state.selection = { type: "edge", index: i };
      renderGraph();
      renderInspector();
    };
    hit.onclick = line.onclick;
    hit.onkeydown = (e) => {
      if (["Enter", " "].includes(e.key)) {
        e.preventDefault();
        line.onclick(e);
      }
    };
    world.append(line, hit);
    if (edge.fx && edge.fx !== "null") {
      const label = svg(
        "text",
        { x: (a.x + b.x) / 2 + 5, y: (a.y + b.y) / 2 - 7, class: "edge-label" },
        edge.fx,
      );
      label.onclick = line.onclick;
      world.append(label);
    }
  });
  display.nodes.forEach((node) => {
    const atom = atomGroup(node.key);
    const member = (net().atom_instances || []).find((g) =>
      g.members.includes(node.key),
    );
    const localName =
      member && !atom
        ? Object.keys(member.node_map).find(
            (key) => member.node_map[key] === node.key,
          )
        : node.key;
    const name = atom ? node.key.replace(/_/g, " ") : localName || node.key;
    const memberStates = atom
      ? atom.members.map((key) => state.statuses[key])
      : [];
    const status = memberStates.includes("failed")
      ? "failed"
      : memberStates.includes("running")
        ? "running"
        : state.statuses[node.key] || "";
    const selected =
      state.selection?.type === "node" && state.selection.key === node.key;
    const group = svg("g", {
      transform: `translate(${node.x - 78},${node.y - 30})`,
      class:
        "node " +
        status +
        (atom ? " atom-node" : "") +
        (selected ? " selected" : ""),
      "data-node": node.key,
      tabindex: 0,
      role: "button",
      "aria-label": "Edit " + node.key,
    });
    group.append(
      svg("rect", { width: 156, height: 60, rx: 9 }),
      svg("circle", { cx: 14, cy: 16, r: 3, class: "dot" }),
      svg(
        "text",
        { x: 25, y: 20, class: "name" },
        name.length > 18 ? name.slice(0, 17) + "…" : name,
      ),
      svg(
        "text",
        { x: 14, y: 43, class: "fx" },
        atom ? `Atom · ${atom.model_steps} model steps` : node.fx || "pass",
      ),
    );
    group.append(svg("title", {}, node.key));
    group.onpointerdown = (e) => {
      e.stopPropagation();
      cancelAnimationFrame(state.graphFrame);
      state.selection = { type: "node", key: node.key };
      renderInspector();
      const graph = $("graph");
      graph.setPointerCapture(e.pointerId);
      const move = (ev) => {
        const r = graph.getBoundingClientRect();
        const x = (ev.clientX - r.left - state.transform.x) / state.transform.k,
          y = (ev.clientY - r.top - state.transform.y) / state.transform.k;
        if (atom?.collapsed)
          for (const child of net().nodes)
            if (child.key !== node.key && atom.members.includes(child.key)) {
              child.x += x - node.x;
              child.y += y - node.y;
            }
        node.x = x;
        node.y = y;
        renderGraph();
      };
      const end = () => {
        graph.removeEventListener("pointermove", move);
        graph.removeEventListener("pointerup", end);
        dirty();
        renderGraph();
      };
      graph.addEventListener("pointermove", move);
      graph.addEventListener("pointerup", end, { once: true });
      renderGraph();
    };
    group.onkeydown = (e) => {
      if (["Enter", " "].includes(e.key)) {
        e.preventDefault();
        state.selection = { type: "node", key: node.key };
        renderGraph();
        renderInspector();
      }
    };
    world.append(group);
  });
  $("graph-summary").textContent = summary();
  if (focusedNode)
    [...world.querySelectorAll("[data-node]")]
      .find((element) => element.dataset.node === focusedNode)
      ?.focus({ preventScroll: true });
  else if (focusedEdge !== null)
    [...world.querySelectorAll("[data-edge]")]
      .find((element) => element.dataset.edge === focusedEdge)
      ?.focus({ preventScroll: true });
  renderHealth();
}
$("graph").onpointerdown = (e) => {
  if (
    e.target.closest(".node") ||
    e.target.closest(".edge") ||
    e.target.closest(".edge-hit")
  )
    return;
  const x = e.clientX,
    y = e.clientY,
    initial = { ...state.transform };
  $("graph").setPointerCapture(e.pointerId);
  const move = (ev) => {
    state.transform.x = initial.x + ev.clientX - x;
    state.transform.y = initial.y + ev.clientY - y;
    renderGraph();
  };
  const end = () => {
    $("graph").removeEventListener("pointermove", move);
    $("graph").removeEventListener("pointerup", end);
  };
  $("graph").addEventListener("pointermove", move);
  $("graph").addEventListener("pointerup", end, { once: true });
};
$("graph").addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const r = $("graph").getBoundingClientRect(),
      x = e.clientX - r.left,
      y = e.clientY - r.top;
    const old = state.transform.k,
      next = Math.min(2.5, Math.max(0.25, old * Math.exp(-e.deltaY * 0.001)));
    state.transform.x = x - ((x - state.transform.x) * next) / old;
    state.transform.y = y - ((y - state.transform.y) * next) / old;
    state.transform.k = next;
    renderGraph();
  },
  { passive: false },
);
function fitGraph() {
  if (!state.doc || state.view !== "graph" || !net().nodes.length) return;
  ensurePositions();
  const nodes = displayedGraph().nodes;
  const minX = Math.min(...nodes.map((n) => n.x)) - 90,
    maxX = Math.max(...nodes.map((n) => n.x)) + 90;
  const minY = Math.min(...nodes.map((n) => n.y)) - 45,
    maxY = Math.max(...nodes.map((n) => n.y)) + 45;
  const w = $("graph").clientWidth,
    h =
      $("graph").clientHeight -
      document.querySelector(".graph-hint").offsetHeight -
      20;
  const k = Math.min(1.4, w / (maxX - minX + 30), h / (maxY - minY + 50));
  state.transform = {
    k,
    x: w / 2 - ((minX + maxX) / 2) * k,
    y: h / 2 - ((minY + maxY) / 2) * k,
  };
  renderGraph();
}
$("fit-graph").onclick = fitGraph;
new ResizeObserver(() => fitGraph()).observe($("graph"));

function forceLayout() {
  cancelAnimationFrame(state.graphFrame);
  ensurePositions();
  state.transform = { x: 0, y: 0, k: 1 };
  const nodes = displayedGraph().nodes;
  const reducedMotion = window.matchMedia?.(
    "(prefers-reduced-motion: reduce)",
  ).matches;
  const width = $("graph").clientWidth,
    height = $("graph").clientHeight;
  const velocities = new Map(nodes.map((n) => [n.key, { x: 0, y: 0 }]));
  let frame = 0;
  function step() {
    if (state.view !== "graph") return;
    const forces = new Map(
      nodes.map((n) => [
        n.key,
        { x: (width / 2 - n.x) * 0.002, y: (height / 2 - n.y) * 0.002 },
      ]),
    );
    for (let i = 0; i < nodes.length; i++)
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i],
          b = nodes[j],
          dx = a.x - b.x || 0.1,
          dy = a.y - b.y || 0.1,
          d = Math.max(50, Math.hypot(dx, dy)),
          f = 4500 / (d * d);
        forces.get(a.key).x += (dx / d) * f;
        forces.get(a.key).y += (dy / d) * f;
        forces.get(b.key).x -= (dx / d) * f;
        forces.get(b.key).y -= (dy / d) * f;
      }
    for (const e of displayedGraph().edges) {
      const a = nodes.find((n) => n.key === e.source),
        b = nodes.find((n) => n.key === e.target);
      if (!a || !b) continue;
      const dx = b.x - a.x,
        dy = b.y - a.y,
        d = Math.max(1, Math.hypot(dx, dy)),
        f = (d - 170) * 0.014;
      forces.get(a.key).x += (dx / d) * f;
      forces.get(a.key).y += (dy / d) * f;
      forces.get(b.key).x -= (dx / d) * f;
      forces.get(b.key).y -= (dy / d) * f;
    }
    for (const n of nodes) {
      const v = velocities.get(n.key),
        f = forces.get(n.key);
      v.x = (v.x + f.x) * 0.78;
      v.y = (v.y + f.y) * 0.78;
      n.x = Math.max(
        90,
        Math.min(width - 90, n.x + Math.max(-7, Math.min(7, v.x))),
      );
      n.y = Math.max(
        50,
        Math.min(height - 60, n.y + Math.max(-7, Math.min(7, v.y))),
      );
    }
    renderGraph();
    if (++frame < 80 && nodes.length < 100 && !reducedMotion)
      state.graphFrame = requestAnimationFrame(step);
    else {
      fitGraph();
      dirty();
    }
  }
  step();
}
$("layout-graph").onclick = forceLayout;
function renderInspector() {
  const selection = state.selection,
    panel = $("inspector");
  if (!selection) {
    panel.innerHTML =
      '<div class="inspector-placeholder"><h2>Structure</h2><p>Select a node to edit its prompt, model and function.</p><button id="inspector-add-node">+ Node</button><div class="structure-list"></div></div>';
    $("inspector-add-node").onclick = () => $("add-node").click();
    const list = panel.querySelector(".structure-list");
    for (const node of displayedGraph().nodes) {
      const button = document.createElement("button");
      button.textContent = node.key;
      button.onclick = () => {
        state.selection = { type: "node", key: node.key };
        renderGraph();
        renderInspector();
      };
      list.append(button);
    }
    return;
  }
  if (selection.type === "edge") {
    const edge = net().edges[selection.index];
    if (!edge) {
      state.selection = null;
      return renderInspector();
    }
    panel.innerHTML = `<div class="eyebrow">EDGE</div><h2>${esc(edge.source)} → ${esc(edge.target)}</h2><label>Condition<select id="inspect-edge-fx">${options([...new Set([...fxNames().filter((x) => x !== "expandIterable"), edge.fx])], edge.fx || "null")}</select></label><label>Display message<textarea id="inspect-edge-prompt" rows="4">${esc(edge.prompt || "")}</textarea></label><button id="delete-selection" class="danger">Remove edge</button>`;
    $("inspect-edge-fx").onchange = (e) => {
      edge.fx = e.target.value;
      dirty();
      renderGraph();
    };
    $("inspect-edge-prompt").oninput = (e) => {
      edge.prompt = e.target.value;
      dirty();
    };
    $("delete-selection").onclick = () => {
      net().edges.splice(selection.index, 1);
      state.selection = null;
      dirty();
      renderGraph();
      renderInspector();
    };
    return;
  }
  const node = net().nodes.find((n) => n.key === selection.key);
  if (!node) {
    state.selection = null;
    return renderInspector();
  }
  const group = atomGroup(node.key);
  if (group) return renderAtomInspector(group);
  const member = (net().atom_instances || []).find((g) =>
    g.members.includes(node.key),
  );
  if (member && ["atom_input", "atom_output"].includes(node.fx)) {
    state.selection = { type: "node", key: member.key };
    return renderAtomInspector(member);
  }
  const models = [
    ...new Set([
      ...(state.doc.routes || []).map((r) => r.model),
      node.model || state.doc.settings.assistant,
      "[assistant_model]",
      "[supervisor_model]",
    ]),
  ];
  const expand = node.fx === "expandIterable";
  const params = node.fx_params || {};
  panel.innerHTML = `<div class="eyebrow">NODE</div><h2>${esc(node.key)}</h2><label>Prompt<textarea id="inspect-prompt" rows="6">${esc(node.prompt || "")}</textarea></label><label>System prompt<textarea id="inspect-persona" rows="3">${esc(node.persona || "")}</textarea></label><label>Handle the response<select id="inspect-fx">${options([...new Set([...fxNames(), node.fx])], node.fx || "pass", fxLabel)}</select></label>${
    expand
      ? `<div class="card nested-settings"><label>Nested CDEN<select id="expand-net">${options(
          Object.keys(state.doc.nets).filter((n) => n !== state.network),
          params.chatNet,
        )}</select></label><button id="open-nested" class="text-button">Open nested CDEN</button><p class="small muted">Child input: <code>[${esc(node.key)}_item]</code></p><label>Failed items<select id="expand-policy">${options(["drop", "keep", "raise"], params.on_error || "drop")}</select></label><label>Parallel items<input id="expand-concurrency" type="number" min="1" value="${esc(params.concurrency || 1)}"></label><label><input id="expand-dataframe" type="checkbox" ${params.as_dataframe ? "checked" : ""}> Return a DataFrame</label><label>Output fields (comma separated)<input id="expand-keys" type="text" value="${esc((params.output_keys || []).join(", "))}"></label></div>`
      : ""
  }<div class="section-row"><label>Model<select id="inspect-model">${options(models, node.model || state.doc.settings.assistant)}</select></label><label>Output tokens<input id="inspect-tokens" type="number" min="1" value="${esc(node.tokens || 2000)}"></label></div><details><summary>Advanced parameters</summary><label>Model parameters (JSON)<textarea id="inspect-extra" rows="3">${esc(JSON.stringify(node.extra_params || {}, null, 2))}</textarea></label><label>FX parameters (JSON)<textarea id="inspect-params" rows="4">${esc(JSON.stringify(node.fx_params || {}, null, 2))}</textarea></label><p id="inspect-json-error" class="field-error" role="alert" hidden></p><label><input id="inspect-self-eval" type="checkbox" ${node.self_eval ? "checked" : ""}> Run legacy self evaluation</label><p class="small muted">Legacy self evaluation can add model calls; those calls appear in the transcript.</p></details><div class="section-row"><button id="connect-selection">Connect…</button><button id="delete-selection" class="danger" ${node.key === "Start" ? 'disabled title="Start is required"' : ""}>Remove node</button></div><button id="inspector-structure" class="text-button">All nodes</button>`;
  $("connect-selection").onclick = () => $("add-edge").click();
  $("inspector-structure").onclick = () => {
    state.selection = null;
    renderGraph();
    renderInspector();
  };
  for (const [id, field] of [
    ["inspect-prompt", "prompt"],
    ["inspect-persona", "persona"],
  ])
    $(id).oninput = (e) => {
      node[field] = e.target.value;
      dirty();
      renderGraph();
    };
  $("inspect-fx").onchange = (e) => {
    node.fx = e.target.value;
    if (node.fx === "expandIterable" && !node.fx_params?.chatNet)
      node.fx_params = {
        chatNet: Object.keys(state.doc.nets).find((n) => n !== state.network),
        on_error: "drop",
        concurrency: 1,
      };
    dirty();
    renderGraph();
    renderInspector();
  };
  $("inspect-model").onchange = (e) => {
    node.model = e.target.value;
    dirty();
  };
  $("inspect-tokens").onchange = (e) => {
    node.tokens = Number(e.target.value);
    dirty();
  };
  $("inspect-self-eval").onchange = (e) => {
    node.self_eval = e.target.checked;
    dirty();
  };
  for (const [id, field] of [
    ["inspect-extra", "extra_params"],
    ["inspect-params", "fx_params"],
  ])
    $(id).onchange = (e) => {
      try {
        const value = JSON.parse(e.target.value || "{}");
        if (!value || Array.isArray(value) || typeof value !== "object")
          throw new Error("Use a JSON object, such as {}.");
        delete e.target.dataset.invalidJson;
        e.target.setCustomValidity("");
        $("inspect-json-error").hidden = true;
        node[field] = value;
        dirty();
      } catch (error) {
        e.target.dataset.invalidJson = "true";
        e.target.setCustomValidity(error.message);
        $("inspect-json-error").textContent = error.message;
        $("inspect-json-error").hidden = false;
      }
    };
  if (expand) {
    $("open-nested").onclick = () => selectNetwork($("expand-net").value);
    const update = () => {
      node.fx_params = {
        ...node.fx_params,
        chatNet: $("expand-net").value,
        on_error: $("expand-policy").value,
        concurrency: Number($("expand-concurrency").value),
        as_dataframe: $("expand-dataframe").checked,
      };
      const keys = $("expand-keys")
        .value.split(",")
        .map((v) => v.trim())
        .filter(Boolean);
      if (keys.length) node.fx_params.output_keys = keys;
      else delete node.fx_params.output_keys;
      $("inspect-params").value = JSON.stringify(node.fx_params, null, 2);
      dirty();
    };
    for (const id of [
      "expand-net",
      "expand-policy",
      "expand-concurrency",
      "expand-dataframe",
      "expand-keys",
    ])
      $(id).onchange = update;
  }
  $("delete-selection").onclick = () => {
    net().nodes = net().nodes.filter((n) => n.key !== node.key);
    net().edges = net().edges.filter(
      (e) => e.source !== node.key && e.target !== node.key,
    );
    if (state.doc.entry === state.network && state.doc.output_node === node.key)
      state.doc.output_node = "";
    net().forced_dependencies = (net().forced_dependencies || []).filter(
      ([a, b]) => a !== node.key && b !== node.key,
    );
    state.selection = null;
    dirty();
    renderGraph();
    renderInspector();
    renderRunSettings();
  };
  if (member) {
    $("delete-selection").disabled = true;
    $("delete-selection").title =
      "Ungroup the atom before removing its internal steps";
    panel.insertAdjacentHTML(
      "afterbegin",
      `<button id="member-atom" class="text-button">${esc(member.title)} settings</button>`,
    );
    $("member-atom").onclick = () => {
      state.selection = { type: "node", key: member.key };
      renderGraph();
      renderAtomInspector(member);
    };
  }
}
async function upload(file, kind) {
  await save();
  const r = await api(base() + "/upload", "POST", {
    name: file.name,
    kind,
    content: await file.text(),
  });
  state.doc = r.project;
  state.inventory = r.fx_inventory;
  state.files = r.files;
  if (["rankings", "routes"].includes(kind)) await suggestRoles(false);
  resetHistory();
  state.network = kind === "cden" ? state.doc.entry : state.network;
  renderAll();
  toast("Imported " + file.name);
  return r;
}
for (const [id, kind] of [
  ["input-file", "input"],
  ["cden-file", "cden"],
  ["routes-file", "routes"],
  ["rankings-file", "rankings"],
  ["fx-file", "fx"],
])
  $(id).onchange = protect(async (e) => {
    for (const file of e.target.files) await upload(file, kind);
    e.target.value = "";
    if (kind === "input") await renderInputs();
    if (kind === "fx") await renderFx();
  });
function renderFiles() {
  const paths = [
    ...new Set([
      ...(state.files || []),
      ...(state.doc.input ? [state.doc.input] : []),
    ]),
    ...Object.keys(state.doc.nets).map(
      (n) => "cdens/" + (state.doc.nets[n].pool || "user") + "/" + n + ".json",
    ),
    ...Object.keys(state.inventory).map((n) => "fx/" + n),
  ];
  $("project-files").innerHTML = paths
    .map(
      (path) =>
        `<div class="file-row"><span class="muted">▤</span><button data-file="${esc(path)}">${esc(path)}</button></div>`,
    )
    .join("");
  $("project-files")
    .querySelectorAll("[data-file]")
    .forEach(
      (button) =>
        (button.onclick = protect(async () => {
          const r = await api(
            base() + "/file?name=" + encodeURIComponent(button.dataset.file),
          );
          inspect(button.dataset.file, r.content);
        })),
    );
}
function renderTable(target, rows, { flag = false, flags = {} } = {}) {
  if (!rows.length) {
    target.innerHTML = '<p class="muted">No rows yet.</p>';
    return;
  }
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))].filter(
    (k) => !k.startsWith("_"),
  );
  target.innerHTML =
    "<table><thead><tr>" +
    (flag ? "<th>Review</th>" : "") +
    cols.map((k) => "<th>" + esc(k) + "</th>").join("") +
    "</tr></thead><tbody>" +
    rows
      .map(
        (row, i) =>
          "<tr>" +
          (flag
            ? `<td><button data-flag="${i}" class="${flags[row._row_id] ? "soft" : ""}">${flags[row._row_id] ? "Flagged" : "Flag"}</button></td>`
            : "") +
          cols
            .map(
              (k) =>
                `<td><button class="cell-value" data-row="${i}" data-column="${esc(k)}">${esc(fmt(row[k]) ?? "")}</button></td>`,
            )
            .join("") +
          "</tr>",
      )
      .join("") +
    "</tbody></table>";
  target
    .querySelectorAll("[data-column]")
    .forEach(
      (b) =>
        (b.onclick = () =>
          inspect(
            b.dataset.column,
            rows[Number(b.dataset.row)][b.dataset.column],
          )),
    );
  target.querySelectorAll("[data-flag]").forEach(
    (b) =>
      (b.onclick = () => {
        const row = rows[Number(b.dataset.flag)];
        modal(
          "Flag this extraction",
          `<label>Observable issue<textarea id="flag-reason" rows="4">${esc(flags[row._row_id]?.reason || "")}</textarea></label><label>Node<select id="flag-node">${options(state.runDefinition?.nodes || [], flags[row._row_id]?.node || state.runDefinition?.output_node || "Start")}</select></label><p class="small muted">Describe what is wrong in the output and where the supporting evidence should be.</p>`,
          async () => {
            await api(base() + "/runs/" + state.run + "/flag", "POST", {
              row_id: row._row_id,
              reason: $("flag-reason").value,
              node: $("flag-node").value,
            });
            await pollRun();
          },
          "Save flag",
        );
      }),
  );
}
async function renderInputs() {
  if (!state.doc) return;
  const r = await api(base() + "/input");
  $("input-info").textContent = r.files.join(" · ");
  $("input-pattern").value = state.doc.input || "";
  renderTable($("input-preview"), r.rows);
  renderFiles();
}
function renderRoutes() {
  $("routes-table").innerHTML =
    "<thead><tr><th>Model name</th><th>Provider / model route</th><th>Backend address or remote</th><th>Key environment variable</th><th></th></tr></thead><tbody>" +
    state.doc.routes
      .map(
        (r, i) =>
          "<tr>" +
          ["model", "route", "ip", "key_env"]
            .map(
              (k) =>
                `<td><input data-route="${i}" data-field="${k}" aria-label="${k} row ${i + 1}" value="${esc(r[k] || "")}"></td>`,
            )
            .join("") +
          `<td><button data-remove-route="${i}" aria-label="Remove route">×</button></td></tr>`,
      )
      .join("") +
    "</tbody>";
  $("routes-table")
    .querySelectorAll("[data-field]")
    .forEach(
      (input) =>
        (input.oninput = (e) => {
          state.doc.routes[Number(e.target.dataset.route)][
            e.target.dataset.field
          ] = e.target.value;
          dirty();
          renderRoles();
        }),
    );
  $("routes-table")
    .querySelectorAll("[data-remove-route]")
    .forEach(
      (b) =>
        (b.onclick = () => {
          state.doc.routes.splice(Number(b.dataset.removeRoute), 1);
          dirty();
          renderRoutes();
          renderRoles();
        }),
    );
}
$("add-route").onclick = () => {
  state.doc.routes.push({
    model: "new-model",
    route: "openai/model-id",
    ip: "http://127.0.0.1:1234/v1",
    key_env: "",
  });
  dirty();
  renderRoutes();
  renderRoles();
};
function renderRoles() {
  const aliases = [
    ...new Set(state.doc.routes.map((r) => r.model).filter(Boolean)),
  ];
  for (const role of ["assistant", "supervisor"]) {
    let current = state.doc.settings[role];
    if (aliases.length && !aliases.includes(current)) {
      current = aliases.length === 1 ? aliases[0] : "";
      state.doc.settings[role] = current;
    }
    const select = $(role + "-model");
    select.innerHTML =
      (!aliases.length
        ? '<option value="">Discover or import routes to set this role</option>'
        : !current
          ? '<option value="">Choose an available model</option>'
          : "") + options(aliases, current);
    select.disabled = !aliases.length;
    select.onchange = (e) => {
      state.doc.settings[role] = e.target.value;
      (state.doc.role_overrides ||= {})[role] = true;
      dirty();
      renderRoles();
    };
  }
  $("assistant-open").disabled = false;
  $("assistant-open").title = aliases.includes(state.doc.settings.assistant)
    ? "Draft with " + state.doc.settings.assistant
    : "Add a model route to generate drafts";
  $("supervisor-review").disabled = !aliases.includes(
    state.doc.settings.supervisor,
  );
}

function endpointRow(
  endpoint = { kind: "ollama", url: "http://127.0.0.1:11434", key_env: "" },
) {
  return (
    '<tr><td><select class="endpoint-kind">' +
    options(["ollama", "lmstudio", "litellm", "owui"], endpoint.kind) +
    '</select></td><td><input class="endpoint-url" type="text" value="' +
    esc(endpoint.url) +
    '"></td><td><input class="endpoint-key" type="text" value="' +
    esc(endpoint.key_env || "") +
    '"></td><td><button type="button" class="endpoint-remove">×</button></td></tr>'
  );
}
$("discover-open").onclick = protect(async () => {
  const defaults = await api("/api/endpoints");
  modal(
    "Find local inference backends",
    '<p class="small muted">Enter an address and port, such as localhost:1234. The workbench asks each backend which models it serves.</p><div class="table-scroll"><table id="endpoint-table"><thead><tr><th>Backend</th><th>Address and port</th><th>Key env. variable</th><th></th></tr></thead><tbody>' +
      defaults.map(endpointRow).join("") +
      '</tbody></table></div><button type="button" id="add-endpoint">+ Computer / backend</button>',
    async () => {
      const endpoints = [
        ...$("endpoint-table").querySelectorAll("tbody tr"),
      ].map((row) => ({
        kind: row.querySelector(".endpoint-kind").value,
        url: row.querySelector(".endpoint-url").value,
        key_env: row.querySelector(".endpoint-key").value,
      }));
      const found = await api(base() + "/discover", "POST", { endpoints });
      const seen = new Set(
        state.doc.routes.map((r) => [r.model, r.route, r.ip].join("|")),
      );
      for (const r of found.routes)
        if (!seen.has([r.model, r.route, r.ip].join("|"))) {
          state.doc.routes.push(r);
          seen.add([r.model, r.route, r.ip].join("|"));
        }
      for (const role of ["assistant", "supervisor"])
        if (
          found.recommendation[role] &&
          (!state.doc.role_overrides?.[role] ||
            !state.doc.routes.some((r) => r.model === state.doc.settings[role]))
        )
          state.doc.settings[role] = found.recommendation[role];
      $("discovery-info").textContent =
        found.probes.map((p) => p.endpoint.url + ": " + p.status).join(" · ") +
        " · " +
        found.recommendation.basis;
      dirty();
      renderRoutes();
      renderRoles();
      toast("Found " + found.routes.length + " model routes");
    },
    "Find models",
  );
  $("add-endpoint").onclick = () => {
    $("endpoint-table")
      .querySelector("tbody")
      .insertAdjacentHTML("beforeend", endpointRow());
  };
  $("endpoint-table").onclick = (e) => {
    if (e.target.classList.contains("endpoint-remove"))
      e.target.closest("tr").remove();
  };
});

async function suggestRoles(force = false) {
  const result = await api(base() + "/recommend-roles", "POST", {
    routes: state.doc.routes,
    rankings: state.doc.rankings || [],
  });
  for (const role of ["assistant", "supervisor"]) {
    if (
      result[role] &&
      (force ||
        !state.doc.role_overrides?.[role] ||
        !state.doc.routes.some((r) => r.model === state.doc.settings[role]))
    ) {
      state.doc.settings[role] = result[role];
      if (force) (state.doc.role_overrides ||= {})[role] = false;
    }
  }
  $("discovery-info").textContent = result.basis;
  dirty();
  renderRoles();
}
$("recommend-roles").onclick = protect(() => suggestRoles(true));

$("download-routes").onclick = () => {
  const keys = ["model", "route", "ip", "key_env"];
  const quote = (x) => '"' + String(x ?? "").replaceAll('"', '""') + '"';
  download(
    "model_routes.csv",
    keys.join(",") +
      "\n" +
      state.doc.routes
        .map((r) => keys.map((k) => quote(r[k])).join(","))
        .join("\n"),
    "text/csv",
  );
};
function download(name, content, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([content], { type })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("test-routes").onclick = protect(async () => {
  await save();
  await startJob("route_test", {
    models: [...new Set(state.doc.routes.map((r) => r.model))],
  });
});
const fxTemplate =
  'def trim_text(response, chatVars):\n    """Trim surrounding whitespace without changing extracted evidence."""\n    return response.strip()\n\n\nCHAT_FX = {"trim_text": trim_text}\n';
const fxDraftKey = () => state.id + "/" + (state.fxName || "__new__");
$("fx-code").oninput = () => {
  state.fxDrafts.set(fxDraftKey(), $("fx-code").value);
  $("save-state").textContent = "Function unsaved";
};
async function renderFx() {
  const names = Object.keys(state.inventory);
  if (state.fxName && !names.includes(state.fxName)) state.fxName = null;
  const projectId = state.id,
    moduleName = state.fxName,
    draftKey = fxDraftKey();
  $("fx-select").innerHTML =
    '<option value="">Unsaved module</option>' + options(names, state.fxName);
  $("fx-heading").textContent = state.fxName || "New module";
  $("fx-inventory").textContent = (state.inventory[state.fxName] || []).join(
    ", ",
  );
  $("test-fx-name").innerHTML = options(Object.values(state.inventory).flat());
  if (state.fxDrafts.has(fxDraftKey())) {
    $("fx-code").value = state.fxDrafts.get(fxDraftKey());
  } else if (state.fxName) {
    const r = await api(
      base() + "/file?name=" + encodeURIComponent("fx/" + state.fxName),
    );
    if (state.id !== projectId || state.fxName !== moduleName) return;
    $("fx-code").value = state.fxDrafts.get(draftKey) ?? r.content;
  } else $("fx-code").value = fxTemplate;
}
$("fx-select").onchange = protect(async (e) => {
  state.fxName = e.target.value || null;
  await renderFx();
});
$("new-fx").onclick = () => {
  state.fxName = null;
  $("fx-code").value = state.fxDrafts.get(fxDraftKey()) || fxTemplate;
  $("fx-heading").textContent = "New module";
  $("fx-select").value = "";
};
$("save-fx").onclick = protect(async () => {
  const saveModule = async (name) => {
    const draftKey = fxDraftKey();
    await save();
    const r = await api(base() + "/upload", "POST", {
      name,
      kind: "fx",
      content: $("fx-code").value,
    });
    state.inventory = r.fx_inventory;
    state.doc = r.project;
    state.fxDrafts.delete(draftKey);
    state.fxName = name;
    resetHistory();
    updateSaveState();
    await renderFx();
    toast("Saved Python module");
  };
  if (state.fxName) await saveModule(state.fxName);
  else
    modal(
      "Save chatFX module",
      '<label>Python filename<input id="fx-save-name" type="text" value="cleaners.py" required></label>',
      () => saveModule($("fx-save-name").value),
      "Save module",
    );
});
$("test-fx").onclick = protect(async () => {
  await save();
  await startJob("fx_test", {
    fx: $("test-fx-name").value,
    text: $("test-fx-text").value,
    variables: JSON.parse($("test-fx-vars").value),
    trust_fx: $("trust-fx").checked,
  });
});
function renderRunSettings() {
  const settings = state.doc.settings;
  const keys = [
    ...new Set([
      ...displayedGraph(state.doc.nets[state.doc.entry]).nodes.map(
        (n) => n.key,
      ),
      ...(state.doc.output_node ? [state.doc.output_node] : []),
    ]),
  ];
  $("output-node").innerHTML =
    '<option value="">All extracted values</option>' +
    options(keys, state.doc.output_node);
  $("workers").value = settings.workers;
  $("draft-tokens").value = settings.draft_tokens || 4096;
  $("timeout").value = settings.timeout;
  $("auto-review").checked = Boolean(settings.auto_review);
  $("review-limit").value = settings.review_limit;
  $("run-input-label").textContent =
    "Entry: " + state.doc.entry + " · Input: " + state.doc.input;
}
for (const [id, key, type] of [
  ["workers", "workers", "number"],
  ["draft-tokens", "draft_tokens", "number"],
  ["timeout", "timeout", "number"],
  ["auto-review", "auto_review", "bool"],
  ["review-limit", "review_limit", "number"],
])
  $(id).onchange = (e) => {
    state.doc.settings[key] =
      type === "bool" ? e.target.checked : Number(e.target.value);
    dirty();
  };
$("output-node").onchange = (e) => {
  state.doc.output_node = e.target.value;
  dirty();
};
$("set-input-pattern").onclick = protect(async () => {
  state.doc.input = $("input-pattern").value.trim();
  await save();
  renderRunSettings();
  await renderInputs();
  toast("Updated work batch");
});
async function startJob(operation, args = {}) {
  $("run-status").textContent = "starting";
  const result = await api(base() + "/start", "POST", { operation, ...args });
  state.run = result.run_id;
  state.runActive = true;
  $("job-feedback").hidden = true;
  state.cursor = 0;
  state.offset = 0;
  resetTranscript();
  state.statuses = {};
  showView("runs");
  await refreshRuns();
  await pollRun();
  toast("Started " + operation);
}
$("run-start").onclick = protect(async () => {
  $("run-status").textContent = "preparing";
  await save();
  const report = await checkGraph(state.doc.entry);
  if (!report.valid) {
    showView("graph");
    throw new Error(report.errors.map((e) => e.message).join("; "));
  }
  await startJob("run", { trust_fx: $("run-trust-fx").checked });
});
for (const action of ["pause", "resume", "stop"])
  $("run-" + action).onclick = protect(async () => {
    if (!state.run) throw new Error("Select a running job");
    await api(base() + "/runs/" + state.run + "/control", "POST", { action });
    await pollRun(true);
  });
async function refreshRuns() {
  if (!state.id) return;
  const runs = await api(base() + "/runs");
  if (!state.run && runs.length) state.run = runs[0].run_id;
  $("run-select").innerHTML =
    '<option value="">Choose a run</option>' +
    options(
      runs.map((r) => r.run_id),
      state.run,
      (id) => {
        const r = runs.find((r) => r.run_id === id);
        return id + " · " + r.operation + " · " + r.status;
      },
    );
  const extractions = runs.filter((r) => r.operation === "run");
  $("review-run").innerHTML = options(
    extractions.map((r) => r.run_id),
    extractions.some((r) => r.run_id === state.run)
      ? state.run
      : extractions[0]?.run_id,
  );
}
$("run-select").onchange = protect(async (e) => {
  state.run = e.target.value;
  state.runActive = true;
  state.cursor = 0;
  state.offset = 0;
  resetTranscript();
  state.statuses = {};
  await pollRun();
});
function addTrace(event) {
  if (["node_start", "edge", "cache_hit", "input_start"].includes(event.event))
    return;
  const detail = document.createElement("details");
  detail.className = "trace-item";
  const summaryEl = document.createElement("summary");
  const path = event.path?.length ? " · " + event.path.join(" / ") : "";
  summaryEl.innerHTML = `<span class="badge">${esc(event.event.replaceAll("_", " "))}</span><strong>${esc(event.node || event.model || "Run")}${esc(path)}</strong><span class="trace-time">${esc(event.elapsed === undefined ? "" : event.elapsed.toFixed(2) + " s")}</span>`;
  detail.append(summaryEl);
  for (const [key, label] of [
    ["prompt", "Prompt"],
    ["persona", "System prompt"],
    ["messages", "Model request"],
    ["raw", "Raw response"],
    ["output", "Cleaned output"],
    ["row", "Result"],
    ["error", "Error"],
    ["draft", "Draft"],
    ["traceback", "Traceback"],
    ["normalisation", "Format adjustments"],
  ])
    if (event[key] !== undefined) {
      const heading = document.createElement("h3");
      heading.textContent = label;
      const pre = document.createElement("pre");
      pre.textContent = fmt(event[key]);
      detail.append(heading, pre);
    }
  detail.addEventListener("toggle", () => {
    if (detail.open) state.transcriptPaused = true;
  });
  $("transcript").append(detail);
  state.traces++;
}
async function pollRun(force = false) {
  if (!state.run) return;
  if (state.polling) {
    if (!force) return state.polling;
    await state.polling;
  }
  const request = pollRunOnce();
  state.polling = request;
  try {
    await request;
  } finally {
    if (state.polling === request) state.polling = null;
  }
}
async function pollRunOnce() {
  const id = state.id,
    run = state.run;
  const r = await api(
    base() +
      "/runs/" +
      run +
      "?after=" +
      state.cursor +
      "&offset=" +
      state.offset,
  );
  if (id !== state.id || run !== state.run) return;
  for (const e of r.events) {
    if (state.traces >= 240) break;
    state.cursor = e._cursor ?? r.cursor;
    if (e.node && (!e.network || e.network === state.network))
      state.statuses[e.node] =
        e.event === "node_start"
          ? "running"
          : e.event === "node_error"
            ? "failed"
            : e.event === "node_complete"
              ? "completed"
              : state.statuses[e.node];
    if (state.traces < 240) addTrace(e);
  }
  state.eventBacklog = Boolean(r.has_more_events || state.cursor < r.cursor);
  state.runDefinition = r.definition;
  $("transcript-status").textContent =
    state.traces +
    " displayed steps" +
    (state.eventBacklog ? " · More available" : "");
  $("more-transcript").disabled = !state.eventBacklog;
  $("download-transcript").href =
    base() + "/runs/" + encodeURIComponent(run) + "/transcript";
  const s = r.status;
  if (s.node_states) state.statuses = s.node_states[state.network] || {};
  $("run-status").textContent = s.status || "Pending";
  $("run-stats").innerHTML = [
    [
      (s.inference_calls ?? 0) + "",
      "Inference calls · " + (s.cache_hits ?? 0) + " cache hits",
    ],
    [(s.mean_calls_per_input ?? 0) + "", "Mean calls per completed input"],
    [
      (s.inputs_done ?? 0) + " / " + (s.rows ?? 0),
      "Inputs completed / output rows",
    ],
    [
      (s.elapsed ?? 0).toFixed(1) + " s",
      (s.output_length ?? 0) + " output characters",
    ],
  ]
    .map(
      ([v, label]) =>
        `<div class="stat"><strong>${esc(v)}</strong><span>${esc(label)}</span></div>`,
    )
    .join("");
  renderTable($("results-table"), r.rows, {
    flag: s.operation === "run",
    flags: r.flags,
  });
  $("results-page").textContent =
    "Rows " +
    (r.rows.length ? state.offset + 1 : 0) +
    "–" +
    (state.offset + r.rows.length) +
    " of " +
    (s.rows ?? 0);
  $("results-prev").disabled = state.offset === 0;
  $("results-next").disabled = state.offset + 100 >= (s.rows ?? 0);
  const active = ["starting", "running", "paused", "stopping"].includes(
    s.status,
  );
  state.runActive = active;
  const assistantJob = s.operation === "assistant";
  if (assistantJob && r.request) state.lastAssistant = r.request;
  $("constructor-diagnostics").hidden = !(assistantJob && r.request?.kind === "cden" && !active);
  $("constructor-diagnostics").href = base() + "/runs/" + encodeURIComponent(run) + "/constructor";
  const failed = ["failed", "interrupted", "stopped"].includes(s.status);
  $("job-feedback").hidden = !(
    assistantJob ||
    failed ||
    s.failed_inputs ||
    s.errors ||
    s.review_error
  );
  $("job-feedback").classList.toggle("failed", failed);
  $("job-message").textContent = failed
    ? s.error || "Run " + s.status + "."
    : s.review_error
      ? "Extraction finished; automatic review failed: " + s.review_error
      : s.failed_inputs || s.errors
        ? (s.failed_inputs || 0) +
          " inputs failed · " +
          (s.errors || 0) +
          " node errors. Inspect the transcript."
        : r.draft
          ? "Draft ready. Inspect it before applying."
          : "Creating draft. Inference calls are recorded below.";
  $("job-review").hidden = !r.draft || !assistantJob;
  $("job-retry").hidden = !assistantJob || !state.lastAssistant || active;
  $("job-retry").textContent = r.request?.kind === "cden" ? "Retry / adjust output limit" : "Edit request";
  $("job-models").hidden = !failed || !assistantJob;
  $("run-results").classList.toggle("assistant-result", assistantJob);
  $("results-table").hidden = assistantJob;
  $("results-prev").hidden =
    $("results-next").hidden =
    $("results-page").hidden =
      assistantJob;
  $("run-pause").disabled = !active || s.status === "paused";
  $("run-resume").disabled = s.status !== "paused";
  $("run-stop").disabled = !active;
  if (s.error) $("run-status").textContent = s.status + ": " + s.error;
  renderGraph();
  if (r.draft && state.view === "review" && !state.seenDrafts.has(run)) {
    state.seenDrafts.add(run);
    await renderReview();
  }
}
setInterval(() => {
  if (
    state.id &&
    state.run &&
    (state.runActive ||
      (state.eventBacklog && state.traces < 240 && !state.transcriptPaused)) &&
    !document.hidden
  )
    protect(pollRun)();
}, 1100);
$("export-csv").onclick = () => {
  if (state.run) location.href = base() + "/runs/" + state.run + "/csv";
  else toast("Select a run first", true);
};
$("results-prev").onclick = protect(async () => {
  state.offset = Math.max(0, state.offset - 100);
  await pollRun();
});
$("results-next").onclick = protect(async () => {
  state.offset += 100;
  await pollRun();
});
document.querySelectorAll("[data-run-tab]").forEach(
  (button) =>
    (button.onclick = () => {
      document
        .querySelectorAll("[data-run-tab]")
        .forEach((b) => b.classList.toggle("active", b === button));
      $("run-results").hidden = button.dataset.runTab !== "results";
      $("run-transcript").hidden = button.dataset.runTab !== "transcript";
    }),
);
function resetTranscript() {
  state.cursor = 0;
  state.traces = 0;
  state.eventBacklog = false;
  state.transcriptPaused = false;
  $("transcript").replaceChildren();
}
$("reset-transcript").onclick = protect(async () => {
  resetTranscript();
  await pollRun(true);
});
$("more-transcript").onclick = protect(async () => {
  state.transcriptPaused = true;
  $("more-transcript").disabled = true;
  if (state.polling) await state.polling;
  state.traces = 0;
  $("transcript").replaceChildren();
  try {
    await pollRun(true);
  } finally {
    $("more-transcript").disabled = !state.eventBacklog;
  }
});
function openAssistant(kind = "prompt", previous = null) {
  if (!state.doc) return toast("Create a project first", true);
  const current = previous ||
    (state.lastAssistant?.kind === kind ? state.lastAssistant : null) || {
      kind,
      node: state.selection?.key || net().nodes[0]?.key,
      request: "",
    };
  modal(
    kind === "cden" ? "Draft a CDEN" : "Draft with assistant",
    `<label>Draft<select id="assistant-kind">${options(["prompt", "fx", "cden"], current.kind, (v) => ({ prompt: "Node prompt", fx: "Python function", cden: "CDEN" })[v])}</select></label><label id="assistant-node-label">Target node<select id="assistant-node">${options(
      net().nodes.map((n) => n.key),
      current.node,
    )}</select></label><label>Describe the task<textarea id="assistant-request" rows="6" required placeholder="Input fields, required output, and the steps or conditions to follow.">${esc(current.request)}</textarea></label><p class="small muted">Model: ${esc(state.doc.settings.assistant)} · Drafts require your review.</p><button id="assistant-model-settings" type="button" class="text-button">Configure model route</button><label id="assistant-output-label">Constructor output limit<input id="assistant-output-limit" type="number" min="1" step="1" value="${esc(current.draft_tokens || state.doc.settings.draft_tokens || 4096)}"><span class="small muted">Raise this if the response is cut short. Backend context stays unchanged.</span></label><details id="assistant-guide"><summary>CDEN authoring guide</summary><pre class="small" id="assistant-guide-text"></pre></details><label><input id="assistant-trust" type="checkbox" ${current.trust_fx ? "checked" : ""}> Run project Python modules</label>`,
    async () => {
      const args = {
        kind: $("assistant-kind").value,
        ...($("assistant-kind").value === "cden" ? {draft_tokens: Number($("assistant-output-limit").value)} : {}),
        node: $("assistant-node").value,
        network: state.network,
        request: $("assistant-request").value,
        trust_fx: $("assistant-trust").checked,
      };
      state.lastAssistant = args;
      if (
        !(state.doc.routes || []).some(
          (route) => route.model === state.doc.settings.assistant,
        )
      )
        throw new Error(
          "Add a route for " +
            state.doc.settings.assistant +
            " in Models, then retry this request.",
        );
      await save();
      await startJob("assistant", args);
    },
    "Create draft",
  );
  const update = () => {
    $("assistant-node-label").hidden = $("assistant-kind").value !== "prompt";
    $("assistant-guide").hidden = $("assistant-kind").value !== "cden";
    $("assistant-output-label").hidden = $("assistant-kind").value !== "cden";
    $("assistant-output-limit").required = $("assistant-kind").value === "cden";
    $("assistant-output-limit").disabled = $("assistant-kind").value !== "cden";
  };
  $("assistant-guide").addEventListener(
    "toggle",
    protect(async () => {
      if ($("assistant-guide").open && !$("assistant-guide-text").textContent) {
        const result = await api("/api/authoring-guide");
        $("assistant-guide-text").textContent = result.guide;
      }
    }),
  );
  $("assistant-kind").onchange = update;
  update();
  $("assistant-model-settings").onclick = () => {
    state.lastAssistant = {
      kind: $("assistant-kind").value,
      ...($("assistant-kind").value === "cden" ? {draft_tokens: Number($("assistant-output-limit").value)} : {}),
      node: $("assistant-node").value,
      request: $("assistant-request").value,
      trust_fx: $("assistant-trust").checked,
    };
    $("dialog").close();
    showView("models");
  };
}
$("assistant-open").onclick = () =>
  openAssistant(state.view === "fx" ? "fx" : "prompt");
$("construct-network").onclick = () => openAssistant("cden");
$("job-review").onclick = () => showView("review");
$("job-retry").onclick = () =>
  openAssistant(state.lastAssistant?.kind, state.lastAssistant);
$("job-models").onclick = () => showView("models");
$("supervisor-review").onclick = protect(async () => {
  const run_id = $("review-run").value;
  if (!run_id) throw new Error("Complete an extraction run first");
  await save();
  await startJob("review", { run_id, trust_fx: $("run-trust-fx").checked });
});
function renderWorkflowSelectors() {
  const labels = {
    prompt: "Prompt improvement",
    fx: "Python function drafting",
    cden: "CDEN drafting",
    review: "Extraction review",
  };
  const mapping = state.doc.workbench_workflows || {};
  $("workflow-selectors").innerHTML = Object.entries(labels)
    .map(([kind, label]) => {
      const name = mapping[kind] || Object.keys(state.doc.nets)[0];
      const network = state.doc.nets[name];
      const outputs = network.result_nodes || ["Start"];
      const conditional = outputs.length > 1;
      return (
        "<div><label>" +
        label +
        '<select data-workflow="' +
        kind +
        '">' +
        options(Object.keys(state.doc.nets), name) +
        '</select></label><label>Return node<select data-workflow-output="' +
        kind +
        '">' +
        (conditional
          ? '<option value="__conditional__">Conditional: ' +
            esc(outputs.join(" / ")) +
            "</option>"
          : "") +
        options(
          network.nodes.map((n) => n.key),
          conditional ? "__conditional__" : outputs[0],
        ) +
        "</select></label></div>"
      );
    })
    .join("");
  $("workflow-selectors")
    .querySelectorAll("[data-workflow]")
    .forEach(
      (select) =>
        (select.onchange = (e) => {
          state.doc.workbench_workflows = {
            ...state.doc.workbench_workflows,
            [e.target.dataset.workflow]: e.target.value,
          };
          dirty();
          renderWorkflowSelectors();
        }),
    );
  $("workflow-selectors")
    .querySelectorAll("[data-workflow-output]")
    .forEach(
      (select) =>
        (select.onchange = (e) => {
          if (e.target.value === "__conditional__") return;
          const name =
            state.doc.workbench_workflows[e.target.dataset.workflowOutput];
          state.doc.nets[name].result_nodes = [e.target.value];
          dirty();
        }),
    );
}

async function renderReview() {
  if (!state.id) return;
  await refreshRuns();
  const runs = await api(base() + "/runs");
  const list = $("draft-list");
  list.replaceChildren();
  for (const run of runs.slice(0, 25)) {
    const r = await api(base() + "/runs/" + run.run_id + "/draft");
    if (!r.draft) continue;
    const container = document.createElement("div");
    container.className = "draft";
    container.innerHTML =
      "<h3>" + esc(run.operation + " · " + run.run_id) + "</h3>";
    if (r.draft.authoring_context) {
      const context = document.createElement("p");
      context.className = "small muted";
      const a = document.createElement("a");
      a.href =
        base() +
        "/runs/" +
        encodeURIComponent(run.run_id) +
        "/authoring-context";
      a.textContent = "Download drafting context";
      context.append(
        a,
        document.createTextNode(
          " · Guide and " +
            r.draft.authoring_context.examples.length +
            " examples",
        ),
      );
      container.append(context);
    }
    if (r.draft.reviewed_records !== undefined) {
      const report = document.createElement("p");
      report.className = "small muted";
      report.textContent = `Reviewed ${r.draft.reviewed_records} records · ${r.draft.flags?.length || 0} flags · ${r.draft.revisions?.length || 0} revision proposals`;
      container.append(report);
    }
    if (r.draft.failure_modes?.length) {
      const p = document.createElement("pre");
      p.textContent = fmt(r.draft.failure_modes);
      container.append(p);
    }
    if (r.draft.flags?.length) {
      const p = document.createElement("pre");
      p.textContent = fmt(r.draft.flags);
      container.append(p);
    }
    const revisions = r.draft.revisions || [r.draft];
    for (const [index, revision] of revisions.entries()) {
      const detail = document.createElement("details");
      const head = document.createElement("summary");
      head.textContent =
        revision.rationale ||
        revision.kind ||
        revision.node ||
        revision.filename ||
        revision.name ||
        "Inspect result";
      detail.append(head);
      if (revision.network) {
        const preview = document.createElement("div");
        preview.className = "draft-preview";
        const n = revision.network;
        preview.innerHTML =
          "<strong>" +
          esc(revision.name || "New CDEN") +
          "</strong><p>" +
          n.nodes.length +
          " nodes · " +
          n.edges.length +
          " connections</p><p>" +
          esc(n.nodes.map((node) => node.key).join(" → ")) +
          "</p>";
        detail.append(preview);
      }
      if (revision.prompt || revision.code) {
        const comparison = document.createElement("div");
        comparison.className = "split-cards revision-comparison";
        const old = revision.code
          ? r.baseline?.files?.[revision.filename]
          : r.baseline?.nodes?.[revision.node];
        const values = revision.code
          ? [old ?? "New module", revision.code]
          : [
              old ?? "Source node unavailable",
              { prompt: revision.prompt, persona: revision.persona },
            ];
        for (const [i, value] of values.entries()) {
          const column = document.createElement("div"),
            title = document.createElement("h3"),
            text = document.createElement("pre");
          title.textContent = i ? "Proposed" : "At source run";
          text.textContent = fmt(value);
          column.append(title, text);
          comparison.append(column);
        }
        detail.append(comparison);
      }
      const raw = document.createElement("details"),
        label = document.createElement("summary"),
        pre = document.createElement("pre");
      label.textContent = "Draft JSON";
      pre.textContent = fmt(revision);
      raw.append(label, pre);
      detail.append(raw);
      if (revision.prompt || revision.code || revision.network) {
        const button = document.createElement("button");
        button.className = "primary";
        button.textContent = revision.network
          ? "Add CDEN and open"
          : "Apply revision";
        button.disabled =
          Boolean(
            r.applications?.[r.draft.revisions ? String(index) : "artifact"],
          ) || state.appliedRuns.has(run.run_id + ":" + index);
        button.onclick = protect(async () => {
          await save();
          const applied = await api(base() + "/apply", "POST", {
            run_id: run.run_id,
            ...(r.draft.revisions ? { index } : {}),
            network: r.draft.target_network,
          });
          state.doc = applied.project;
          state.inventory = applied.fx_inventory;
          state.appliedRuns.add(run.run_id + ":" + index);
          if (applied.applied_network) {
            state.network = applied.applied_network;
            state.selection = null;
          }
          resetHistory();
          renderAll();
          if (applied.applied_network) {
            showView("graph");
            fitGraph();
          } else if (revision.code) showView("fx");
          button.disabled = true;
          toast(
            applied.applied_network
              ? "Added " + applied.applied_network
              : "Revision applied",
          );
        });
        detail.append(button);
      }
      if (revision.network || revision.prompt || revision.code)
        detail.open = true;
      container.append(detail);
    }
    list.append(container);
  }
  if (!list.children.length)
    list.innerHTML =
      '<p class="muted">No drafts. Use Draft CDEN or review a run.</p>';
}
$("refresh-review").onclick = protect(renderReview);
try {
  const projects = await projectList();
  const first = projects.length
    ? projects[0]
    : await api("/api/starter", "POST");
  await loadProject(first.id);
} catch (error) {
  toast("Could not open the workspace: " + error.message, true);
}
