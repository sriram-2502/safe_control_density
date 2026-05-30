function densityDemo() {
  const plot = document.getElementById("density-plot");
  if (!plot || !window.Plotly) return;

  const controls = {
    shape: document.getElementById("density-shape"),
    r1: document.getElementById("density-r1"),
    r2: document.getElementById("density-r2"),
    alpha: document.getElementById("density-alpha"),
    r1Label: document.getElementById("density-r1-label"),
    r2Label: document.getElementById("density-r2-label"),
    alphaLabel: document.getElementById("density-alpha-label")
  };
  const goal = [1.8, 1.4];
  const n = 65;
  const xs = Array.from({ length: n }, (_, i) => -2.4 + (4.8 * i) / (n - 1));
  const ys = Array.from({ length: n }, (_, i) => -2.4 + (4.8 * i) / (n - 1));

  function shapeParams(shape) {
    if (shape === "diamond") return { p: 1.15, scale: [1.0, 1.0] };
    if (shape === "square") return { p: 6.0, scale: [1.0, 1.0] };
    if (shape === "ellipse") return { p: 2.0, scale: [1.45, 0.72] };
    return { p: 2.0, scale: [1.0, 1.0] };
  }

  function pNormDistance(x, y, params) {
    const sx = params.scale[0];
    const sy = params.scale[1];
    const ax = Math.abs(x / sx);
    const ay = Math.abs(y / sy);
    return Math.pow(Math.pow(ax, params.p) + Math.pow(ay, params.p), 1 / params.p);
  }

  function bump(distance, r1, r2, p) {
    if (distance <= r1) return 0;
    if (distance >= r2) return 1;
    const m = (Math.pow(distance, p) - Math.pow(r1, p)) / (Math.pow(r2, p) - Math.pow(r1, p));
    const a = Math.exp(-1 / Math.max(m, 1e-6));
    const b = Math.exp(-1 / Math.max(1 - m, 1e-6));
    return a / (a + b);
  }

  function compute() {
    const r1 = Number(controls.r1.value);
    const r2Min = r1 + 0.1;
    controls.r2.min = r2Min.toFixed(2);
    const r2 = Math.max(Number(controls.r2.value), r2Min);
    const alpha = Number(controls.alpha.value);
    controls.r2.value = r2.toFixed(2);
    controls.r1Label.textContent = r1.toFixed(2);
    controls.r2Label.textContent = r2.toFixed(2);
    controls.alphaLabel.textContent = alpha.toFixed(2);

    const params = shapeParams(controls.shape.value);
    const z = ys.map((y) =>
      xs.map((x) => {
        const d = pNormDistance(x, y, params);
        const phi = bump(d, r1, r2, params.p);
        const goalDistSq = Math.pow(x - goal[0], 2) + Math.pow(y - goal[1], 2) + 0.04;
        return Math.min(phi / Math.pow(goalDistSq, alpha), 4.0);
      })
    );
    return { z };
  }

  function draw() {
    const data = compute();
    const surface = {
      type: "surface",
      x: xs,
      y: ys,
      z: data.z,
      colorscale: "Viridis",
      showscale: false,
      contours: {
        z: { show: true, usecolormap: true, highlightcolor: "#ffffff", project: { z: true } }
      }
    };
    const goalMarker = {
      type: "scatter3d",
      mode: "markers+text",
      x: [goal[0]],
      y: [goal[1]],
      z: [4.15],
      text: ["goal"],
      textposition: "top center",
      marker: { size: 5, color: "#d62728" },
      showlegend: false
    };
    const layout = {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      scene: {
        xaxis: { title: "x", range: [-2.4, 2.4], backgroundcolor: "#f7f8fb" },
        yaxis: { title: "y", range: [-2.4, 2.4], backgroundcolor: "#f7f8fb" },
        zaxis: { title: "density", range: [0, 4.2], backgroundcolor: "#f7f8fb" },
        camera: { eye: { x: 1.35, y: -1.55, z: 0.95 } }
      }
    };
    Plotly.react(plot, [surface, goalMarker], layout, {
      responsive: true,
      displaylogo: false
    });
  }

  ["input", "change"].forEach((eventName) => {
    [controls.shape, controls.r1, controls.r2, controls.alpha].forEach((el) => {
      el.addEventListener(eventName, draw);
    });
  });
  draw();
}

window.addEventListener("load", densityDemo);
