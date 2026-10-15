// Builds GPU_Medical_Imaging.pptx (25 slides) from ../results/results.json and ../results/figures.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(path.join(process.env.PPTX_SKILL, "scripts", "apply_theme.js"));

const ROOT = path.resolve(__dirname, "..");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "results.json"), "utf8"));
// every figure is converted to grayscale (presentation/.bw/) so the deck stays black and white
const BW_DIR = path.join(__dirname, ".bw");
const BW = {};
const SRC = (f) => path.join(ROOT, "results", "figures", f);
const FIG = (f) => BW[SRC(f)] || SRC(f);
const TILE = (f) => FIG(path.join("tiles", f + ".png"));
const OUT = path.join(__dirname, "GPU_Medical_Imaging_Detailed.pptx");

// ---------- theme ---------------------------------------------------------------------------
const THEME = {
  name: "Monochrome",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "000000", lt1: "FFFFFF", dk2: "000000", lt2: "F2F2F2",
    accent1: "000000", accent2: "404040", accent3: "595959", accent4: "808080",
    accent5: "262626", accent6: "666666", hlink: "000000", folHlink: "404040",
  },
};
const HEX = { navy: "000000", teal: "000000", coral: "404040", amber: "7F7F7F", slate: "A6A6A6",
              blue: "595959", mint: "7F7F7F", ink: "000000", muted: "595959", grid: "D9D9D9", ice: "F2F2F2" };

// ---------- formatting helpers --------------------------------------------------------------
const fx = (v) => (v >= 100 ? Math.round(v).toLocaleString("en-US") : v >= 10 ? v.toFixed(0) : v.toFixed(1)) + "×";
const ms = (v) => (v >= 1000 ? (v / 1000).toFixed(1) + " s" : v >= 10 ? Math.round(v) + " ms" : v.toFixed(v >= 1 ? 1 : 2) + " ms");
const f1 = (v) => v.toFixed(1);

// ---------- icons ---------------------------------------------------------------------------
const iconCache = {};
async function icon(name, color = "FFFFFF") {
  const key = name + color;
  if (!iconCache[key]) {
    const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color: "#" + color, size: 256 }));
    const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
    iconCache[key] = "image/png;base64," + buf.toString("base64");
  }
  return iconCache[key];
}

// ---------- image sizes ---------------------------------------------------------------------
const dims = {};
async function loadDims() {
  const files = ["ct_fbp.png", "mri_recon.png", "denoise.png", "segmentation.png", "render3d.png", "preprocess.png", "threshold.png"].map(SRC);
  for (const t of fs.readdirSync(SRC("tiles"))) files.push(SRC(path.join("tiles", t)));
  fs.mkdirSync(path.join(BW_DIR, "tiles"), { recursive: true });
  for (const f of files) {
    const out = path.join(BW_DIR, path.relative(SRC(""), f));
    await sharp(f).grayscale().png().toFile(out);
    BW[f] = out;
    const m = await sharp(out).metadata();
    dims[out] = m.width / m.height;
  }
}
// fit an image inside a box, keeping aspect; align 'c' centre, 'l' left, 't' top
function fitImage(slide, file, x, y, w, h, opts = {}) {
  const ar = dims[file];
  let iw = w, ih = w / ar;
  if (ih > h) { ih = h; iw = h * ar; }
  const ix = opts.align === "l" ? x : x + (w - iw) / 2;
  const iy = opts.valign === "t" ? y : y + (h - ih) / 2;
  slide.addImage({ path: file, x: ix, y: iy, w: iw, h: ih, objectName: opts.name || path.basename(file),
                   altText: opts.alt || path.basename(file), rounding: opts.rounding || false });
  return { x: ix, y: iy, w: iw, h: ih };
}

async function main() {
  await loadDims();
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5 in
  pres.title = "GPU-Accelerated Medical Image Analysis";
  pres.subject = "Case study: CT/MRI processing with GPU computing";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const W = 13.333;

  // ---------- layouts -----------------------------------------------------------------------
  const footer = (dark) => ({ text: { text: "GPU-Accelerated Medical Image Analysis  |  PDS Case Study",
    options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: dark ? "A6A6A6" : HEX.muted, fontFace: "Calibri" } } });
  pres.defineSlideMaster({
    title: "TITLE_DARK", background: { color: HEX.navy },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.55, w: 6.7, h: 2.3, fontSize: 42, bold: true,
          color: C.background1, fontFace: "Cambria", valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.05, w: 6.6, h: 1.1, fontSize: 18,
          color: "BFBFBF", fontFace: "Calibri", valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: "FFFFFF" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 34, bold: true,
          color: C.text2, fontFace: "Cambria", valign: "middle", margin: 0 }, text: "" } },
      footer(false),
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.muted, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CONTENT_DARK", background: { color: HEX.navy },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 34, bold: true,
          color: C.background1, fontFace: "Cambria", valign: "middle", margin: 0 }, text: "" } },
      footer(true),
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: "A6A6A6", align: "right" },
  });

  // ---------- drawing helpers ---------------------------------------------------------------
  const shadow = () => ({ type: "outer", color: "000000", blur: 6, offset: 1.5, angle: 90, opacity: 0.12 });
  const card = (s, x, y, w, h, fill = C.background1, name = "card", sh = true) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
      line: { color: HEX.grid, width: 0.75 }, shadow: sh ? shadow() : undefined, objectName: name });
  const txt = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontFace: "Calibri", color: C.text1, valign: "top", ...o });
  async function iconCircle(s, name, x, y, d, fill = C.accent1, fg = "FFFFFF") {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" }, objectName: "icon-bg-" + name });
    s.addImage({ data: await icon(name, fg), x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52,
                 objectName: "icon-" + name, altText: name.replace(/^Fa/, "") + " icon" });
  }
  const bullets = (items, o = {}) => items.map((t, i) => ({
    text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: o.gap ?? 6 } }));
  function stat(s, x, y, w, big, label, color = C.accent1, bigSize = 40) {
    txt(s, big, { x, y, w, h: 0.75, fontSize: bigSize, bold: true, color, fontFace: "Cambria", valign: "bottom" });
    txt(s, label, { x, y: y + 0.8, w, h: 0.75, fontSize: 13, color: HEX.muted });
  }
  const code = (s, text, x, y, w, h, size = 11) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: "1A1A1A" }, line: { type: "none" }, objectName: "code-bg" });
    txt(s, text, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: h - 0.3, fontFace: "Courier New", fontSize: size, color: "F2F2F2" });
  };
  const chartBase = (o = {}) => ({
    catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: HEX.ink,
    titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.ink, showTitle: true,
    valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
    legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.muted, legendPos: "t", ...o,
  });
  const notes = (s, t) => s.addNotes(t);
  const slide = (master = "CONTENT") => pres.addSlide({ masterName: master });

  // shortcuts to results
  const hw = R.hardware, ct = R.ct_fbp, ctv = ct.volume, mri = R.mri.cs_recon, fft = R.mri.fft3d;
  const flt = R.filter3d.scaling, f256 = flt[flt.length - 1], seg = R.segmentation, vis = R.render3d, pipe = R.pipeline;
  const thr = R.threshold.scaling, thr1 = thr[0], thr2 = thr[thr.length - 1], lesionThr = R.threshold.lesion_best_threshold;
  const ct512 = ct.scaling[2], big = fft[fft.length - 1], tr = seg.training, inf = seg.inference;
  const cores = hw.sm_count * 128;
  const transferPct = (f256.h2d_ms + f256.d2h_ms) / (f256.h2d_ms + f256.d2h_ms + f256.gpu_tiled_ms) * 100;
  const pcieGBs = (f256.voxels * 4 / 1e9) / (f256.h2d_ms / 1e3);
  const AREAS = { 1: "Medical imaging & problem definition", 2: "GPU architecture & parallel computing", 3: "Algorithms & system architecture",
                  4: "Implementation & visualisation", 5: "Performance evaluation", 6: "Discussion & conclusion" };
  const tag = (s, m) => txt(s, `MEMBER ${m}  ·  ${AREAS[m].toUpperCase()}`, { x: 0.6, y: 0.1, w: 12.1, h: 0.25, fontSize: 10, bold: true, color: HEX.muted, charSpacing: 1 });
  const th = (t, o = {}) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "000000" }, ...o } });
  const table = (s, rows, o) => s.addTable(rows, { fontSize: 12, fontFace: "Calibri", color: HEX.ink, border: { type: "solid", color: HEX.grid, pt: 0.75 },
    valign: "middle", margin: [2, 5, 2, 5], ...o });
  const box = (s, x, y, w, h, label, fill = "000000", fg = "FFFFFF", size = 12, name = "box") => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.05, fill: { color: fill }, line: { color: "000000", width: 1 }, objectName: name });
    txt(s, label, { x: x + 0.05, y, w: w - 0.1, h, fontSize: size, bold: true, color: fg, align: "center", valign: "middle" });
  };
  // negative sizes corrupt the file: draw from the other corner and flip instead
  const arrow = (s, x, y, w, h, name = "arrow") => s.addShape(pres.shapes.LINE, {
    x: w < 0 ? x + w : x, y: h < 0 ? y + h : y, w: Math.abs(w), h: Math.abs(h), flipH: w < 0, flipV: h < 0,
    line: { color: "000000", width: 1.75, endArrowType: "triangle" }, objectName: name });
  const head = (s, t, x, y, w) => txt(s, t, { x, y, w, h: 0.35, fontSize: 15, bold: true, color: C.text1 });

  // =========================================================================================
  // MEMBER 1
  // =========================================================================================
  // 1. Title
  let s = slide("TITLE_DARK");
  s.addText("GPU-Accelerated Medical Image Analysis", { placeholder: "title" });
  s.addText("Case study: processing CT and MRI scans with CUDA on an NVIDIA " + hw.gpu.replace("NVIDIA ", ""), { placeholder: "body" });
  txt(s, "Parallel & Distributed Systems  |  Case Study  |  Team of 6", { x: 0.7, y: 5.45, w: 6.6, h: 0.35, fontSize: 14, color: "D9D9D9" });
  txt(s, "October 2026", { x: 0.7, y: 5.85, w: 6.6, h: 0.35, fontSize: 14, color: "D9D9D9" });
  [["ct_recon", "CT reconstruction"], ["mri", "MRI"], ["seg_overlay", "U-Net segmentation"], ["render", "3D volume rendering"]].forEach(([t, lab], i) => {
    const x = 8.0 + (i % 2) * 2.45, y = 1.2 + Math.floor(i / 2) * 2.65;
    s.addImage({ path: TILE(t), x, y, w: 2.25, h: 2.25, objectName: "hero-" + t, altText: lab });
    txt(s, lab, { x, y: y + 2.3, w: 2.25, h: 0.28, fontSize: 11, color: "A6A6A6", align: "center" });
  });
  notes(s, "This case study shows how GPU computing speeds up CT and MRI processing. Everything in the deck was built and measured by our team: " +
    "a pipeline of CT reconstruction, MRI reconstruction, 3D denoising, threshold and deep-learning segmentation, and 3D rendering, each written for CPU and GPU, " +
    "including four hand-written CUDA kernels, benchmarked on an RTX 5060. The four images are real outputs of our code.");

  // 2. Team and roles
  s = slide(); tag(s, 1);
  s.addText("Team, roles and flow of the presentation", { placeholder: "title" });
  const roles = [
    ["1", "Medical imaging & problem", "CT vs MRI, data formats, data sources, problem statement", "Slides 1–6"],
    ["2", "GPU architecture", "Preprocessing, image operations, CPU vs GPU, thread & memory hierarchy", "Slides 7–11"],
    ["3", "Algorithms & system", "Gaussian filter, Otsu threshold, U-Net, CPU/GPU workflow, data movement", "Slides 12–17"],
    ["4", "Implementation", "Code walkthrough of every stage, output images, CPU vs GPU validation", "Slides 18–24"],
    ["5", "Performance", "Benchmark design, timings, speedup, scaling, transfer cost", "Slides 25–28"],
    ["6", "Discussion", "Challenges, real-world use, future scope, conclusion, references", "Slides 29–32"],
  ];
  roles.forEach(([n, t, d, r], i) => {
    const x = 0.6 + (i % 3) * 4.1, y = 1.45 + Math.floor(i / 3) * 2.7;
    card(s, x, y, 3.9, 2.5, C.background1, "role-" + n);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.65, h: 0.65, fill: { color: "000000" }, line: { type: "none" }, objectName: "role-num-" + n });
    txt(s, n, { x: x + 0.25, y: y + 0.25, w: 0.65, h: 0.65, fontSize: 22, bold: true, color: "FFFFFF", align: "center", valign: "middle", fontFace: "Cambria" });
    txt(s, "Member " + n, { x: x + 1.1, y: y + 0.25, w: 2.6, h: 0.3, fontSize: 12, color: HEX.muted });
    txt(s, t, { x: x + 1.1, y: y + 0.55, w: 2.65, h: 0.4, fontSize: 16, bold: true });
    txt(s, d, { x: x + 0.25, y: y + 1.1, w: 3.45, h: 0.85, fontSize: 13 });
    txt(s, r, { x: x + 0.25, y: y + 2.0, w: 3.4, h: 0.3, fontSize: 12, italic: true, color: HEX.muted });
  });
  notes(s, "Six members, six parts. The flow goes from the medical problem, to the GPU hardware, to the algorithms, to our actual code, to the measured performance, and finally the discussion.");

  // 3. How CT works
  s = slide(); tag(s, 1);
  s.addText("How CT works: X-ray projections → sinogram → image", { placeholder: "title" });
  // scanner sketch
  card(s, 0.6, 1.45, 4.2, 3.6, C.background1, "ct-sketch");
  s.addShape(pres.shapes.OVAL, { x: 1.55, y: 1.85, w: 2.3, h: 2.3, fill: { color: "FFFFFF" }, line: { color: "000000", width: 1.5, dashType: "dash" }, objectName: "gantry" });
  s.addShape(pres.shapes.OVAL, { x: 2.25, y: 2.55, w: 0.9, h: 0.9, fill: { color: "BFBFBF" }, line: { type: "none" }, objectName: "patient" });
  txt(s, "patient", { x: 2.25, y: 2.55, w: 0.9, h: 0.9, fontSize: 10, align: "center", valign: "middle" });
  box(s, 2.35, 1.6, 0.7, 0.4, "X-ray", "000000", "FFFFFF", 10, "source");
  s.addShape(pres.shapes.RECTANGLE, { x: 1.9, y: 4.05, w: 1.6, h: 0.18, fill: { color: "000000" }, line: { type: "none" }, objectName: "detector" });
  arrow(s, 2.7, 2.0, 0, 2.0, "beam");
  txt(s, "detector row (D bins)", { x: 1.3, y: 4.3, w: 2.8, h: 0.3, fontSize: 11, color: HEX.muted, align: "center" });
  txt(s, "Tube and detector rotate 180°+; one projection per angle", { x: 0.8, y: 4.6, w: 3.8, h: 0.4, fontSize: 11, color: HEX.muted, align: "center" });
  head(s, "What is measured", 5.1, 1.45, 4);
  txt(s, bullets([
    "Each detector bin records how much X-ray is absorbed along one line",
    "Radon transform: p(θ, t) = ∫∫ f(x, y) δ(x cos θ + y sin θ − t) dx dy",
    "All projections stacked = sinogram, A angles × D detectors",
    "Reconstruction inverts this (stage 1: filtered back projection)",
  ]), { x: 5.1, y: 1.85, w: 4.6, h: 2.6, fontSize: 13 });
  s.addImage({ path: TILE("sinogram"), x: 10.05, y: 1.45, w: 2.65, h: 2.65, objectName: "sinogram", altText: "sinogram" });
  txt(s, "Our sinogram: 256 angles × 256 detectors", { x: 9.95, y: 4.12, w: 2.85, h: 0.3, fontSize: 11, color: HEX.muted, align: "center" });
  head(s, "Hounsfield units (HU): how CT stores intensity", 0.6, 5.25, 7);
  table(s, [[th("Tissue"), th("Air"), th("Fat"), th("Water"), th("Soft tissue"), th("Bone")],
            ["HU value", "-1000", "-100 to -50", "0", "+20 to +80", "+400 to +3000"]],
    { x: 0.6, y: 5.65, w: 7.6, colW: [1.4, 1.0, 1.4, 1.0, 1.4, 1.4], rowH: 0.36, objectName: "hu-table" });
  card(s, 8.55, 5.25, 4.15, 1.45, C.background2, "ct-our", false);
  txt(s, [{ text: "In our data: ", options: { bold: true } }, { text: "the real head CT is stored as 16-bit integers (0–3272), 113 axial slices of 256 × 256, 2 mm apart." }],
    { x: 8.75, y: 5.38, w: 3.8, h: 1.25, fontSize: 13 });
  notes(s, "A CT scanner rotates an X-ray tube and detector around the patient. Each detector reading is the total absorption along one line, so the scanner measures line integrals, mathematically the Radon transform. " +
    "Stacking all angles gives the sinogram on the right; each point in the body traces a sine curve, hence the name. The image must be computed from it, which is our stage 1. " +
    "CT intensities are in Hounsfield units: air -1000, water 0, bone several hundred to thousands. That is why bone is bright and air is black.");

  // 4. How MRI works
  s = slide(); tag(s, 1);
  s.addText("How MRI works: magnetic signals sampled in k-space", { placeholder: "title" });
  const mriSteps = [["1", "Magnet", "B₀ field (1.5–3 T) aligns hydrogen protons"], ["2", "RF pulse", "tips the protons; they emit a radio signal as they relax"],
                    ["3", "Gradients", "make frequency and phase depend on position"], ["4", "Sample", "one line of k-space per repetition (TR)"], ["5", "Inverse FFT", "turns k-space into the image"]];
  mriSteps.forEach(([n, t, d], i) => {
    const x = 0.6 + i * 2.47;
    box(s, x, 1.45, 2.2, 0.55, n + "  " + t, "000000", "FFFFFF", 14, "mri-step-" + n);
    txt(s, d, { x, y: 2.1, w: 2.2, h: 0.75, fontSize: 12, align: "center" });
    if (i < 4) arrow(s, x + 2.2, 1.72, 0.27, 0, "mri-arr-" + i);
  });
  head(s, "The maths", 0.6, 3.05, 6);
  txt(s, bullets([
    "Signal: s(kx, ky) = ∫∫ m(x, y) · e^(−i2π(kx·x + ky·y)) dx dy, a 2D Fourier transform",
    "So the image is m(x, y) = IFFT2{ s(kx, ky) }",
    "Centre of k-space = contrast; edges = fine detail",
    "256 phase-encode lines × TR (e.g. 2 s) ≈ 8.5 min for one 2D series, so MRI is slow",
    "Measuring only 1 in 4 lines makes it 4× faster but causes aliasing; our stage 2 removes it",
  ]), { x: 0.6, y: 3.45, w: 6.9, h: 3.2, fontSize: 13 });
  s.addImage({ path: TILE("kspace"), x: 7.85, y: 3.05, w: 2.3, h: 2.3, objectName: "kspace", altText: "k-space" });
  s.addImage({ path: TILE("mri"), x: 10.4, y: 3.05, w: 2.3, h: 2.3, objectName: "mri-img", altText: "MRI image" });
  arrow(s, 10.17, 4.2, 0.2, 0, "ifft-arrow");
  txt(s, "k-space (log magnitude)", { x: 7.85, y: 5.4, w: 2.3, h: 0.3, fontSize: 11, color: HEX.muted, align: "center" });
  txt(s, "image after inverse FFT", { x: 10.4, y: 5.4, w: 2.3, h: 0.3, fontSize: 11, color: HEX.muted, align: "center" });
  txt(s, "Our data: real T1 brain MRI, 109 sagittal slices, 256 × 256, 16-bit (values 1026–5119)", { x: 7.85, y: 5.85, w: 4.85, h: 0.65, fontSize: 13, bold: true });
  notes(s, "MRI uses a strong magnet and radio pulses instead of X-rays. Gradient coils make the signal frequency depend on position, so what the scanner records is the Fourier transform of the image, called k-space. " +
    "The image is recovered with an inverse FFT, which is why FFT speed matters so much. MRI is slow because it fills k-space one line at a time; skipping lines speeds up the scan but creates aliasing, which our iterative reconstruction fixes.");

  // 5. Data formats and sources
  s = slide(); tag(s, 1);
  s.addText("Data formats and where our data comes from", { placeholder: "title" });
  table(s, [
    [th("Dataset"), th("Real?"), th("Origin"), th("Format → how we read it"), th("Size"), th("Used in")],
    ["Head CT (CThead)", "Real", "Stanford Volume Data Archive", "raw 16-bit big-endian, no header → np.fromfile('>u2')", "113 × 256², 14.8 MB", "Threshold, 3D render"],
    ["Brain MRI (MRbrain)", "Real", "Stanford Volume Data Archive", "raw 16-bit big-endian → np.fromfile('>u2')", "109 × 256², 14.3 MB", "MRI recon"],
    ["Shepp-Logan phantom", "Simulated", "Shepp & Logan 1974, generated in code", "float32 arrays (2D 128–512 px, 3D 64³–256³)", "up to 67 MB", "CT recon, denoising"],
    ["Lesion slices", "Simulated", "our generator (seeded, reproducible)", "float32 images + exact binary masks", "2,200 × 128²", "U-Net, threshold test"],
  ], { x: 0.6, y: 1.45, w: 12.1, colW: [1.9, 0.9, 2.3, 3.7, 1.6, 1.7], rowH: 0.5, fontSize: 12, objectName: "data-table" });
  head(s, "Clinical standard: DICOM", 0.6, 4.25, 6);
  txt(s, bullets([
    "One file per slice: header tags + pixel data",
    "PixelSpacing / SliceThickness give voxel size in mm (needed for resampling)",
    "HU = raw × RescaleSlope + RescaleIntercept",
    "Patient tags must be removed before research use (anonymisation)",
  ]), { x: 0.6, y: 4.65, w: 6.0, h: 2.0, fontSize: 13 });
  card(s, 7.0, 4.25, 5.7, 2.4, C.background2, "why-sim", false);
  txt(s, "Why use simulated data at all?", { x: 7.25, y: 4.4, w: 5.2, h: 0.35, fontSize: 15, bold: true });
  txt(s, bullets([
    "Accuracy needs a known correct answer: the phantom is the exact truth for CT reconstruction",
    "Lesion masks are exact, so Dice is measured precisely",
    "Large labelled clinical sets (e.g. Medical Segmentation Decathlon) need registration and GBs of download",
  ]), { x: 7.25, y: 4.8, w: 5.25, h: 1.8, fontSize: 13 });
  notes(s, "Two of our datasets are real scans from the Stanford Volume Data Archive, a public research archive. They are raw 16-bit big-endian files without headers, so we read them with numpy fromfile using the big-endian format code. " +
    "In hospitals data comes as DICOM: a header with voxel spacing and a rescale formula to Hounsfield units, plus pixel data. " +
    "We also used simulated data on purpose, because to measure accuracy you need the exact correct answer.");

  // 6. Problem statement
  s = slide(); tag(s, 1);
  s.addText("Problem statement, motivation and applications", { placeholder: "title" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.45, w: 5.6, h: 2.5, rectRadius: 0.08, fill: { color: "000000" }, line: { type: "none" }, objectName: "problem-box" });
  txt(s, "PROBLEM STATEMENT", { x: 0.9, y: 1.62, w: 5, h: 0.3, fontSize: 12, bold: true, color: "BFBFBF" });
  txt(s, "CT/MRI pipelines process 10⁷–10⁹ voxels per study and take seconds to minutes on a CPU. Can GPU parallelism make them interactive without changing the result?",
    { x: 0.9, y: 2.0, w: 5.05, h: 1.85, fontSize: 17, color: "FFFFFF", fontFace: "Cambria" });
  head(s, "Motivation", 6.6, 1.45, 6);
  txt(s, bullets([
    "Size: one 512 × 512 × 1,000 CT = 524 MB; a 4D cardiac CT ≈ 10 GB",
    "Work grows with N³: doubling resolution = 8× more voxels",
    "Time: 1.9 M neurons lost per minute of untreated stroke (Saver, 2006)",
    "Workload: radiologists have 3–4 s per image (McDonald et al., 2015)",
    `Our CPU baseline: ${ms(pipe.total_cpu_ms)} for one study through our pipeline`,
  ]), { x: 6.6, y: 1.85, w: 6.1, h: 2.2, fontSize: 13 });
  head(s, "Applications that need fast processing", 0.6, 4.2, 8);
  const apps = [["FaHeartbeat", "Stroke & trauma triage"], ["FaUserMd", "Surgical navigation"], ["FaBullseye", "Radiotherapy planning"],
                ["FaSearch", "Tumour detection & volume"], ["FaBolt", "Faster, low-dose scans"], ["FaCube", "3D visualisation"]];
  for (let i = 0; i < 6; i++) {
    const x = 0.6 + i * 2.05;
    await iconCircle(s, apps[i][0], x + 0.6, 4.7, 0.7, "000000");
    txt(s, apps[i][1], { x, y: 5.5, w: 1.9, h: 0.6, fontSize: 13, bold: true, align: "center" });
  }
  txt(s, "Objectives: build a CPU + GPU pipeline · write our own CUDA kernels · measure speedup and scaling · prove outputs are identical",
    { x: 0.6, y: 6.25, w: 12.1, h: 0.4, fontSize: 13, italic: true });
  notes(s, "Our problem statement: medical pipelines touch tens of millions to billions of voxels and are slow on a CPU. We ask whether GPU parallelism makes them interactive with identical results. " +
    "The motivation is data size, cubic growth with resolution, and time-critical care. Applications range from stroke triage to radiotherapy planning. " +
    "Our four objectives are listed at the bottom; the rest of the talk answers each one.");

  // =========================================================================================
  // MEMBER 2
  // =========================================================================================
  // 7. Preprocessing
  s = slide(); tag(s, 2);
  s.addText("Preprocessing: raw scan → GPU-ready arrays", { placeholder: "title" });
  fitImage(s, FIG("preprocess.png"), 0.6, 1.45, 7.3, 5.25, { align: "l", name: "preprocess-figure", alt: "Preprocessing steps" });
  const pre = [
    ["Type conversion", "16-bit integers → float32 (4 bytes / voxel), the type our kernels use"],
    ["Normalisation", "CT: v' = clip(v / p99.9, 0, 1). MRI: v' = clip((v − p1)/(p99.5 − p1), 0, 1). Robust to outliers"],
    ["Resampling", "CT slices 2 mm vs 1 mm pixels → trilinear interpolation to isotropic voxels (113 → 226)"],
    ["Noise model", "Low-dose CT simulated: v + N(0, 0.08²) on the phantom"],
    ["k-space", "MRI slice → FFT → 4× undersampling mask (8% centre lines + random lines)"],
    ["Layout", "Contiguous C-order arrays [slice][row][col], so neighbouring threads read neighbouring addresses"],
  ];
  pre.forEach(([t, d], i) => {
    const y = 1.45 + i * 0.88;
    txt(s, t, { x: 8.2, y, w: 4.5, h: 0.3, fontSize: 14, bold: true });
    txt(s, d, { x: 8.2, y: y + 0.3, w: 4.5, h: 0.58, fontSize: 12 });
  });
  notes(s, "Before any GPU work, raw data is converted. Integers become float32. Intensities are normalised to the range 0 to 1 using percentiles so a few extreme voxels do not dominate. " +
    "The CT has 2 mm slices but 1 mm pixels, so we resample with trilinear interpolation; panel 2 shows the squashed head before and panel 3 the corrected one. " +
    "Panel 4 shows why normalisation matters: the raw MRI uses only a tiny part of the 16-bit range. Finally the arrays are stored contiguously so GPU memory access is coalesced.");

  // 8. Image operations as parallel patterns
  s = slide(); tag(s, 2);
  s.addText("Image operations and the parallel pattern behind each", { placeholder: "title" });
  table(s, [
    [th("Operation type"), th("Where in our pipeline"), th("Parallel pattern"), th("Reads per output"), th("GPU mapping we used")],
    ["Point operation", "normalise, threshold labelling", "Map", "1", "1 thread per voxel, fully independent"],
    ["Neighbourhood (stencil)", "5×5×5 Gaussian filter", "Stencil", "125", "3D blocks + shared-memory tile with halo"],
    ["Histogram", "Otsu threshold", "Reduction with atomics", "1 (but many writers / bin)", "per-block private histogram in shared memory"],
    ["Global transform", "FFT in MRI recon", "Butterfly (log₂N stages)", "all inputs", "cuFFT library"],
    ["Gather", "FBP back-projection, resampling", "Gather (irregular reads)", "A (angles) = 256", "1 thread per pixel, read-only cache (__ldg)"],
    ["Ray marching", "MIP / volume rendering", "Map + scan along ray", "depth = 256", "1 ray per pixel, cumulative product"],
    ["Convolution layers", "U-Net", "Dense matrix multiply", "kernel × channels", "cuDNN on Tensor Cores (FP16)"],
  ], { x: 0.6, y: 1.45, w: 12.1, colW: [2.0, 2.6, 2.2, 1.9, 3.4], rowH: 0.56, fontSize: 12, objectName: "patterns-table" });
  txt(s, "Key idea: identify the pattern first. Map and gather need no communication; stencils reuse neighbours (shared memory); histograms need atomics; FFT and GEMM are best left to tuned libraries.",
    { x: 0.6, y: 6.1, w: 12.1, h: 0.6, fontSize: 13, italic: true });
  notes(s, "Every image operation in our project falls into a classic parallel pattern. Point operations are a map, the easiest case. Filtering is a stencil, where neighbouring outputs share inputs, so shared memory helps. " +
    "A histogram is a reduction where many threads write the same bin, so it needs atomic operations. The FFT is a butterfly network, best handled by cuFFT. Back-projection is a gather with irregular reads. " +
    "Rendering marches along rays, and neural network layers are matrix multiplications on Tensor Cores.");

  // 9. CPU vs GPU
  s = slide(); tag(s, 2);
  s.addText("CPU vs GPU: latency vs throughput", { placeholder: "title" });
  card(s, 0.6, 1.45, 5.0, 2.55, C.background1, "cpu-panel");
  txt(s, "CPU: few complex cores", { x: 0.85, y: 1.55, w: 4.5, h: 0.35, fontSize: 15, bold: true });
  box(s, 0.85, 2.0, 1.3, 1.1, "Control", "E6E6E6", "000000", 11, "cpu-ctl");
  for (let i = 0; i < 6; i++) box(s, 2.3 + (i % 3) * 1.05, 2.0 + Math.floor(i / 3) * 0.57, 0.95, 0.5, "Core", "000000", "FFFFFF", 10, "cpu-core-" + i);
  box(s, 0.85, 3.2, 4.5, 0.55, "Large caches (32 MB L3)", "CCCCCC", "000000", 11, "cpu-cache");
  card(s, 5.85, 1.45, 6.85, 2.55, C.background1, "gpu-panel");
  txt(s, "GPU: thousands of simple cores", { x: 6.1, y: 1.55, w: 6.3, h: 0.35, fontSize: 15, bold: true });
  for (let r = 0; r < 6; r++) for (let c = 0; c < 22; c++)
    s.addShape(pres.shapes.RECTANGLE, { x: 6.1 + c * 0.285, y: 2.0 + r * 0.28, w: 0.23, h: 0.22, fill: { color: "000000" }, line: { type: "none" }, objectName: `gc-${r}-${c}` });
  txt(s, "each square ≈ 1 group of cores; 30 SMs × 128 cores in total", { x: 6.1, y: 3.7, w: 6.3, h: 0.25, fontSize: 11, color: HEX.muted });
  table(s, [
    [th(""), th("CPU: AMD Ryzen 5 7600"), th("GPU: NVIDIA RTX 5060")],
    ["Cores", "6 cores / 12 threads", `${hw.sm_count} SMs / ${cores.toLocaleString("en-US")} CUDA cores + Tensor Cores`],
    ["Design goal", "low latency for one thread (big caches, branch prediction, out-of-order)", "high throughput for many threads (hide latency by switching warps)"],
    ["Peak FP32 (approx.)", "≈ 1 TFLOPS (AVX-512, all cores)", "≈ 19 TFLOPS"],
    ["Memory", "system DDR5, ≈ 60–80 GB/s", `${hw.gpu_mem_gb} GB GDDR7, 448 GB/s`],
    ["Threads in flight", "12", "tens of thousands (we launched up to 16.8 M per kernel)"],
  ], { x: 0.6, y: 4.2, w: 12.1, colW: [2.0, 4.6, 5.5], rowH: 0.4, fontSize: 12, objectName: "cpu-gpu-table" });
  notes(s, "A CPU spends its transistors on making one thread fast: caches, branch prediction, out-of-order execution. A GPU spends them on many simple cores and hides memory latency by switching between thousands of threads. " +
    "Our GPU has about 19 times the peak floating-point throughput and 6 times the memory bandwidth of our CPU. Medical images suit the GPU because millions of voxels need the same operation.");

  // 10. Thread hierarchy
  s = slide(); tag(s, 2);
  s.addText("CUDA thread hierarchy: grid → block → warp → thread", { placeholder: "title" });
  card(s, 0.6, 1.45, 5.6, 5.2, C.background1, "hier-panel");
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const hl = r === 1 && c === 1;
    box(s, 0.85 + c * 0.85, 1.85 + r * 0.85, 0.75, 0.75, `(${c},${r})`, hl ? "000000" : "F2F2F2", hl ? "FFFFFF" : "000000", 10, `blk-${r}-${c}`);
  }
  txt(s, "Grid of blocks (blockIdx)", { x: 0.85, y: 4.45, w: 2.5, h: 0.3, fontSize: 11, color: HEX.muted, align: "center" });
  arrow(s, 3.45, 2.95, 0.45, 0, "blk-arrow");
  for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++)
    s.addShape(pres.shapes.OVAL, { x: 4.0 + c * 0.24, y: 2.3 + r * 0.32, w: 0.17, h: 0.17, fill: { color: r === 0 ? "A6A6A6" : "000000" }, line: { type: "none" }, objectName: `t-${r}-${c}` });
  txt(s, "Block = threads (threadIdx) sharing shared memory and __syncthreads(); gray row = 1 warp of 32 threads", { x: 3.95, y: 3.65, w: 2.1, h: 1.3, fontSize: 11, color: HEX.muted });
  code(s, "// global 3D index of this thread\nint x = blockIdx.x*blockDim.x + threadIdx.x;\nint y = blockIdx.y*blockDim.y + threadIdx.y;\nint z = blockIdx.z*blockDim.z + threadIdx.z;\nif (x < W && y < H && z < D) out[(z*H+y)*W+x] = f(...);",
    0.85, 4.9, 5.1, 1.5, 10);
  head(s, "Rules that shape our kernels", 6.5, 1.45, 6);
  txt(s, bullets([
    "Warp = 32 threads executing the same instruction (SIMT). If threads in a warp take different branches, both paths run (divergence)",
    "Blocks are scheduled on SMs; several blocks per SM keep it busy (occupancy) while others wait for memory",
    "Threads in a block can share fast memory and synchronise; different blocks cannot, so each block must be independent",
    "Max 1024 threads per block; we use 256–512",
  ]), { x: 6.5, y: 1.85, w: 6.2, h: 2.7, fontSize: 13 });
  table(s, [
    [th("Our kernel"), th("Block"), th("Grid"), th("Threads"), th("Warps / block")],
    ["FBP back-projection", "16×16", "16×16×128", (ctv.gpu_threads / 1e6).toFixed(1) + " M", "8"],
    ["3D convolution", "8×8×8", "32×32×32", (f256.voxels / 1e6).toFixed(1) + " M", "16"],
    ["Histogram (grid-stride)", "256", `${thr1.hist_blocks} (= 30 SMs × 8)`, "61 k", "8"],
    ["Label (threshold)", "256", thr1.label_blocks.toLocaleString("en-US"), (thr1.voxels / 1e6).toFixed(1) + " M", "8"],
  ], { x: 6.5, y: 4.6, w: 6.2, colW: [2.0, 0.8, 1.5, 0.9, 1.0], rowH: 0.4, fontSize: 11, objectName: "launch-table" });
  notes(s, "CUDA organises threads in a hierarchy. A grid of blocks covers the whole volume; each block holds up to 1024 threads that can share memory and synchronise. " +
    "The hardware runs threads in warps of 32 in lockstep, so branches that split a warp cost time. The code shows how each thread computes which voxel it owns. " +
    "The table lists the exact launch configuration of each of our kernels. The histogram uses a grid-stride loop: a fixed 240 blocks, each thread looping over many voxels.");

  // 11. Memory hierarchy
  s = slide(); tag(s, 2);
  s.addText("GPU memory hierarchy and how we used each level", { placeholder: "title" });
  card(s, 0.6, 1.45, 5.6, 5.2, C.background1, "arch-panel");
  txt(s, hw.gpu.replace("NVIDIA ", "") + " (Blackwell)", { x: 0.85, y: 1.55, w: 5.1, h: 0.35, fontSize: 14, bold: true });
  for (let i = 0; i < 3; i++) {
    const x = 0.85 + i * 1.72;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.95, w: 1.6, h: 2.2, rectRadius: 0.05, fill: { color: "F2F2F2" }, line: { color: "000000", width: 1 }, objectName: "sm-" + i });
    txt(s, i === 2 ? `… SM ${hw.sm_count - 1}` : "SM " + i, { x, y: 2.0, w: 1.6, h: 0.3, fontSize: 11, bold: true, align: "center" });
    ["128 CUDA cores", "Tensor Cores", "Registers", "Shared mem / L1"].forEach((l, j) => box(s, x + 0.1, 2.35 + j * 0.44, 1.4, 0.37, l, ["000000", "404040", "262626", "595959"][j], "FFFFFF", 10, `sm${i}-${j}`));
  }
  box(s, 0.85, 4.3, 5.1, 0.45, "L2 cache (shared by all SMs)", "404040", "FFFFFF", 12, "l2");
  box(s, 0.85, 4.9, 5.1, 0.45, `Global memory: ${hw.gpu_mem_gb} GB GDDR7, 448 GB/s`, "000000", "FFFFFF", 12, "gmem");
  s.addShape(pres.shapes.LINE, { x: 3.4, y: 5.4, w: 0, h: 0.42, line: { color: "000000", width: 2.5, beginArrowType: "triangle", endArrowType: "triangle" }, objectName: "pcie" });
  txt(s, `PCIe: ${f1(pcieGBs)} GB/s measured`, { x: 3.55, y: 5.45, w: 2.5, h: 0.3, fontSize: 11, bold: true });
  box(s, 0.85, 5.85, 5.1, 0.45, "Host: CPU + system RAM", "A6A6A6", "000000", 12, "host");
  table(s, [
    [th("Level"), th("Scope / size"), th("Speed"), th("Used for")],
    ["Registers", "per thread", "1 cycle", "accumulators (acc) in every kernel"],
    ["Shared memory", "per block, ≤ 128 KB / SM", "≈ 20–30 cycles", "12³ halo tile (6.9 KB); 256-bin histogram (1 KB)"],
    ["Constant memory", "64 KB, broadcast", "fast if all threads read same address", "cos/sin tables (FBP), 125 filter weights"],
    ["Read-only cache", "via __ldg()", "cached global reads", "sinogram reads in back-projection"],
    ["Global memory", hw.gpu_mem_gb + " GB", "≈ 400–600 cycles", "volumes, sinograms, network weights"],
    ["PCIe link", "host ↔ device", `${f1(pcieGBs)} GB/s`, "upload scans, download results"],
  ], { x: 6.5, y: 1.45, w: 6.2, colW: [1.35, 1.55, 1.5, 1.8], rowH: 0.5, fontSize: 11, objectName: "mem-table" });
  txt(s, "Rule: load from global memory once, reuse from registers/shared memory many times, and cross PCIe as rarely as possible.",
    { x: 6.5, y: 5.8, w: 6.2, h: 0.8, fontSize: 13, italic: true });
  notes(s, `Each SM has registers, shared memory and caches; all SMs share the L2 and the 8 GB global memory. Global memory is hundreds of cycles away, and PCIe to the CPU is far slower again: we measured ${f1(pcieGBs)} GB/s. ` +
    "The table maps every level to how our code uses it: shared memory for the convolution tile and histogram, constant memory for lookup tables and filter weights, the read-only cache for the sinogram.");

  // =========================================================================================
  // MEMBER 3
  // =========================================================================================
  // 12. Gaussian filter maths
  s = slide(); tag(s, 3);
  s.addText("Gaussian filtering: the maths of 3D denoising", { placeholder: "title" });
  head(s, "Definition", 0.6, 1.45, 6);
  txt(s, bullets([
    "Weight of a neighbour at offset (i, j, k): w(i,j,k) = exp(−(i² + j² + k²) / 2σ²) / Σ exp(...)",
    "Output voxel: out(x,y,z) = Σᵢ Σⱼ Σₖ w(i,j,k) · in(x+i, y+j, z+k), with i, j, k ∈ {−2, …, 2}",
    "Our settings: σ = 1 voxel, radius 2 → 5 × 5 × 5 = 125 weights, weights sum to 1 (brightness preserved)",
    "Borders: clamp-to-edge (repeat the edge voxel), same as SciPy mode='nearest'",
    "Work: 125 multiply-adds per voxel → 2.1 billion for a 256³ volume",
    "Could be separable (3 passes × 5 taps = 15 ops), but we use the direct 125-tap form to study memory reuse",
  ]), { x: 0.6, y: 1.85, w: 6.6, h: 3.6, fontSize: 13 });
  const g = [-2, -1, 0, 1, 2].map((d) => Math.exp(-(d * d) / 2));
  const gs = g.reduce((a, b) => a + b, 0);
  const g3 = (i, j, k) => (g[i] * g[j] * g[k]) / gs ** 3;
  head(s, "Centre slice (k = 0) of the kernel, × 10⁻³", 7.5, 1.45, 5.2);
  const kr = [[th(""), ...[-2, -1, 0, 1, 2].map((d) => th("j=" + d, { align: "center" }))]];
  for (let i = 0; i < 5; i++) kr.push([{ text: "i=" + (i - 2), options: { bold: true } }, ...[0, 1, 2, 3, 4].map((j) => ({
    text: (g3(i, j, 2) * 1000).toFixed(1), options: { align: "center", bold: i === 2 && j === 2, fill: { color: i === 2 && j === 2 ? "D9D9D9" : "FFFFFF" } } }))]);
  table(s, kr, { x: 7.5, y: 1.85, w: 5.2, colW: [0.7, 0.9, 0.9, 0.9, 0.9, 0.9], rowH: 0.4, fontSize: 12, objectName: "kernel-table" });
  txt(s, `Centre weight ${(g3(2, 2, 2) * 1000).toFixed(1)} × 10⁻³; corner weight ${(g3(0, 0, 0) * 1000).toFixed(2)} × 10⁻³`, { x: 7.5, y: 4.35, w: 5.2, h: 0.3, fontSize: 12, color: HEX.muted });
  fitImage(s, FIG("denoise.png"), 0.6, 5.0, 8.0, 1.75, { align: "l", name: "denoise-figure", alt: "clean, noisy, denoised" });
  card(s, 8.95, 4.85, 3.75, 1.85, C.background2, "psnr-card", false);
  stat(s, 9.2, 4.9, 3.3, `+${f1(f256.psnr_denoised - f256.psnr_noisy)} dB`, `PSNR on 256³: ${f1(f256.psnr_noisy)} → ${f1(f256.psnr_denoised)} dB`, C.text1, 34);
  notes(s, "A Gaussian filter replaces each voxel by a weighted average of its neighbours, with weights falling off with distance according to the Gaussian formula. " +
    "With sigma 1 and radius 2 we have 125 weights; the table shows the middle slice, with the biggest weight in the centre. The weights sum to 1 so brightness is kept. " +
    "That is 125 multiply-adds per voxel, 2.1 billion for a 256-cubed volume. The filter could be split into three 1D passes, but we kept the direct 3D form because it is the textbook case for studying memory reuse on the GPU.");

  // 13. Naive vs tiled kernel
  s = slide(); tag(s, 3);
  s.addText("Naive vs shared-memory tiled Gaussian kernel", { placeholder: "title" });
  [["Naive kernel", `Each thread reads its 125 neighbours straight from global memory. Neighbouring threads read mostly the same voxels again and again.`, `125 global reads / voxel  ·  ${ms(f256.gpu_naive_ms)}`],
   ["Tiled kernel", "A block of 8×8×8 = 512 threads first copies a 12×12×12 tile (8 + 2×2 halo) into shared memory, waits at __syncthreads(), then reads all neighbours from the tile.", `1728 / 512 = 3.4 global reads / voxel  ·  ${ms(f256.gpu_tiled_ms)}`]].forEach(([h, d, k], i) => {
    const x = 0.6, y = 1.45 + i * 1.6;
    card(s, x, y, 6.2, 1.45, C.background1, "kpanel-" + i);
    txt(s, h, { x: x + 0.2, y: y + 0.1, w: 5.8, h: 0.3, fontSize: 15, bold: true });
    txt(s, d, { x: x + 0.2, y: y + 0.42, w: 5.8, h: 0.6, fontSize: 12 });
    txt(s, k, { x: x + 0.2, y: y + 1.05, w: 5.8, h: 0.3, fontSize: 13, bold: true });
  });
  head(s, "Why tiling wins", 0.6, 4.75, 6);
  txt(s, bullets([
    `${Math.round(125 / 3.375)}× fewer global memory reads per voxel`,
    "Shared memory is on-chip: ~20–30 cycles vs ~400–600",
    "Filter weights in __constant__ memory: all threads read the same weight at once (broadcast)",
    `Result: ${f256.tiled_vs_naive.toFixed(2)}× faster than naive, ${f1(f256.gvoxels_per_s_tiled)} billion voxels/s`,
  ]), { x: 0.6, y: 5.15, w: 6.2, h: 1.55, fontSize: 13 });
  code(s, "__global__ void conv_tiled(const float* in, float* out,\n                           int D, int H, int W) {\n  __shared__ float tile[12][12][12];      // 6.9 KB\n  int tid = (tz*8 + ty)*8 + tx;            // 0..511\n  // 1) cooperative load: 512 threads copy 1728 values\n  for (int i = tid; i < 12*12*12; i += 512) {\n    int lz = i/144, ly = (i/12)%12, lx = i%12;\n    tile[lz][ly][lx] = in[clamp(z0+lz)][clamp(y0+ly)]\n                         [clamp(x0+lx)];\n  }\n  __syncthreads();                       // tile complete\n  // 2) compute from shared memory only\n  float acc = 0.f;\n  #pragma unroll\n  for (dz..5) for (dy..5) for (dx..5)\n    acc += c_w[dz][dy][dx] *\n           tile[tz+dz][ty+dy][tx+dx];\n  out[(z*H + y)*W + x] = acc;\n}",
    7.1, 1.45, 5.6, 5.25, 10.5);
  notes(s, "The naive kernel is correct but wasteful: neighbouring threads read almost the same 125 voxels from slow global memory. " +
    "The tiled kernel first has all 512 threads of a block copy a 12-cubed tile, the 8-cubed output region plus a 2-voxel halo, into shared memory. After a barrier, every thread computes its sum from the tile. " +
    `Global reads drop from 125 to 3.4 per voxel, and the kernel becomes ${f256.tiled_vs_naive.toFixed(2)} times faster. Weights sit in constant memory, which broadcasts one value to all threads of a warp.`);

  // 14. Threshold segmentation
  s = slide(); tag(s, 3);
  s.addText("Threshold segmentation with Otsu's method on the GPU", { placeholder: "title" });
  fitImage(s, FIG("threshold.png"), 0.6, 1.4, 12.1, 2.3, { name: "threshold-figure", alt: "CT, histogram with Otsu thresholds, labels" });
  head(s, "Algorithm", 0.6, 3.85, 4);
  txt(s, bullets([
    "Build a 256-bin intensity histogram",
    "Otsu: choose t maximising between-class variance σ²_B(t) = ω₀(t)·ω₁(t)·(μ₀(t) − μ₁(t))²",
    `Run Otsu twice on the same histogram: t1 = ${thr1.t1.toFixed(2)} (air | tissue), t2 = ${thr1.t2.toFixed(2)} (tissue | bone)`,
    "Label every voxel: 0 air, 1 soft tissue, 2 bone",
  ]), { x: 0.6, y: 4.25, w: 4.4, h: 2.45, fontSize: 12.5 });
  head(s, "GPU histogram: atomics problem", 5.25, 3.85, 3.6);
  txt(s, bullets([
    "Millions of threads add to only 256 bins → collisions",
    `Global atomics: ${ms(thr2.hist_global_ms)} on 59 M voxels`,
    `Fix: each block counts in its own shared-memory histogram, then adds 256 totals: ${ms(thr2.hist_shared_ms)}`,
    `${f1(thr2.shared_vs_global)}× faster; result identical`,
  ]), { x: 5.25, y: 4.25, w: 3.7, h: 2.45, fontSize: 12.5 });
  card(s, 9.2, 3.85, 3.5, 2.85, C.background2, "thr-stats", false);
  stat(s, 9.4, 3.9, 3.1, fx(thr2.speedup_compute), `GPU vs CPU, 59 M voxels (${fx(thr2.speedup_total)} incl. transfers)`, C.text1, 34);
  stat(s, 9.4, 5.3, 3.1, String(thr2.label_mismatch_voxels), "voxels differ between CPU and GPU labels", C.text1, 34);
  notes(s, "Threshold segmentation is the classic baseline. First we build a 256-bin histogram of intensities. Otsu's method picks the threshold that best separates the histogram into two classes, by maximising the between-class variance. " +
    "We apply it twice to the same histogram: once to split air from the head, then again above that threshold to split soft tissue from bone. " +
    "The interesting GPU part is the histogram: millions of threads update only 256 counters, so global atomic adds collide. Giving each block a private histogram in shared memory and merging at the end is " +
    `${f1(thr2.shared_vs_global)} times faster. The labelling itself is a simple map, one thread per voxel.`);

  // 15. U-Net
  s = slide(); tag(s, 3);
  s.addText("Deep-learning segmentation: the U-Net", { placeholder: "title" });
  card(s, 0.6, 1.45, 6.2, 4.0, C.background1, "unet-panel");
  txt(s, `U-Net · ${f1(seg.params_millions)} M parameters · input 1 × 128 × 128`, { x: 0.85, y: 1.55, w: 5.8, h: 0.3, fontSize: 13, bold: true });
  [32, 64, 128, 256].forEach((c, i) => {
    const h = Math.max(2.0 - i * 0.42, 0.55), y = 2.2 + i * 0.55;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.95 + i * 0.62, y, w: 0.42, h, fill: { color: "000000" }, line: { type: "none" }, objectName: "enc-" + i });
    txt(s, String(c), { x: 0.85 + i * 0.62, y: y - 0.28, w: 0.62, h: 0.25, fontSize: 10, align: "center", color: HEX.muted });
    s.addShape(pres.shapes.RECTANGLE, { x: 5.8 - i * 0.62, y, w: 0.42, h, fill: { color: "7F7F7F" }, line: { type: "none" }, objectName: "dec-" + i });
    txt(s, String(c), { x: 5.7 - i * 0.62, y: y - 0.28, w: 0.62, h: 0.25, fontSize: 10, align: "center", color: HEX.muted });
    s.addShape(pres.shapes.LINE, { x: 1.4 + i * 0.62, y: y + 0.2, w: 4.35 - i * 1.24, h: 0, line: { color: "7F7F7F", width: 1, dashType: "dash", endArrowType: "triangle" }, objectName: "skip-" + i });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 3.3, y: 4.35, w: 0.55, h: 0.4, fill: { color: "404040" }, line: { type: "none" }, objectName: "bottleneck" });
  txt(s, "black = encoder (2× conv3×3 + BN + ReLU, max-pool) · dark = 512-ch bottleneck · gray = decoder (up-conv + concat) · dashed = skip connections",
    { x: 0.85, y: 5.0, w: 5.8, h: 0.4, fontSize: 10.5, color: HEX.muted });
  table(s, [
    [th("Training setting"), th("Value")],
    ["Data", "2,000 train / 200 test slices, 128²"], ["Loss", "binary cross-entropy + Dice loss"],
    ["Optimiser", "Adam, lr 10⁻³, cosine schedule"], ["Batch / epochs", `${tr.batch} / ${tr.epochs} (${tr.iters_per_epoch} steps per epoch)`],
    ["Precision", "FP16 autocast + GradScaler (Tensor Cores)"], ["Memory layout", "channels_last (NHWC) for cuDNN"],
    ["Output", "sigmoid > 0.5 → lesion mask"],
  ], { x: 7.1, y: 1.45, w: 5.6, colW: [1.8, 3.8], rowH: 0.42, fontSize: 12, objectName: "train-table" });
  card(s, 0.6, 5.65, 12.1, 1.05, C.background2, "why-unet", false);
  txt(s, [{ text: "Why not just threshold? ", options: { bold: true } },
    { text: `Lesions are both brighter and darker than the organ they sit in, so no single threshold works: even the best possible threshold (chosen using the answers) reaches Dice ${lesionThr.dice.toFixed(2)}. The U-Net reaches ${inf.test_dice.toFixed(3)}.` }],
    { x: 0.85, y: 5.75, w: 11.6, h: 0.85, fontSize: 13, valign: "middle" });
  notes(s, `The U-Net is an encoder-decoder network. The encoder halves the image size four times while increasing channels from 32 to 512, capturing context; the decoder upsamples back and skip connections bring back fine detail. ` +
    "We train with cross-entropy plus Dice loss, Adam, and FP16 mixed precision so the Tensor Cores are used. " +
    `Why do we need it? Our lesions can be brighter or darker than their surroundings, so a threshold fails: the best possible one scores Dice ${lesionThr.dice.toFixed(2)}, the U-Net ${inf.test_dice.toFixed(3)}.`);

  // 16. CPU/GPU workflow
  s = slide(); tag(s, 3);
  s.addText("System architecture: CPU/GPU workflow of one stage", { placeholder: "title" });
  const yH = 1.6, yP = 2.6, yD = 3.6, bh = 0.7;
  const lanes = [["HOST (CPU)", 1.45, 1.0], ["PCIe", 2.45, 1.0], ["DEVICE (GPU)", 3.45, 1.0]];
  lanes.forEach(([l, y, h], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 12.1, h, fill: { color: i === 1 ? "F2F2F2" : "FFFFFF" }, line: { color: "BFBFBF", width: 0.75 }, objectName: "lane-" + i });
    txt(s, l, { x: 0.7, y, w: 1.3, h, fontSize: 12, bold: true, valign: "middle" });
  });
  const flowSteps = [
    [2.1, yH, "1. Load + preprocess (NumPy, float32)"], [4.0, yH, "2. cudaMalloc (device buffers)"],
    [4.0, yP, "3. cudaMemcpy H→D"], [5.9, yD, "4. Launch kernel <<<grid, block>>>"], [7.8, yD, "5. Kernel runs on all SMs in parallel"],
    [9.7, yD, "6. Next kernel uses data still on GPU"], [9.7, yP, "7. cudaMemcpy D→H"], [10.9, yH, "8. Save / display"]];
  flowSteps.forEach(([x, y, l], i) => box(s, x, y, 1.7, bh, l, i === 2 || i === 6 ? "595959" : "000000", "FFFFFF", 10.5, "flow-" + i));
  [[3.8, yH + bh / 2, 0.2, 0], [4.85, yH + bh, 0, yP - yH - bh], [5.7, yP + bh / 2, 0.2, yD - yP], [7.6, yD + bh / 2, 0.2, 0], [9.5, yD + bh / 2, 0.2, 0],
   [10.55, yD, 0, -(yD - yP - bh)], [11.2, yP, 0, -(yP - yH - bh)]].forEach(([x, y, w, h], i) => arrow(s, x, y, w, h, "flow-arrow-" + i));
  table(s, [
    [th("Stage"), th("On CPU"), th("Copied over PCIe"), th("On GPU")],
    ["CT reconstruction", "phantom, sinogram (radon)", `sinograms ${(ctv.slices * 256 * 256 * 4 / 1e6).toFixed(0)} MB in, volume out`, "cuFFT ramp filter → FBP kernel"],
    ["MRI reconstruction", "load + normalise", "10 slices in, images out", "200 × (TV step → FFT → data consistency → IFFT)"],
    ["Denoise / threshold", "load + normalise", "volume in, result out", "conv kernel / histogram → label kernels"],
    ["U-Net / rendering", "data generation, plots", "batches / volume in, masks / frames out", "cuDNN convolutions / grid_sample + ray casting"],
  ], { x: 0.6, y: 4.7, w: 12.1, colW: [2.0, 2.6, 3.4, 4.1], rowH: 0.38, fontSize: 11.5, objectName: "where-table" });
  notes(s, "Every GPU stage follows the same eight steps. The CPU loads and preprocesses the data, allocates GPU memory and copies the input across PCIe. It then launches kernels; the kernel call returns immediately and the GPU runs it on all SMs. " +
    "Several kernels can run back-to-back on data that stays on the GPU, for example FFT then filter then back-projection. Finally the result is copied back. " +
    "The copies in gray are the expensive part we come back to in the performance section. The table shows what lives where for each stage.");

  // 17. Software stack & data movement
  s = slide(); tag(s, 3);
  s.addText("Software stack and data movement", { placeholder: "title" });
  const stk = [["Application", "our Python pipeline: 7 scripts, 1 per stage + benchmark"], ["Framework", "PyTorch " + hw.torch.split("+")[0] + " (tensors on device='cuda')"],
               ["GPU libraries", "cuFFT (FFT) · cuDNN (convolutions) · cuBLAS (matrices)"], ["Our kernels", "fbp.cu · conv3d.cu · threshold.cu (CUDA C++, nvcc -O3 -arch=sm_120)"],
               ["Runtime / driver", "CUDA 13.4 toolkit, runtime API, NVIDIA driver"], ["Hardware", hw.gpu + " · " + hw.sm_count + " SMs · " + hw.gpu_mem_gb + " GB"]];
  stk.forEach(([l, d], i) => {
    const y = 1.45 + i * 0.6;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 6.1, h: 0.52, fill: { color: ["000000", "262626", "404040", "595959", "7F7F7F", "A6A6A6"][i] }, line: { type: "none" }, objectName: "stk-" + i });
    txt(s, l, { x: 0.75, y, w: 1.6, h: 0.52, fontSize: 12, bold: true, color: i < 5 ? "FFFFFF" : "000000", valign: "middle" });
    txt(s, d, { x: 2.35, y, w: 4.3, h: 0.52, fontSize: 11, color: i < 5 ? "FFFFFF" : "000000", valign: "middle" });
  });
  txt(s, "Python calls our kernels as compiled executables: data is written to a binary file, the kernel reads it, times itself with cudaEvent and prints JSON timings.",
    { x: 0.6, y: 5.15, w: 6.1, h: 0.75, fontSize: 12, italic: true });
  head(s, "How much data crosses PCIe", 7.0, 1.45, 5.7);
  table(s, [
    [th("Transfer"), th("Size"), th("Time")],
    ["CT sinograms 128 × 256 × 256 float32", `${(ctv.slices * 256 * 256 * 4 / 1e6).toFixed(1)} MB`, ms(ctv.gpu_h2d_ms)],
    ["Denoise volume 256³ float32, upload", `${(f256.voxels * 4 / 1e6).toFixed(1)} MB`, ms(f256.h2d_ms)],
    ["Denoise volume 256³ float32, download", `${(f256.voxels * 4 / 1e6).toFixed(1)} MB`, ms(f256.d2h_ms)],
    ["Threshold labels 59 M × uint8", `${(thr2.voxels / 1e6).toFixed(1)} MB`, ms(thr2.d2h_ms)],
  ], { x: 7.0, y: 1.85, w: 5.7, colW: [3.3, 1.2, 1.2], rowH: 0.42, fontSize: 11.5, objectName: "pcie-table" });
  card(s, 7.0, 4.15, 5.7, 1.75, C.background2, "pcie-card", false);
  stat(s, 7.25, 4.2, 5.2, `${Math.round(transferPct)}%`, `of the denoising stage's GPU time is spent copying data (compute is only ${ms(f256.gpu_tiled_ms)})`, C.text1, 34);
  txt(s, "Design choice: labels are sent back as uint8 (1 byte) instead of float32, so the download is 4× smaller.", { x: 0.6, y: 6.05, w: 12.1, h: 0.5, fontSize: 13 });
  notes(s, "This is the full software stack. At the top our Python scripts; PyTorch gives GPU tensors and calls NVIDIA's libraries; our own kernels are compiled with nvcc for the Blackwell architecture, sm_120. " +
    `The right side shows the actual data crossing the PCIe bus. At about ${f1(pcieGBs)} GB/s, copying a 67 MB volume takes longer than filtering it on the GPU, so ${Math.round(transferPct)}% of the denoising time is transfer. ` +
    "That is why data movement is a central design concern, and why we return labels as 1-byte integers.");

  // =========================================================================================
  // MEMBER 4
  // =========================================================================================
  // 18. Code structure
  s = slide(); tag(s, 4);
  s.addText("Implementation: code structure and how to run it", { placeholder: "title" });
  code(s, "pds_present/\n├─ src/kernels/\n│  ├─ fbp.cu         CT back-projection\n│  ├─ conv3d.cu      naive + tiled filter\n│  └─ threshold.cu   histogram + labels\n├─ src/\n│  ├─ common.py      timing, nvcc build, PSNR\n│  ├─ phantoms.py    2D/3D Shepp-Logan\n│  ├─ ct_recon.py    stage 1\n│  ├─ mri_recon.py   stage 2\n│  ├─ filtering.py   stage 3\n│  ├─ threshold_seg.py  stage 4a\n│  ├─ segmentation.py   stage 4b (U-Net)\n│  ├─ visualize.py   stage 5\n│  └─ benchmark.py   runs all, summary\n├─ data/   CThead.npy, MRbrain.npy\n└─ results/results.json + figures/",
    0.6, 1.45, 5.6, 5.25, 11);
  table(s, [
    [th("Script"), th("Key functions"), th("GPU part")],
    ["common.py", "timeit(), build_kernel(), run_kernel(), psnr()", "compiles .cu with nvcc + MSVC on first use"],
    ["ct_recon.py", "sinogram(), filter_gpu(), fbp_gpu()", "torch.fft + fbp.cu"],
    ["mri_recon.py", "make_mask(), fft2c(), tv_grad(), cs_recon()", "same PyTorch code on 'cpu' or 'cuda'"],
    ["filtering.py", "gauss_kernel(), main()", "conv3d.cu, F.conv3d (cuDNN)"],
    ["threshold_seg.py", "otsu(), cpu_pipeline(), gpu_pipeline()", "threshold.cu"],
    ["segmentation.py", "make_dataset(), UNet, train_step()", "autocast FP16, channels_last"],
    ["visualize.py", "rotate(), mip(), dvr()", "grid_sample + cumprod on GPU"],
  ], { x: 6.5, y: 1.45, w: 6.2, colW: [1.55, 2.5, 2.15], rowH: 0.48, fontSize: 11, objectName: "scripts-table" });
  code(s, "pip install -r requirements.txt\npython src/benchmark.py      # all stages\npython src/benchmark.py --summary", 6.5, 5.55, 6.2, 1.15, 11);
  notes(s, "The project has three CUDA files and one Python script per stage. common.py holds shared tools: timing with GPU synchronisation, compiling the CUDA files with nvcc, running them and computing PSNR. " +
    "Each stage script has a CPU version and a GPU version of the same algorithm and saves its timings and accuracy to results.json. benchmark.py runs everything and prints the summary table.");

  // 19. CT FBP implementation
  s = slide(); tag(s, 4);
  s.addText("Stage 1 code: filtered back projection", { placeholder: "title" });
  head(s, "Step 1: ramp filter with cuFFT (Python)", 0.6, 1.45, 6);
  code(s, "def filter_gpu(sino_t):              # [S, A, D] on GPU\n    D = sino_t.shape[-1]\n    size, h = ramp(D)                 # h = 2|f|, padded\n    F = torch.fft.fft(sino_t, n=size, dim=-1) * h\n    return torch.fft.ifft(F, dim=-1).real[..., :D]",
    0.6, 1.85, 6.1, 1.6, 10.5);
  head(s, "Step 2: back-projection kernel (CUDA)", 0.6, 3.6, 6);
  code(s, "__constant__ float c_cos[2048], c_sin[2048];\n__global__ void backproject(const float* sino, float* img,\n                            int A, int D, int N) {\n  int col = blockIdx.x*blockDim.x + threadIdx.x;\n  int row = blockIdx.y*blockDim.y + threadIdx.y;\n  int s = blockIdx.z;  float acc = 0.f;\n  for (int a = 0; a < A; ++a) {\n    float t = y*c_cos[a] - x*c_sin[a] + D/2;\n    int i = floorf(t);  float w = t - i;\n    acc += (1-w)*__ldg(p+i) + w*__ldg(p+i+1);\n  }\n  img[s][row][col] = acc * PI / (2*A);  }",
    0.6, 4.0, 6.1, 2.7, 10);
  head(s, "What happens", 7.0, 1.45, 5.7);
  txt(s, bullets([
    "Ramp filter |f| sharpens each projection; without it the image is blurred",
    "Each thread owns one pixel (x, y) of one slice s and loops over all 256 angles",
    "t = position where the pixel projects onto the detector; linear interpolation between the 2 nearest bins",
    "cos/sin tables in constant memory (computed once on the host)",
    "__ldg reads the sinogram through the read-only cache",
    "Grid 16 × 16 × 128 blocks of 16 × 16 → every pixel of every slice in one launch",
  ]), { x: 7.0, y: 1.85, w: 5.7, h: 3.0, fontSize: 12.5 });
  fitImage(s, FIG("ct_fbp.png"), 7.0, 4.95, 5.7, 1.75, { name: "ct-fbp-figure", alt: "phantom, sinogram, CPU and GPU reconstructions" });
  notes(s, "Stage one in code. First the ramp filter: FFT each projection, multiply by the absolute frequency, inverse FFT, all batched on the GPU with cuFFT through PyTorch. " +
    "Then our CUDA kernel: every thread reconstructs one pixel of one slice. It loops over the 256 angles, computes where its pixel lands on the detector, interpolates the filtered value and adds it up. " +
    `cos and sin values come from constant memory. One launch of ${(ctv.gpu_threads / 1e6).toFixed(1)} million threads reconstructs the whole 128-slice volume in ${ms(ctv.gpu_kernel_ms)}.`);

  // 20. MRI implementation
  s = slide(); tag(s, 4);
  s.addText("Stage 2 code: iterative MRI reconstruction", { placeholder: "title" });
  code(s, "def make_mask(n, accel=4, centre_frac=0.08):\n    keep[n//2 - c//2 : n//2 + c//2] = True   # centre lines\n    keep[random other lines] = True          # total n/4\n\ndef cs_recon(y, mask, iters=200, lam=0.005):\n    x = ifft2c(y).real                # zero-filled start\n    for _ in range(iters):\n        x = x - lam * tv_grad(x)      # 1) smooth noise/aliasing\n        k = fft2c(x)                  # 2) to k-space\n        k = k*(1-mask) + y*mask       # 3) keep measured data\n        x = ifft2c(k).real            # 4) back to image\n    return x",
    0.6, 1.45, 6.4, 3.6, 10.5);
  head(s, "What each step does", 7.3, 1.45, 5.4);
  txt(s, bullets([
    "Mask keeps 25% of k-space lines: the centre (contrast) + random lines",
    "tv_grad: gradient of total variation, which penalises noise and streaks but keeps edges",
    "Data consistency: measured k-space values are put back exactly every iteration",
    `${mri.iterations} iterations × 2 FFTs = ${mri.ffts_per_slice} FFTs per slice`,
    "Same function runs on CPU or GPU: tensors on device='cuda' make every FFT a cuFFT call",
  ]), { x: 7.3, y: 1.85, w: 5.4, h: 3.2, fontSize: 12.5 });
  fitImage(s, FIG("mri_recon.png"), 0.6, 5.2, 12.1, 1.5, { name: "mri-figure", alt: "MRI reconstruction steps" });
  notes(s, "This is compressed sensing in a few lines. We start from the zero-filled image, then repeat 200 times: take a small step that reduces total variation, which removes noise and aliasing streaks while keeping edges; " +
    "go to k-space; overwrite the lines we actually measured, so the image always agrees with the scanner data; and come back. " +
    `That is 400 FFTs per slice. The same code runs on the CPU or the GPU just by where the tensors live; on the GPU every FFT goes to cuFFT, giving ${fx(mri.speedup)}.`);

  // 21. U-Net implementation
  s = slide(); tag(s, 4);
  s.addText("Stage 4 code: training and running the U-Net on the GPU", { placeholder: "title" });
  code(s, "model = UNet().to('cuda').to(memory_format=torch.channels_last)\nopt = torch.optim.Adam(model.parameters(), 1e-3)\nscaler = torch.amp.GradScaler('cuda')\nX, Y = torch.from_numpy(xtr).cuda(), torch.from_numpy(ytr).cuda()\n\ndef train_step(x, y):\n    opt.zero_grad(set_to_none=True)\n    with torch.autocast('cuda', dtype=torch.float16):\n        out = model(x)\n        loss = bce(out, y) + dice_loss(out.float(), y)\n    scaler.scale(loss).backward()   # avoid FP16 underflow\n    scaler.step(opt); scaler.update()\n\n# inference\nwith torch.no_grad(), torch.autocast('cuda', torch.float16):\n    mask = model(X_test) > 0",
    0.6, 1.45, 6.6, 4.3, 10.5);
  head(s, "Implementation details", 7.5, 1.45, 5.2);
  txt(s, bullets([
    "Whole dataset copied to the GPU once (2,000 × 128² floats ≈ 131 MB), so no PCIe traffic per batch",
    "autocast runs convolutions in FP16 on Tensor Cores; loss stays FP32",
    "GradScaler multiplies the loss so small FP16 gradients do not become zero",
    "channels_last memory layout = cuDNN's fastest format",
    "Lesion generator: body ellipse + 2–3 organs + 1–3 lesions (±0.12–0.2 intensity), blur σ = 1, noise σ = 0.04",
  ]), { x: 7.5, y: 1.85, w: 5.2, h: 3.9, fontSize: 12.5 });
  [["CPU epoch", `${f1(tr.cpu_epoch_s)} s`], ["GPU epoch", `${f1(tr.gpu_epoch_s)} s`], ["Total GPU training", `${Math.round(tr.gpu_total_train_s)} s`], ["Test Dice", inf.test_dice.toFixed(3)]].forEach(([l, v], i) => {
    const x = 0.6 + i * 3.08;
    card(s, x, 5.95, 2.88, 0.8, C.background2, "unet-num-" + i, false);
    txt(s, v, { x: x + 0.2, y: 5.98, w: 1.5, h: 0.75, fontSize: 22, bold: true, fontFace: "Cambria", valign: "middle" });
    txt(s, l, { x: x + 1.6, y: 5.98, w: 1.2, h: 0.75, fontSize: 12, color: HEX.muted, valign: "middle" });
  });
  notes(s, "The U-Net is implemented in PyTorch. The key GPU details: we copy the whole training set to GPU memory once so batches never cross PCIe; autocast runs the convolutions in half precision on Tensor Cores; " +
    "the gradient scaler prevents tiny half-precision gradients from rounding to zero; and channels_last is the layout cuDNN prefers. " +
    `One epoch takes ${f1(tr.cpu_epoch_s)} s on the CPU and ${f1(tr.gpu_epoch_s)} s on the GPU.`);

  // 22. Rendering implementation
  s = slide(); tag(s, 4);
  s.addText("Stage 5 code: 3D visualisation by ray casting", { placeholder: "title" });
  fitImage(s, FIG("render3d.png"), 0.6, 1.45, 6.6, 3.1, { align: "l", valign: "t", name: "render-figure", alt: "MIP and volume rendering views" });
  head(s, "Equations", 7.5, 1.45, 5.2);
  txt(s, bullets([
    "MIP (top row): pixel = max over the ray of v(sample)",
    "Volume rendering (bottom): C = Σᵢ cᵢ · αᵢ · Πⱼ<ᵢ (1 − αⱼ), front-to-back",
    "Transfer function: α = 0.25 · clamp((v − 0.25)/0.35, 0, 1) + 0.004 · [v > 0.08] → bone opaque, soft tissue faint",
    "Each view: rotate the volume with trilinear grid_sample, then march along depth",
  ]), { x: 7.5, y: 1.85, w: 5.2, h: 2.7, fontSize: 12.5 });
  code(s, "def dvr(v, opacity=0.25):          # v: [1,1,D,H,W] on GPU\n    d = v.clamp(0, 1)\n    alpha = opacity*((d-0.25)/0.35).clamp(0,1) + 0.004*(d>0.08)\n    trans = torch.cumprod(torch.cat([ones, 1-alpha[:,:,:-1]], 2), 2)\n    return (d * alpha * trans).sum(2)   # one value per ray\n\nfor angle in range(0, 360, 10):          # 36 views\n    frame = dvr(rotate(volume, angle))",
    0.6, 4.75, 7.3, 1.95, 10.5);
  card(s, 8.2, 4.75, 4.5, 1.95, C.background2, "render-card", false);
  stat(s, 8.45, 4.8, 4.0, `${Math.round(vis.dvr_gpu_fps)} fps`, `on GPU vs ${f1(vis.dvr_cpu_fps)} fps on CPU (${fx(vis.dvr_speedup)}), real head CT ${vis.volume_shape.join(" × ")}`, C.text1, 34);
  notes(s, "For 3D views, each screen pixel shoots a ray through the volume. Maximum intensity projection keeps the brightest sample. Volume rendering blends samples front to back: each sample adds its brightness times its opacity times how much light still gets through. " +
    "The transfer function makes bone opaque and soft tissue almost transparent, which reveals the skull. The cumulative product computes the transmittance for all rays at once, so every ray is processed in parallel.");

  // 23. Results gallery
  s = slide(); tag(s, 4);
  s.addText("Outputs: original, filtered and segmented images", { placeholder: "title" });
  const gal = [["noisy", "Original: noisy CT (σ = 0.08)"], ["denoised", "Filtered: Gaussian 5³ on GPU"], ["ct_norm", "Original: real head CT"], ["ct_labels", "Segmented: Otsu air / tissue / bone"],
               ["lesion_input", "Original: lesion slice"], ["seg_overlay", "Segmented: U-Net (dashed) vs truth"], ["mri", "MRI reconstructed on GPU"], ["render", "3D volume rendering"]];
  gal.forEach(([t, l], i) => {
    const x = 0.6 + (i % 4) * 3.08, y = 1.45 + Math.floor(i / 4) * 2.68;
    s.addImage({ path: TILE(t), x: x + 0.39, y, w: 2.1, h: 2.1, objectName: "gal-" + t, altText: l });
    txt(s, l, { x, y: y + 2.15, w: 2.88, h: 0.35, fontSize: 12, align: "center", bold: i % 2 === 1 });
  });
  notes(s, "These are the actual outputs of our pipeline, in pairs: original then processed. Noisy CT and its filtered version; the real head CT and its threshold labels; a lesion slice and the U-Net outline compared with the truth; and the MRI and 3D rendering results.");

  // 24. Validation
  s = slide(); tag(s, 4);
  s.addText("Validation: GPU and CPU give the same answer", { placeholder: "title" });
  table(s, [
    [th("Stage"), th("Check"), th("Max |CPU − GPU|"), th("Quality vs ground truth")],
    ["1  CT reconstruction (128 slices)", "same algorithm, compare volumes", ctv.max_abs_diff_cpu_gpu.toExponential(1), `PSNR ${f1(ctv.psnr_gpu)} dB vs phantom`],
    ["2  MRI iterative recon", "same PyTorch code on both devices", mri.max_abs_diff_cpu_gpu.toExponential(1), `PSNR ${f1(mri.psnr_cs_gpu)} dB (zero-filled ${f1(mri.psnr_zero_filled)})`],
    ["3  Gaussian filter 256³", "SciPy vs our tiled kernel", f256.max_abs_diff_cpu_gpu.toExponential(1), `PSNR ${f1(f256.psnr_noisy)} → ${f1(f256.psnr_denoised)} dB`],
    ["4a Otsu threshold 59 M voxels", "histogram + labels", `${thr2.label_mismatch_voxels} voxels differ`, `histograms identical: ${thr2.hist_equal_cpu_gpu ? "yes" : "no"}`],
    ["4b U-Net", "same trained weights", "—", `Dice ${inf.test_dice.toFixed(3)} on 200 test slices`],
    ["5  Volume rendering", "same PyTorch code", vis.max_abs_diff_dvr.toExponential(1), "visually identical frames"],
  ], { x: 0.6, y: 1.45, w: 12.1, colW: [3.2, 3.1, 2.1, 3.7], rowH: 0.45, fontSize: 12, objectName: "verify-table" });
  head(s, "Metrics used", 0.6, 4.85, 6);
  txt(s, bullets([
    "Max abs. difference: largest voxel-wise gap; ~10⁻³–10⁻⁷ = floating-point rounding only (float32, different summation order)",
    "PSNR = 10 · log₁₀(range² / MSE) dB, higher is better",
    "Dice = 2|A ∩ B| / (|A| + |B|), 1 = perfect overlap",
  ]), { x: 0.6, y: 5.25, w: 12.1, h: 1.45, fontSize: 13 });
  notes(s, "Speed means nothing if the answer changes, so every stage is checked. Differences between CPU and GPU are at floating-point rounding level, because the GPU adds numbers in a different order. " +
    "The threshold stage matches exactly: identical histograms and zero different labels out of 59 million voxels. Quality against ground truth uses PSNR for images and Dice for masks; the formulas are below.");

  // =========================================================================================
  // MEMBER 5
  // =========================================================================================
  // 25. Benchmark design
  const pendingIcons = [];
  s = slide(); tag(s, 5);
  s.addText("Benchmark design: how we measured fairly", { placeholder: "title" });
  [["FaRocket", "Warm-up", "first run untimed: CUDA context, cuFFT plans, kernel loading"], ["FaStopwatch", "Time", "cudaEvent inside kernels; perf_counter + synchronize in Python"],
   ["FaChartLine", "Repeat", "3–10 runs, mean reported (1 run for very long CPU cases)"], ["FaCheckCircle", "Verify", "compare outputs CPU vs GPU every time"]].forEach(([ic, t, d], i) => {
    const x = 0.6 + i * 3.08;
    card(s, x, 1.45, 2.88, 1.85, C.background1, "bstep-" + i);
    pendingIcons.push(iconCircle(s, ic, x + 0.2, 1.6, 0.55, "000000"));
    txt(s, t, { x: x + 0.9, y: 1.68, w: 1.9, h: 0.4, fontSize: 16, bold: true });
    txt(s, d, { x: x + 0.2, y: 2.25, w: 2.5, h: 0.95, fontSize: 12 });
  });
  await Promise.all(pendingIcons);
  code(s, "cudaEventRecord(e1);\nfor (int r = 0; r < reps; ++r)\n    conv_tiled<<<grid, block>>>(dIn, dOut, D, H, W);\ncudaEventRecord(e2);\ncudaEventSynchronize(e2);\ncudaEventElapsedTime(&ms, e1, e2);   // GPU clock\nprintf(\"tiled_ms %f\", ms / reps);",
    0.6, 3.5, 6.0, 2.2, 10.5);
  txt(s, "Kernel launches are asynchronous: timing without an event or synchronize would only measure the launch, not the work.", { x: 0.6, y: 5.8, w: 6.0, h: 0.85, fontSize: 12, italic: true });
  head(s, "Fairness rules", 6.9, 3.5, 5.8);
  txt(s, bullets([
    `Same algorithm and data on both sides; float32 everywhere (except U-Net FP16 option)`,
    `CPU baselines: NumPy/SciPy (CT, filter, threshold) and PyTorch CPU with ${mri.cpu_threads} threads (MRI, U-Net, render)`,
    "Report both: kernel-only time and total time including PCIe copies",
    "Also compare GPU vs GPU (naive vs tiled, global vs shared atomics, cuDNN vs ours)",
    "Scaling tests: 128²–512² CT, 64³–512³ FFT, 64³–256³ filter, 7 M–59 M voxel threshold",
  ]), { x: 6.9, y: 3.9, w: 5.8, h: 2.8, fontSize: 12.5 });
  notes(s, "Our benchmark protocol: warm up first, because the first GPU call pays for setup; time with CUDA events, because kernel launches return immediately; repeat and average; and always verify the output. " +
    "For fairness we report both kernel-only and total time including transfers, and where the CPU library is multi-threaded we let it use all cores. We also compare GPU variants against each other to isolate the effect of each optimisation.");

  // 26. Per-stage timings
  s = slide(); tag(s, 5);
  s.addText("Results: CPU vs GPU time for every stage", { placeholder: "title" });
  const rows26 = [[th("Stage (workload)"), th("CPU", { align: "right" }), th("GPU incl. transfers", { align: "right" }), th("Speedup", { align: "right" })]];
  const stageRows = [...pipe.stages.map((p) => [p.stage.replace(/\^2/g, "²").replace(/\^3/g, "³").replace(/ x /g, " × "), p.cpu_ms, p.gpu_ms]),
    ["Otsu threshold (59 M voxels)", thr2.cpu_ms, thr2.gpu_total_ms]];
  stageRows.forEach(([n, c, g]) => rows26.push([n, { text: ms(c), options: { align: "right" } }, { text: ms(g), options: { align: "right" } }, { text: fx(c / g), options: { align: "right", bold: true } }]));
  rows26.push([{ text: "Pipeline total (5 main stages)", options: { bold: true } }, { text: ms(pipe.total_cpu_ms), options: { align: "right", bold: true } },
    { text: ms(pipe.total_gpu_ms), options: { align: "right", bold: true } }, { text: fx(pipe.speedup), options: { align: "right", bold: true } }]);
  table(s, rows26, { x: 0.6, y: 1.45, w: 6.9, colW: [3.5, 1.1, 1.3, 1.0], rowH: 0.45, fontSize: 12, objectName: "stage-table" });
  s.addChart(pres.charts.BAR, [{ name: "Speedup", labels: ["CT recon", "MRI recon", "Denoise", "Threshold", "U-Net infer", "Render"],
    values: [ctv.speedup_total, mri.speedup, f256.speedup_tiled_with_transfer, thr2.speedup_total, inf.speedup_amp, vis.dvr_speedup].map((v) => +v.toFixed(1)) }],
    chartBase({ x: 7.8, y: 1.45, w: 4.9, h: 4.6, barDir: "bar", barGapWidthPct: 40, chartColors: ["000000"], title: "Speedup incl. transfers (×, log scale)",
      valAxisLogScaleBase: 10, valAxisMinVal: 1, valAxisCrossesAt: 1, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", showLegend: false, objectName: "speedup-chart" }));
  txt(s, "Speedup depends on arithmetic per byte: CT back-projection does 256 interpolations per pixel (265×); MRI is many small FFTs with less work each (7×).",
    { x: 0.6, y: 5.25, w: 6.9, h: 1.0, fontSize: 13, italic: true });
  notes(s, "Here is every stage. All GPU times include copying data in and out. The whole five-stage pipeline goes from " + ms(pipe.total_cpu_ms) + " to " + ms(pipe.total_gpu_ms) + ", " + fx(pipe.speedup) + ". " +
    "The speedups differ a lot, and the reason is arithmetic intensity: stages that do a lot of maths per byte, like back-projection, gain hundreds of times; stages dominated by memory or small operations gain less.");

  // 27. Scaling
  s = slide(); tag(s, 5);
  s.addText("Scaling: bigger data → bigger GPU advantage", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "CPU (NumPy)", labels: ct.scaling.map((r) => r.N + "²"), values: ct.scaling.map((r) => +r.cpu_numpy_ms.toFixed(2)) },
    { name: "GPU", labels: ct.scaling.map((r) => r.N + "²"), values: ct.scaling.map((r) => +r.gpu_total_ms.toFixed(2)) },
  ], chartBase({ x: 0.6, y: 1.45, w: 6.0, h: 3.0, barDir: "col", barGapWidthPct: 60, chartColors: ["A6A6A6", "000000"], title: "CT slice reconstruction, ms (log)",
    valAxisLogScaleBase: 10, valAxisMinVal: 0.1, valAxisCrossesAt: 0.1, showLegend: true, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisLabelFormatCode: "General", objectName: "ct-scaling" }));
  s.addChart(pres.charts.BAR, [
    { name: "NumPy", labels: fft.map((r) => r.n + "³"), values: fft.map((r) => +r.numpy_ms.toPrecision(3)) },
    { name: `PyTorch CPU ${mri.cpu_threads} thr.`, labels: fft.map((r) => r.n + "³"), values: fft.map((r) => +r.torch_cpu_ms.toPrecision(3)) },
    { name: "cuFFT", labels: fft.map((r) => r.n + "³"), values: fft.map((r) => +r.cufft_ms.toPrecision(2)) },
  ], chartBase({ x: 6.8, y: 1.45, w: 5.9, h: 3.0, barDir: "col", barGapWidthPct: 50, chartColors: ["BFBFBF", "7F7F7F", "000000"], title: "3D FFT, ms (log)",
    valAxisLogScaleBase: 10, valAxisMinVal: 0.01, valAxisCrossesAt: 0.01, showLegend: true, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "General", dataLabelFontSize: 8, valAxisLabelFormatCode: "General", objectName: "fft-scaling" }));
  s.addChart(pres.charts.LINE, [
    { name: "Tiled kernel", labels: flt.map((r) => r.n + "³"), values: flt.map((r) => +r.speedup_tiled.toFixed(0)) },
    { name: "Tiled + transfers", labels: flt.map((r) => r.n + "³"), values: flt.map((r) => +r.speedup_tiled_with_transfer.toFixed(0)) },
  ], chartBase({ x: 0.6, y: 4.6, w: 6.0, h: 2.15, chartColors: ["000000", "7F7F7F"], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6, title: "Gaussian filter speedup by size (×)",
    showLegend: true, legendPos: "r", showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0", objectName: "conv-scaling" }));
  card(s, 6.8, 4.6, 5.9, 2.15, C.background2, "scaling-notes", false);
  txt(s, bullets([
    `CT: ${fx(ct.scaling[0].speedup_total)} at 128² → ${fx(ct512.speedup_total)} at 512²: small jobs cannot fill ${cores.toLocaleString("en-US")} cores`,
    `FFT 512³: ${ms(big.cufft_ms)} vs ${ms(big.torch_cpu_ms)} (${fx(big.speedup_vs_torch_cpu)}); vs NumPy ${fx(big.speedup_vs_numpy)}`,
    `Threshold: 7.4 M → 59 M voxels, compute speedup ${fx(thr1.speedup_compute)} → ${fx(thr2.speedup_compute)}`,
  ]), { x: 7.0, y: 4.72, w: 5.55, h: 1.95, fontSize: 12 });
  notes(s, "Scaling is the key performance lesson. With small inputs the GPU cannot fill its thousands of cores and fixed costs dominate; as the data grows the advantage grows. " +
    "CT reconstruction goes from about 26 times at 128 pixels to over 900 times at 512. The FFT chart also shows why the baseline matters: compared with single-threaded NumPy cuFFT looks 300 times faster, but against a multi-threaded CPU FFT it is about 12 times.");

  // 28. Transfer vs compute + Amdahl
  s = slide(); tag(s, 5);
  s.addText("Where the GPU time goes: transfers and Amdahl's law", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "Upload (H→D)", labels: ["CT volume", "Denoise 256³", "Threshold 59 M"], values: [ctv.gpu_h2d_ms, f256.h2d_ms, thr2.h2d_ms].map((v) => +v.toFixed(1)) },
    { name: "Compute", labels: ["CT volume", "Denoise 256³", "Threshold 59 M"], values: [ctv.gpu_kernel_ms + ctv.gpu_filter_ms, f256.gpu_tiled_ms, thr2.gpu_compute_ms].map((v) => +v.toFixed(1)) },
    { name: "Download (D→H)", labels: ["CT volume", "Denoise 256³", "Threshold 59 M"], values: [ctv.gpu_d2h_ms, f256.d2h_ms, thr2.d2h_ms].map((v) => +v.toFixed(1)) },
  ], chartBase({ x: 0.6, y: 1.45, w: 7.0, h: 3.3, barDir: "bar", barGrouping: "stacked", chartColors: ["7F7F7F", "000000", "BFBFBF"], title: "GPU time split, ms",
    showLegend: true, showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: "0.0", dataLabelColor: "FFFFFF", objectName: "breakdown-chart" }));
  head(s, "Amdahl's law", 7.9, 1.45, 4.8);
  txt(s, "Speedup = 1 / ((1 − p) + p / s)", { x: 7.9, y: 1.85, w: 4.8, h: 0.5, fontSize: 18, bold: true, fontFace: "Cambria" });
  txt(s, bullets([
    "p = part of the work that is accelerated, s = how much faster it gets",
    "Copies over PCIe are the part that is not accelerated",
    `Denoise: compute alone is ${fx(f256.speedup_tiled)}, but with copies only ${fx(f256.speedup_tiled_with_transfer)}`,
    `Threshold: ${fx(thr2.speedup_compute)} → ${fx(thr2.speedup_total)}`,
  ]), { x: 7.9, y: 2.4, w: 4.8, h: 2.4, fontSize: 12.5 });
  table(s, [
    [th("Fix for the transfer bottleneck"), th("Effect")],
    ["Keep data on the GPU between stages (chain kernels)", "pay upload once, download once per study"],
    ["Pinned (page-locked) host memory", "faster copies, often close to PCIe peak"],
    ["CUDA streams: copy chunk k+1 while computing chunk k", "hides copy time behind compute"],
    ["Send compact results (uint8 labels, not float32)", "4× less data back (we do this)"],
  ], { x: 0.6, y: 4.95, w: 12.1, colW: [6.6, 5.5], rowH: 0.35, fontSize: 12, objectName: "fix-table" });
  notes(s, "Once the kernels are fast, the copies dominate. For denoising and thresholding, most of the GPU time is moving data over PCIe. Amdahl's law explains it: the overall speedup is limited by the part you did not accelerate. " +
    "The fixes are standard: keep data on the GPU for the whole pipeline, use pinned memory, overlap copies with compute using streams, and send back compact results, which we already do with 1-byte labels.");

  // =========================================================================================
  // MEMBER 6
  // =========================================================================================
  // 29. Challenges & limitations
  s = slide(); tag(s, 6);
  s.addText("Challenges and limitations", { placeholder: "title" });
  const ch = [
    ["FaExchangeAlt", "PCIe transfers", `${f1(pcieGBs)} GB/s measured; ${Math.round(transferPct)}% of denoise GPU time is copying`],
    ["FaMemory", "GPU memory", `${hw.gpu_mem_gb} GB VRAM: a 512³ complex FFT needs 1 GB; big 4D studies need tiling or streaming`],
    ["FaBalanceScale", "Precision", "FP16 must be validated; results differ in the last bits because sums are reordered"],
    ["FaFlask", "Our data", "2 real scans; CT recon, denoise and lesions use simulated data with known truth"],
    ["FaCogs", "Engineering", "Windows build needed MSVC + nvcc; CuPy was blocked by OS security policy, so we used PyTorch + own kernels"],
    ["FaLock", "Clinical use", "patient privacy (HIPAA/GDPR) and FDA/CE approval apply to real deployments"],
  ];
  for (let i = 0; i < 6; i++) {
    const x = 0.6 + (i % 3) * 4.1, y = 1.45 + Math.floor(i / 3) * 2.65;
    card(s, x, y, 3.9, 2.45, C.background1, "chal-" + i);
    await iconCircle(s, ch[i][0], x + 0.25, y + 0.25, 0.6, "000000");
    txt(s, ch[i][1], { x: x + 1.0, y: y + 0.3, w: 2.75, h: 0.5, fontSize: 16, bold: true, valign: "middle" });
    txt(s, ch[i][2], { x: x + 0.25, y: y + 1.05, w: 3.45, h: 1.3, fontSize: 13 });
  }
  notes(s, "Honest limitations. Transfers and GPU memory size are the main technical limits. Half precision needs validation. Most of our accuracy tests use simulated data, chosen deliberately for exact ground truth. " +
    "We also hit practical issues: on Windows the CUDA compiler needs Visual Studio, and the CuPy library was blocked by the operating system's security policy, so we wrote our own kernels instead. Real clinical use adds privacy and regulation.");

  // 30. Real world & future
  s = slide(); tag(s, 6);
  s.addText("Real-world use and future scope", { placeholder: "title" });
  head(s, "Already in hospitals and scanners", 0.6, 1.45, 6);
  txt(s, bullets([
    "GPU deep-learning reconstruction in scanners: GE TrueFidelity, Siemens Deep Resolve, Canon AiCE",
    "NVIDIA Clara / Holoscan: GPU platform for medical imaging and devices",
    "MONAI: open-source PyTorch framework for medical AI (segmentation, deployment)",
    "GPU iterative reconstruction allows lower-dose CT with similar image quality",
  ]), { x: 0.6, y: 1.85, w: 5.9, h: 2.6, fontSize: 13 });
  head(s, "Next steps for our project", 6.8, 1.45, 5.9);
  txt(s, bullets([
    "GPU-resident pipeline: one upload, all stages on GPU, one download",
    "Pinned memory + CUDA streams to overlap copies and compute",
    "Separable Gaussian (15 instead of 125 ops/voxel) and texture memory for FBP",
    "3D U-Net on real labelled data (Medical Segmentation Decathlon, LIDC)",
    "Multi-GPU / cloud for large studies; TensorRT for faster inference",
  ]), { x: 6.8, y: 1.85, w: 5.9, h: 2.6, fontSize: 13 });
  card(s, 0.6, 4.7, 12.1, 2.0, C.background2, "impact", false);
  txt(s, "Impact", { x: 0.85, y: 4.8, w: 4, h: 0.35, fontSize: 15, bold: true });
  [["Faster scans", "4× less MRI data, recon in < 0.1 s"], ["Interactive 3D", `${Math.round(vis.dvr_gpu_fps)} fps rotating views`], ["AI at the scanner", `${ms(inf.ms_per_slice_gpu_amp)} per slice`], ["Same answers", "verified CPU = GPU"]].forEach(([t, d], i) => {
    const x = 0.85 + i * 2.95;
    txt(s, t, { x, y: 5.25, w: 2.8, h: 0.4, fontSize: 18, bold: true, fontFace: "Cambria" });
    txt(s, d, { x, y: 5.7, w: 2.8, h: 0.7, fontSize: 13, color: HEX.muted });
  });
  notes(s, "What we built is a small version of what industry already does: major scanner vendors ship GPU-based deep-learning reconstruction, and NVIDIA Clara and MONAI are used in hospitals. " +
    "Our next steps follow directly from our measurements: keep the data on the GPU, overlap copies with compute, use cheaper separable filters, and train a 3D U-Net on real labelled data.");

  // 31. Conclusion
  s = slide("CONTENT_DARK"); tag(s, 6);
  s.addText("Conclusion", { placeholder: "title" });
  const conc = [
    ["FaRocket", `${fx(pipe.speedup)} faster end to end`, `${ms(pipe.total_cpu_ms)} → ${ms(pipe.total_gpu_ms)} per study; CT ${fx(ctv.speedup_total)}, U-Net ${fx(inf.speedup_amp)}, threshold ${fx(thr2.speedup_total)}`],
    ["FaCheckCircle", "Same answers", "differences only at floating-point rounding; threshold labels identical"],
    ["FaMemory", "Memory decides performance", `shared-memory tiling ${f256.tiled_vs_naive.toFixed(2)}×, shared-memory histogram ${f1(thr2.shared_vs_global)}×, PCIe copies dominate simple stages`],
    ["FaCube", "Pattern → technique", "map, stencil, histogram, FFT, gather, ray casting, GEMM each need a different GPU strategy"],
  ];
  for (let i = 0; i < 4; i++) {
    const y = 1.5 + i * 1.3;
    await iconCircle(s, conc[i][0], 0.6, y, 0.75, "404040");
    txt(s, conc[i][1], { x: 1.6, y: y - 0.02, w: 11, h: 0.45, fontSize: 20, bold: true, color: "FFFFFF" });
    txt(s, conc[i][2], { x: 1.6, y: y + 0.45, w: 11, h: 0.6, fontSize: 15, color: "D9D9D9" });
  }
  notes(s, `To conclude: our GPU pipeline is ${fx(pipe.speedup)} faster end to end with the same results. The biggest lesson is that memory, not arithmetic, decides GPU performance: shared memory, coalesced access and avoiding PCIe copies. ` +
    "And each image operation maps to a parallel pattern that tells you which GPU technique to use. Thank you.");

  // 32. References
  s = slide("CONTENT_DARK"); tag(s, 6);
  s.addText("References", { placeholder: "title" });
  const refs = [
    "Kak & Slaney, Principles of Computerized Tomographic Imaging, IEEE Press, 1988.",
    "Shepp & Logan, “The Fourier reconstruction of a head section,” IEEE Trans. Nuclear Science, 1974.",
    "Otsu, “A threshold selection method from gray-level histograms,” IEEE Trans. Systems, Man, and Cybernetics, 1979.",
    "Lustig, Donoho & Pauly, “Sparse MRI: the application of compressed sensing for rapid MR imaging,” Magn. Reson. Med., 2007.",
    "Ronneberger, Fischer & Brox, “U-Net: convolutional networks for biomedical image segmentation,” MICCAI, 2015.",
    "Eklund et al., “Medical image processing on the GPU: past, present and future,” Medical Image Analysis, 2013.",
    "Kirk & Hwu, Programming Massively Parallel Processors, 4th ed., Morgan Kaufmann, 2022.",
    "Cardoso et al., “MONAI: an open-source framework for deep learning in healthcare,” arXiv, 2022.",
    "Saver, “Time is brain, quantified,” Stroke, 2006.",
    "McDonald et al., radiologist workload study, Academic Radiology, 2015.",
    "NVIDIA, CUDA C++ Programming Guide; cuFFT and cuDNN documentation, 2026.",
    "Stanford Volume Data Archive (CThead, MRbrain), graphics.stanford.edu/data/voldata.",
  ];
  txt(s, refs.map((r, i) => ({ text: r, options: { bullet: { type: "number" }, breakLine: i < refs.length - 1, paraSpaceAfter: 5 } })),
    { x: 0.6, y: 1.5, w: 12.1, h: 5.2, fontSize: 14, color: "D9D9D9" });
  notes(s, "These are our sources: the classic papers for each algorithm, the GPU programming references, and the public data archive.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
