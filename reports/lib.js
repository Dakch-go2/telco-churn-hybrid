const fs = require("fs");
const D = require("docx");
const { Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType, HeadingLevel,
  ShadingType, BorderStyle, TabStopType, LeaderType, LevelFormat, PageBreak } = D;

const FONT = "Times New Roman";
const W = 9355; // мәтін ені DXA (A4, сол жақ 3 см, оң жақ 1,5 см)
const state = { fig: 0, tab: 0, toc: [] };
const TABORDER = ["cost","rules","decisions","examples","features","grids","compare","thresholds","imbalance","errcontract","worst5","groups","steps","endpoints","fields","inputs","dbschema","testlist","hybrid","rulecov","versions"];
const T = (k) => { const i = TABORDER.indexOf(k); if (i < 0) throw new Error("tab key " + k); return i + 1; };

const num = (x, d = 1) => Number(x).toFixed(d).replace(".", ",");
const pct = (x, d = 1) => num(x * 100, d) + "%";
const thousands = (x) => String(Math.round(x)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

function run(t, o = {}) { return new TextRun({ text: t, font: FONT, size: o.size || 28, bold: o.bold, italics: o.italics }); }

function P(text, o = {}) {
  const parts = Array.isArray(text) ? text : [text];
  return new Paragraph({
    alignment: o.align || AlignmentType.JUSTIFIED,
    indent: o.noIndent ? undefined : { firstLine: 709 },
    spacing: { before: 0, after: o.after ?? 100, line: 240 },
    keepNext: o.keepNext,
    children: parts.map((p) => typeof p === "string" ? run(p, o) : p),
  });
}
const B = (t) => new TextRun({ text: t, font: FONT, size: 28, bold: true });

function H1(text, pageBreak = true) {
  state.toc.push({ text, level: 1 });
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: pageBreak, children: [new TextRun({ text, font: FONT, size: 32, bold: true })] });
}
function H2(text) {
  state.toc.push({ text, level: 2 });
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT, size: 28, bold: true })] });
}
function bullets(items) {
  return items.map((t) => new Paragraph({
    numbering: { reference: "bul", level: 0 }, alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 60, line: 240 },
    children: (Array.isArray(t) ? t : [t]).map((p) => typeof p === "string" ? run(p) : p),
  }));
}
function pngSize(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fig(path, caption, widthPx = 520) {
  state.fig++;
  const { w, h } = pngSize(path);
  const height = Math.round((widthPx * h) / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: widthPx, height },
        altText: { title: caption, description: caption, name: "fig" + state.fig } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 160 },
      children: [run(`${state.fig}-сурет – ${caption}`, { size: 28 })] }),
  ];
}
function caption(text, key) {
  state.tab++;
  if (key && T(key) !== state.tab) throw new Error(`table order mismatch: ${key} expected ${T(key)} got ${state.tab}`);
  return new Paragraph({ alignment: AlignmentType.LEFT, keepNext: true, spacing: { before: 120, after: 60 },
    children: [run(`${state.tab}-кесте – ${text}`)] });
}
const bd = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
function table(header, rows, widths, o = {}) {
  const fs_ = o.size || 22;
  const cell = (t, w, hdr, align) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: hdr ? { type: ShadingType.CLEAR, fill: "D9E2F3", color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: align || AlignmentType.LEFT, spacing: { before: 0, after: 0, line: 240 },
      children: [new TextRun({ text: String(t), font: FONT, size: fs_, bold: hdr })] })],
  });
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true, AlignmentType.CENTER)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], false, (o.center || []).includes(i) ? AlignmentType.CENTER : AlignmentType.LEFT)) }))],
  });
}
function code(text) {
  return text.split("\n").map((ln, i, a) => new Paragraph({
    spacing: { before: 0, after: 0, line: 240 },
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
    children: [new TextRun({ text: ln === "" ? " " : ln, font: "Courier New", size: 19 })],
  }));
}
const gap = () => new Paragraph({ spacing: { before: 0, after: 120 }, children: [] });
function excerpt(file, from, to) {
  return fs.readFileSync(file, "utf8").split("\n").slice(from - 1, to).join("\n");
}
function tocPara(text, level, page) {
  return new Paragraph({
    tabStops: [{ type: TabStopType.RIGHT, position: W, leader: LeaderType.DOT }],
    indent: { left: level === 2 ? 400 : 0 }, spacing: { before: 0, after: 60, line: 240 },
    children: [new TextRun({ text, font: FONT, size: 28, bold: level === 1 }),
      new TextRun({ text: "\t" + page, font: FONT, size: 28 })],
  });
}
module.exports = { T, D, fs, FONT, W, state, num, pct, thousands, run, P, B, H1, H2, bullets, fig, caption, table, code, gap, excerpt, tocPara };
