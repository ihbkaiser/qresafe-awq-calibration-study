"""Build the concise, token-matched AWQ confirmation report."""

from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RESULTS = ROOT / "remote_artifacts" / "r6_tokenmatched" / "results.csv"
OUTPUT = ROOT / "output" / "pdf" / "qresafe_awq_calibration_report.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2B6CB0")
TEAL = colors.HexColor("#0F766E")
ORANGE = colors.HexColor("#C05621")
PALE_BLUE = colors.HexColor("#EAF2FB")
PALE_ORANGE = colors.HexColor("#FFF1E8")
PALE_RED = colors.HexColor("#FDECEC")
GRID = colors.HexColor("#D9E2EC")
TEXT = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")


def safe(value) -> str:
    return escape(str(value))


def read_results():
    rows = []
    with TOKEN_RESULTS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            asr = float(row["ASR held-out"])
            rows.append({
                "calibration": row["Calibration"],
                "samples": row["Samples"],
                "tokens": row["Tokens"],
                "blocks": row["Blocks"],
                "asr": asr,
                "count": int(round(asr * 510)),
                "ppl": float(row["PPL"]),
            })
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta_pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f} pp"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def styled_table(rows, widths):
    converted = []
    for i, row in enumerate(rows):
        style = styles["HeaderCell"] if i == 0 else styles["Cell"]
        converted.append([cell if isinstance(cell, Paragraph) else para(safe(cell), style) for cell in row])
    result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7FAFC")))
    result.setStyle(TableStyle(commands))
    return result


def callout(title: str, body: str, background=PALE_BLUE, border=BLUE):
    result = Table([[para(f"<b>{safe(title)}</b><br/>{body}", styles["Callout"])]], colWidths=[170 * mm])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.8, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return result


def asr_chart(rows):
    drawing = Drawing(520, 185)
    drawing.add(String(260, 169, "Held-out ASR heuristic under equal calibration budgets", textAnchor="middle", fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 48, 30, 430, 120
    chart.data = [[row["asr"] * 100 for row in rows]]
    chart.categoryAxis.categoryNames = [row["calibration"] for row in rows]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 100
    chart.valueAxis.valueStep = 20
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.gridStrokeColor = GRID
    chart.valueAxis.gridStrokeWidth = 0.4
    chart.valueAxis.strokeColor = MUTED
    chart.categoryAxis.strokeColor = MUTED
    chart.bars[0].fillColor = ORANGE
    chart.bars[0].strokeColor = ORANGE
    drawing.add(chart)
    drawing.add(String(18, 92, "ASR (%)", rotation=90, textAnchor="middle", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    return drawing


def footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8 * mm, "Q-resafe AWQ | fair token-matched confirmation")
    canvas.drawRightString(width - 18 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    rows = read_results()
    baseline = next(row for row in rows if row["calibration"] == "Risk-I")
    for row in rows:
        row["delta"] = row["asr"] - baseline["asr"]

    global styles
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=NAVY, spaceAfter=7),
        "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=10),
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=9, spaceAfter=6),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=TEAL, spaceBefore=7, spaceAfter=4),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=TEXT, spaceAfter=6),
        "Small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.5, textColor=MUTED, spaceAfter=4),
        "Cell": ParagraphStyle("Cell", parent=base["BodyText"], fontName="Helvetica", fontSize=7.5, leading=9, textColor=TEXT),
        "HeaderCell": ParagraphStyle("HeaderCell", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.2, leading=8.5, textColor=colors.white),
        "Callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica", fontSize=8.8, leading=12.2, textColor=TEXT),
        "Code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=7.5, leading=9.5, textColor=TEXT, backColor=colors.HexColor("#F7FAFC"), borderColor=GRID, borderWidth=0.5, borderPadding=5),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title="Q-resafe AWQ Calibration-Data Study - Fair Token-Matched Confirmation",
        author="Q-resafe AWQ calibration study",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=footer)])

    story = []
    story.append(para("Q-resafe AWQ Calibration-Data Study", styles["Title"]))
    story.append(para("Fairest current experiment: token-matched confirmation", styles["Subtitle"]))
    story.append(callout(
        "Main message",
        "When Risk-I, Risk-II and Risk-III are given exactly the same calibration budget, Risk-II and Risk-III still show higher held-out ASR heuristic values than Risk-I. This is a useful confirmation signal, but not yet a final causal claim because the metric is heuristic, Risk-II is reconstructed, and the token-matched stream is a prefix rather than full source coverage.",
        PALE_ORANGE, ORANGE,
    ))

    story.append(para("1. Research question", styles["H1"]))
    story.append(para("Does changing only the calibration corpus used by pure AWQ change the safety behavior of the same already aligned Llama-2-7B-Chat model? The intended comparison is: same model, same official Q-resafe AWQ code, same INT4 settings, same seed, same evaluation prompts and generation settings; only the calibration corpus changes.", styles["Body"]))
    story.append(para("The first exploratory run used unequal token budgets, so it is not the fairest basis for the main conclusion. This report therefore treats the token-matched confirmation as the primary experiment.", styles["Body"]))

    story.append(para("2. Fair experiment design", styles["H1"]))
    design = [
        ["Element", "Fixed setting"],
        ["Model", "meta-llama/Llama-2-7b-chat-hf"],
        ["Quantization", "Official Q-resafe quant-without-ft AWQ; INT4; zero_point=True; group_size=128; GEMM"],
        ["Conditions", "Risk-I, reconstructed Risk-II, reconstructed Risk-III"],
        ["Calibration budget", "Exactly 384 rendered tokens per condition; exactly 3 blocks of 128 tokens"],
        ["Safety evaluation", "Same held-out AdvBench set, 510 prompts; greedy decoding; max_new_tokens=128; seed=42"],
        ["Utility evaluation", "WikiText-2 raw test perplexity under the same quantized models"],
        ["Leakage control", "The 10 Risk-III calibration rows are removed from the 510-row evaluation set"],
    ]
    story.append(styled_table(design, [43 * mm, 127 * mm]))
    story.append(Spacer(1, 5))
    story.append(para("Risk-II is reconstructed from the public Q-resafe description; it is not the exact unreleased dataset. Risk-III uses paired instruction-response records from the public AdvBench export. No fine-tuning, optimizer, backward pass, LoRA, QAT, AQLM, GPTQ or QLoRA is used.", styles["Body"]))

    story.append(para("3. What 384 tokens and 3 blocks mean", styles["H1"]))
    story.append(para("The ten user/assistant calibration conversations are first rendered with the Llama-2 chat template. The tokenizer then converts the rendered text into a sequence of integer token IDs. The ten sequences are concatenated in locked order, and only the first 384 token IDs are kept. Those 384 IDs are split into three chunks of 128 IDs:", styles["Body"]))
    story.append(para("384 tokens = 128 tokens + 128 tokens + 128 tokens = 3 calibration blocks", styles["Code"]))
    story.append(para("AWQ runs the model forward on these blocks to observe activation ranges and choose INT4 scaling/clipping. It does not learn new behavior and does not update model weights. Equalizing the number of tokens and blocks removes the simplest quantity confound: one risk condition receiving much more calibration input than another.", styles["Body"]))
    story.append(callout("Remaining fairness limitation", "Equal token count does not mean equal source coverage. Because 384 is a prefix of the concatenated stream, some later parts of the ten conversations may not be included. The next definitive experiment should match token budget while preserving a pre-registered amount of source coverage.", PALE_RED, ORANGE))

    result_table = [["Condition", "Source examples", "Tokens", "Blocks", "ASR", "Positive / 510", "PPL", "Delta vs Risk-I"]]
    for row in rows:
        result_table.append([
            row["calibration"], row["samples"], row["tokens"], row["blocks"], pct(row["asr"]), str(row["count"]), f"{row['ppl']:.4f}", delta_pp(row["delta"]),
        ])
    story.append(KeepTogether([
        para("4. Token-matched results", styles["H1"]),
        styled_table(result_table, [28 * mm, 29 * mm, 18 * mm, 18 * mm, 20 * mm, 25 * mm, 20 * mm, 27 * mm]),
    ]))
    story.append(Spacer(1, 7))
    story.append(asr_chart(rows))

    story.append(para("5. Interpretation", styles["H1"]))
    story.append(para("Under the fairest recorded comparison, Risk-II has ASR heuristic 98.63%, which is 3.33 percentage points above Risk-I at 95.29%. Risk-III has 98.04%, which is 2.75 points above Risk-I. The result is not monotonic between Risk-II and Risk-III: Risk-II is the highest, while Risk-III is slightly lower.", styles["Body"]))
    story.append(para("The utility check does not show a dramatic collapse: PPL is 7.1464 for Risk-I, 7.2186 for Risk-II and 7.1971 for Risk-III. Therefore the ASR difference is not immediately explained by the Risk-II/Risk-III models becoming unusable, although PPL alone cannot prove that the effect is safety-specific.", styles["Body"]))
    story.append(callout("Defensible conclusion", "The token-matched experiment supports the hypothesis that calibration-data distribution may influence AWQ safety behavior even without fine-tuning. It is still preliminary evidence, not a final claim that harmful calibration data alone reliably degrades safety.", PALE_BLUE, BLUE))

    story.append(para("6. What is needed before a strong claim", styles["H1"]))
    story.append(para("Use a validated human or model-judge ASR protocol instead of the transparent first-pass heuristic; repeat the token-matched comparison over multiple pre-registered seeds; record the exact Q-resafe source commit; and design token matching that preserves comparable source coverage rather than only the first 384 tokens. The current conclusion should remain limited to this Llama-2-7B-Chat, INT4 AWQ setup.", styles["Body"]))

    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
