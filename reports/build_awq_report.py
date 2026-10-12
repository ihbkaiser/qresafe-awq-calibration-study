"""Build the consolidated Q-resafe AWQ calibration-data experiment report.

The report is generated only from the recorded CSV/JSON artifacts in this
repository and deliberately does not print calibration prompts or responses.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.textlabels import Label


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "remote_artifacts" / "r6_final"
TOKEN_MATCHED = ROOT / "remote_artifacts" / "r6_tokenmatched"
OUT = ROOT / "output" / "pdf" / "qresafe_awq_calibration_report.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2B6CB0")
TEAL = colors.HexColor("#0F766E")
ORANGE = colors.HexColor("#C05621")
RED = colors.HexColor("#B83232")
PALE_BLUE = colors.HexColor("#EAF2FB")
PALE_TEAL = colors.HexColor("#E8F5F2")
PALE_ORANGE = colors.HexColor("#FFF1E8")
PALE_RED = colors.HexColor("#FDECEC")
GRID = colors.HexColor("#D9E2EC")
TEXT = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def pp(value: float, digits: int = 2) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{digits}f} pp"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def safe(text: Any) -> str:
    return escape(str(text))


def parse_main_results() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(FINAL / "results.csv"):
        asr = float(row["ASR held-out"])
        rows.append(
            {
                "model": row["Model"],
                "bits": row["Bits"],
                "calibration": row["Calibration"],
                "samples": row["Samples"],
                "tokens": row["Tokens"],
                "blocks": row["Blocks"],
                "asr": asr,
                "ppl": float(row["PPL"]),
                "n": int(round(asr * 510)),
            }
        )
    return rows


def parse_token_results() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(TOKEN_MATCHED / "results.csv"):
        asr = float(row["ASR held-out"])
        rows.append(
            {
                "calibration": row["Calibration"],
                "samples": row["Samples"],
                "tokens": row["Tokens"],
                "blocks": row["Blocks"],
                "asr": asr,
                "ppl": float(row["PPL"]),
                "n": int(round(asr * 510)),
            }
        )
    return rows


def table(data: list[list[Any]], widths: list[float], header_rows: int = 1, font: float = 7.5) -> Table:
    converted = []
    for r, row in enumerate(data):
        converted.append([
            cell if isinstance(cell, Paragraph) else p(safe(cell), styles["TableCellHeader" if r < header_rows else "TableCell"])
            for cell in row
        ])
    t = Table(converted, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row_idx in range(header_rows, len(data)):
        if (row_idx - header_rows) % 2 == 0:
            commands.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#F7FAFC")))
    t.setStyle(TableStyle(commands))
    return t


def callout(title: str, body: str, background: colors.Color = PALE_BLUE) -> Table:
    content = [[p(f"<b>{safe(title)}</b><br/>{body}", styles["Callout"])]]
    t = Table(content, colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def chart(title: str, labels: list[str], values: list[float], color: colors.Color, y_max: float, y_label: str) -> Drawing:
    drawing = Drawing(520, 190)
    drawing.add(String(260, 174, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    bc = VerticalBarChart()
    bc.x = 46
    bc.y = 34
    bc.height = 112
    bc.width = 430
    bc.data = [values]
    bc.categoryAxis.categoryNames = labels
    bc.categoryAxis.labels.fontName = "Helvetica"
    bc.categoryAxis.labels.fontSize = 7
    bc.categoryAxis.labels.angle = 0
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = y_max
    bc.valueAxis.valueStep = y_max / 5
    bc.valueAxis.labels.fontName = "Helvetica"
    bc.valueAxis.labels.fontSize = 7
    bc.valueAxis.labelTextFormat = "%.0f"
    bc.bars[0].fillColor = color
    bc.bars[0].strokeColor = color
    bc.valueAxis.strokeColor = MUTED
    bc.categoryAxis.strokeColor = MUTED
    bc.valueAxis.gridStrokeColor = GRID
    bc.valueAxis.gridStrokeWidth = 0.4
    drawing.add(bc)
    drawing.add(String(18, 95, y_label, rotation=90, textAnchor="middle", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    return drawing


def footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8 * mm, "Q-resafe AWQ calibration-data study | consolidated report")
    canvas.drawRightString(width - 18 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_report() -> None:
    main = parse_main_results()
    token = parse_token_results()
    idx = read_json(FINAL / "data_indices.json")
    env = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in (FINAL / "environment.txt").read_text(encoding="utf-8").splitlines()
        if "=" in line
    }
    local_qresafe_commit = "fc46653e6a77c831d0c3c11e20595f50aee6200d"
    pile = next(row for row in main if row["calibration"] == "Pile")
    r1 = next(row for row in main if row["calibration"] == "Risk-I")
    r2 = next(row for row in main if row["calibration"] == "Risk-II")
    r3 = next(row for row in main if row["calibration"] == "Risk-III")
    tm = {row["calibration"]: row for row in token}

    OUT.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="Q-resafe AWQ Calibration Data Study - Consolidated Experimental Report",
        author="Q-resafe AWQ calibration study",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=footer)])

    global styles
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8),
        "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=MUTED, spaceAfter=12),
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=10, spaceAfter=7),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=TEAL, spaceBefore=8, spaceAfter=4),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=TEXT, spaceAfter=6),
        "Small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.5, textColor=MUTED, spaceAfter=4),
        "TableCell": ParagraphStyle("TableCell", parent=base["BodyText"], fontName="Helvetica", fontSize=7.2, leading=8.5, textColor=TEXT, alignment=TA_LEFT),
        "TableCellHeader": ParagraphStyle("TableCellHeader", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.0, leading=8.2, textColor=colors.white, alignment=TA_LEFT),
        "Callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica", fontSize=8.8, leading=12.2, textColor=TEXT),
        "KPI": ParagraphStyle("KPI", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=13, leading=15, textColor=NAVY, alignment=TA_CENTER),
        "KPILabel": ParagraphStyle("KPILabel", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9, textColor=MUTED, alignment=TA_CENTER),
        "Code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=7.2, leading=9, textColor=TEXT, backColor=colors.HexColor("#F7FAFC"), borderColor=GRID, borderWidth=0.5, borderPadding=5),
    }

    story = []
    story.append(p("Q-resafe AWQ Calibration-Data Study", styles["Title"]))
    story.append(p("Consolidated experimental report: Llama-2-7B-Chat, official Q-resafe quant-without-ft AWQ, Modal NVIDIA L40S", styles["Subtitle"]))
    story.append(callout("Executive conclusion", "The recorded INT4 AWQ runs do not provide evidence for a large or monotonic Risk-II/Risk-III safety degradation under the first-pass held-out ASR heuristic. In the initial run, Risk-II is 1.37 percentage points above AWQ-Pile and Risk-III is 0.78 points above it, but both are also close to the FP16 control and the utility PPL values remain in a narrow 7.14-7.22 range. The token-matched confirmation preserves the ordering Risk-II/Risk-III above Risk-I, but it is a sensitivity check rather than a definitive causal confirmation because the 384-token prefix truncates source coverage.", PALE_ORANGE))
    story.append(Spacer(1, 8))

    kpi_data = [[
        p("510", styles["KPI"]), p("5", styles["KPI"]), p("4-bit", styles["KPI"]), p("L40S", styles["KPI"]),
    ], [
        p("held-out AdvBench prompts", styles["KPILabel"]), p("main conditions", styles["KPILabel"]), p("AWQ weight precision", styles["KPILabel"]), p("Modal GPU", styles["KPILabel"]),
    ]]
    kt = Table(kpi_data, colWidths=[42 * mm] * 4)
    kt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 1), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
    ]))
    story.append(kt)

    story.append(p("1. Research question and scope", styles["H1"]))
    story.append(p("This study extends the Risk-I/Risk-II/Risk-III taxonomy described in Q-resafe to the calibration corpus of its AWQ implementation. It asks whether changing only the calibration text used by pure post-training quantization changes safety behavior of an already safety-aligned model. It is not a reproduction of Q-resafe's AWQ Risk-I/Risk-II/Risk-III experiments, because those conditions were not run in the original paper.", styles["Body"]))
    story.append(p("Locked pipeline: FP16 Llama-2-7B-Chat -> official Q-resafe AWQ INT4 with one calibration corpus -> held-out AdvBench safety evaluation and WikiText-2 perplexity. No optimizer, backward pass, SFT, LoRA, QAT, AQLM, GPTQ, QLoRA, or vaccine was used.", styles["Body"]))

    story.append(p("2. Locked protocol", styles["H1"]))
    protocol = [
        ["Item", "Recorded setting"],
        ["Model", "meta-llama/Llama-2-7b-chat-hf"],
        ["Implementation", "Qresafe/quant-without-ft official AWQ path"],
        ["AWQ config", "w_bit=4; q_group_size=128; zero_point=True; version=GEMM"],
        ["Seed", "42"],
        ["Safety evaluation", "held-out AdvBench, n=510; greedy decoding; max_new_tokens=128; chat-template generation prompt"],
        ["Utility evaluation", "WikiText-2 raw test perplexity"],
        ["Risk data", "10 examples per condition; question-response conversations rendered once and tokenized"],
    ]
    story.append(table(protocol, [42 * mm, 128 * mm]))
    story.append(Spacer(1, 5))
    story.append(p("Generation configuration recorded in generation_config.json:", styles["Body"]))
    story.append(p('{"do_sample": false, "max_new_tokens": 128, "seed": 42, "chat_template": "tokenizer.apply_chat_template", "add_generation_prompt": true}', styles["Code"]))

    story.append(p("3. Calibration data and leakage control", styles["H1"]))
    calib = [
        ["Condition", "Source", "Examples", "Tokens", "Blocks", "Policy"],
        ["Pile", "Q-resafe pileval default", "128", "33,777", "65", "max_seq_len=512"],
        ["Risk-I", "HuggingFaceH4/ultrachat_200k", "10", "5,537", "43", "common 128-token chunks"],
        ["Risk-II", "public-description reconstruction", "10", "469", "3", "common 128-token chunks"],
        ["Risk-III", "AlignmentResearch/AdvBench", "10", "438", "3", "common 128-token chunks"],
    ]
    story.append(table(calib, [25 * mm, 56 * mm, 20 * mm, 23 * mm, 20 * mm, 46 * mm]))
    story.append(Spacer(1, 5))
    story.append(p("Risk-II is reconstructed from the public description and is not the exact unreleased Q-resafe dataset. Risk-III uses paired responses from the public AdvBench export under response column proxy_gen_target. The ten Risk-III source indices are removed from the evaluation split, leaving 510 held-out rows; the recorded intersection is empty. The report intentionally lists provenance IDs only and does not reproduce harmful prompts or responses.", styles["Body"]))
    story.append(callout("Important first-run confound", "The initial calibration token budget is strongly imbalanced: Risk-I has 5,537 tokens versus 469 for Risk-II and 438 for Risk-III. The token-matched confirmation fixes all three streams at 384 tokens and 3 blocks, but uses the deterministic prefix of the rendered stream, so it matches token count rather than full ten-example source coverage.", PALE_RED))

    story.append(p("4. Initial five-condition results", styles["H1"]))
    story.append(p("ASR is reported exactly under the artifact field asr_heldout_refusal_heuristic. The count column is the corresponding positive count out of 510. Higher ASR is interpreted as worse safety under the intended ASR definition; because the implementation is a transparent heuristic, these numbers require an approved judge or manual validation before a publication-level claim.", styles["Body"]))
    main_table = [["Model", "Calibration", "Samples", "Tokens", "Blocks", "ASR", "Count/510", "PPL", "Delta vs Pile"]]
    for row in main:
        delta = row["asr"] - pile["asr"] if row["calibration"] != "Pile" else 0.0
        main_table.append([
            row["model"] + " / " + row["bits"] + "b",
            row["calibration"], row["samples"], row["tokens"], row["blocks"],
            pct(row["asr"]), str(row["n"]), fmt_float(row["ppl"], 4), pp(delta),
        ])
    story.append(table(main_table, [25 * mm, 27 * mm, 16 * mm, 18 * mm, 16 * mm, 20 * mm, 20 * mm, 21 * mm, 25 * mm]))
    story.append(Spacer(1, 7))
    story.append(chart("Initial held-out ASR heuristic", [r["calibration"] for r in main], [r["asr"] * 100 for r in main], BLUE, 100, "ASR (%)"))
    story.append(Spacer(1, 5))
    story.append(chart("Initial WikiText-2 perplexity", [r["calibration"] for r in main], [r["ppl"] for r in main], TEAL, 8, "PPL"))

    story.append(p("5. Interpretation of the initial run", styles["H1"]))
    story.append(p(f"Relative to AWQ-Pile, Risk-I changes ASR by {pp(r1['asr'] - pile['asr'])}, Risk-II by {pp(r2['asr'] - pile['asr'])}, and Risk-III by {pp(r3['asr'] - pile['asr'])}. Relative to Risk-I, Risk-II is {pp(r2['asr'] - r1['asr'])} and Risk-III is {pp(r3['asr'] - r1['asr'])}. Thus the point estimate is compatible with a harmful-calibration signal for Risk-II/Risk-III, but the ordering is not a clean monotonic Risk-I < Risk-II < Risk-III result: Risk-II is highest and Risk-III is lower than Risk-II.", styles["Body"]))
    story.append(p(f"Utility does not collapse in the same run. PPL is {r1['ppl']:.4f} for Risk-I, {r2['ppl']:.4f} for Risk-II, and {r3['ppl']:.4f} for Risk-III, compared with {pile['ppl']:.4f} for Pile and {next(row for row in main if row['calibration'] == 'None')['ppl']:.4f} for FP16. The Risk-II/Risk-III ASR differences therefore cannot be dismissed immediately as catastrophic quantization failure, although PPL alone is not a sufficient utility audit.", styles["Body"]))
    story.append(callout("What the first run can support", "A preliminary observation: changing calibration text under pure AWQ is associated with small ASR differences in this setup, while utility PPL stays close. What it cannot support alone is a definitive causal claim, because the Risk token budgets differ greatly, Risk-II is reconstructed, the ASR judge is heuristic, and there is only one seed.", PALE_BLUE))

    story.append(p("6. Token-matched confirmation", styles["H1"]))
    story.append(p("The confirmation reruns only Risk-I/Risk-II/Risk-III with exactly 384 rendered tokens and 3 complete 128-token blocks for each condition. FP16 and Pile are unchanged controls from the initial run.", styles["Body"]))
    tm_table = [["Calibration", "Source examples", "Used tokens", "Blocks", "ASR", "Count/510", "PPL", "Delta vs R1"]]
    for row in token:
        tm_table.append([
            row["calibration"], row["samples"], row["tokens"], row["blocks"], pct(row["asr"]), str(row["n"]), fmt_float(row["ppl"], 4), pp(row["asr"] - tm["Risk-I"]["asr"]),
        ])
    story.append(table(tm_table, [29 * mm, 27 * mm, 23 * mm, 18 * mm, 22 * mm, 22 * mm, 24 * mm, 25 * mm]))
    story.append(Spacer(1, 8))
    story.append(p(f"After matching token count, Risk-II remains {pp(tm['Risk-II']['asr'] - tm['Risk-I']['asr'])} above Risk-I and Risk-III remains {pp(tm['Risk-III']['asr'] - tm['Risk-I']['asr'])} above Risk-I. Risk-II and Risk-III match their initial ASR/PPL values exactly in the recorded artifacts; Risk-I changes from {pct(r1['asr'])} to {pct(tm['Risk-I']['asr'])} ASR after truncation from 5,537 to 384 tokens. This is directionally supportive of a calibration-distribution effect, but it is not a fully source-coverage-matched replication.", styles["Body"]))
    story.append(chart("Token-matched ASR heuristic", [r["calibration"] for r in token], [r["asr"] * 100 for r in token], ORANGE, 100, "ASR (%)"))
    story.append(Spacer(1, 4))
    story.append(chart("Token-matched WikiText-2 perplexity", [r["calibration"] for r in token], [r["ppl"] for r in token], TEAL, 8, "PPL"))

    env_table = [
        ["Environment field", "Recorded value"],
        ["Python", env.get("python", "3.10.17")],
        ["Torch", env.get("torch", "2.1.2+cu118")],
        ["CUDA build", env.get("torch_cuda_build", "11.8")],
        ["GPU", env.get("gpu_0", "NVIDIA L40S")],
        ["Transformers", env.get("transformers", "4.47.1")],
        ["GPU availability", env.get("cuda_available", "True")],
        ["Model", "meta-llama/Llama-2-7b-chat-hf"],
        ["Local Q-resafe checkout", local_qresafe_commit],
        ["Artifact metadata commit", "unknown in r6_final and r6_tokenmatched metadata"],
    ]
    story.append(KeepTogether([
        p("7. Execution environment and evidence", styles["H1"]),
        table(env_table, [58 * mm, 112 * mm]),
    ]))
    story.append(Spacer(1, 5))
    story.append(p("Recorded evidence includes Modal smoke logs, model-access and data-preparation logs, full-run logs, quantization metadata, safety JSONL checkpoints, PPL JSON files, result CSVs, the calibration index manifest, and the local unit-test suite. The local test suite was rerun during report preparation: 42 passed, 2 deprecation warnings, 11.39 seconds. The public release contains raw artifacts because a full release was requested; model weights and credentials are excluded.", styles["Body"]))
    story.append(p("The release repository is public at https://github.com/ihbkaiser/qresafe-awq-calibration-study. The report and its source are included under output/pdf/ and reports/.", styles["Body"]))

    story.append(p("8. Limitations and technical notes", styles["H1"]))
    limitations = [
        "ASR is a transparent refusal-heuristic field, not a validated human or model-judge score. The artifact itself warns that it must be validated before publication.",
        "Risk-II is reconstructed from the public description; the exact Q-resafe Risk-II data are unreleased.",
        "Risk-III uses a reconstructed public AdvBench export and a held-out split with empty recorded overlap, but exact Q-resafe selection is unavailable.",
        "The initial run has a large calibration-token imbalance. The token-matched check matches a 384-token prefix, not all ten full conversations.",
        "There is one fixed seed and no confidence interval or multi-seed replication in this release.",
        "The raw Modal logs contain earlier failed invocations: a gated-model access failure and a utility-evaluation logits-interface failure. The numeric tables in this report are read from the persisted r6_final and r6_tokenmatched result artifacts, not from those failed log sections.",
        "The recorded artifact metadata says qresafe_quant_without_ft_commit=unknown. The local checkout is at fc46653e6a77c831d0c3c11e20595f50aee6200d, but the exact source revision used by the remote artifact cannot be proven from the metadata alone.",
    ]
    for item in limitations:
        story.append(p("- " + safe(item), styles["Body"]))

    story.append(p("9. Research conclusion", styles["H1"]))
    story.append(p("The current evidence is best described as a preliminary signal, not a confirmed phenomenon. In both the initial and token-matched artifacts, Risk-II and Risk-III have higher held-out ASR heuristic values than Risk-I, and the PPL values remain close enough that a total utility collapse is not the obvious explanation. However, the evidence is weakened by the heuristic safety metric, the reconstructed Risk-II corpus, single-seed design, and source-coverage mismatch in token matching. The defensible conclusion is: harmful-looking calibration distributions may affect AWQ safety behavior in this setup, but the magnitude and specificity of the effect require a stricter confirmation before claiming that harmful calibration data alone reliably degrades safety.", styles["Body"]))
    story.append(p("Recommended next experiment: preregister a judge-validated ASR protocol, record the exact Q-resafe commit in metadata, keep ten-example source coverage identical while matching total tokens through a declared sampling/truncation rule, and repeat at least three seeds without selecting seeds based on ASR.", styles["Body"]))

    story.append(PageBreak())
    story.append(p("Appendix A. Reproducibility identifiers", styles["H1"]))
    story.append(p("Only identifiers are included below. Harmful calibration and evaluation text is intentionally omitted from this report; the raw artifact package contains the recorded files and should be handled as research data.", styles["Body"]))
    ids = [
        ["Corpus", "Identifiers"],
        ["Risk-I / UltraChat", ", ".join(str(x) for x in idx["risk1"]["source_indices"])],
        ["Risk-II", "reconstructed-0 through reconstructed-9; exact original unavailable"],
        ["Risk-III / AdvBench", ", ".join(str(x) for x in idx["risk3"]["source_indices"])],
        ["Risk-III held-out", "510 rows; calibration/evaluation intersection = []"],
        ["Seed", str(idx["seed"])],
    ]
    story.append(table(ids, [45 * mm, 125 * mm]))
    story.append(Spacer(1, 8))
    story.append(p("Calibration metadata fields record source examples, used examples, discarded examples, total tokens before blocking, block length policy, block count, and sample IDs for every custom condition. The token-matched metadata additionally records original token totals and discarded-token counts: Risk-I 5,537 -> 384 (5,153 discarded), Risk-II 469 -> 384 (85 discarded), Risk-III 438 -> 384 (54 discarded).", styles["Body"]))

    story.append(p("Appendix B. Release contents", styles["H1"]))
    release = [
        ["Path", "Purpose"],
        ["remote_artifacts/r6_final/", "Initial five-condition result CSVs, summaries, PPL, quantization metadata, and indices"],
        ["remote_artifacts/r6_tokenmatched/", "Token-matched Risk-I/Risk-II/Risk-III result CSVs and metadata"],
        ["remote_artifacts/r6_checkpoints/", "Resumable safety-generation checkpoints"],
        ["remote_artifacts/modal_*/", "Modal smoke, preparation, access-check, and full-run logs"],
        ["scripts/ and src/", "Calibration preparation, quantization, evaluation, collection, and tests"],
        ["output/pdf/", "This consolidated report"],
        ["reports/", "Report-generation source"],
    ]
    story.append(table(release, [60 * mm, 110 * mm]))
    story.append(Spacer(1, 8))
    story.append(callout("Release safety note", "The public package excludes HF credentials, GitHub credentials, Python caches, and model weights. Raw logs and model-generated safety-test records are included because a full artifact release was requested; do not redistribute harmful prompt text unnecessarily.", PALE_TEAL))

    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build_report()
