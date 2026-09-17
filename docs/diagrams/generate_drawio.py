#!/usr/bin/env python3
"""
Generates the draft class diagrams for the ABSA circuit fine-tuning pipeline
as draw.io (diagrams.net) files.

Run:
    python docs/diagrams/generate_drawio.py

Produces, in docs/diagrams/:
    00_pipeline_overview.drawio
    01_dataset_creation.drawio
    02_full_sft.drawio
    03_circuit_discovery.drawio
    04_selective_circuit_sft.drawio
    05_evaluation.drawio
    absa_eap_class_diagrams.drawio   (all six as pages of a single file)

The diagrams are drafts meant to be opened and refined in draw.io. If you edit
the .drawio files by hand, re-running this script will overwrite your edits.
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ROW_H = 22
TITLE_H = 46
DIVIDER_H = 8
COL_GAP = 70
BOX_GAP = 36

# ---------------------------------------------------------------- styles ----

PALETTE = {
    "script":   ("#fff2cc", "#d6b656"),  # SLURM / bash job scripts
    "entry":    ("#ffe6cc", "#d79b00"),  # run_*.py entry points
    "util":     ("#dae8fc", "#6c8ebf"),  # src/* helper modules
    "eap":      ("#e1d5e7", "#9673a6"),  # eap/* circuit package
    "data":     ("#d5e8d4", "#82b366"),  # on-disk artifacts
    "external": ("#f5f5f5", "#666666"),  # third-party classes
    "note":     ("#ffffff", "#b0b0b0"),  # annotations
}

CLASS_STYLE = (
    "swimlane;fontStyle=0;childLayout=stackLayout;horizontal=1;startSize={title_h};"
    "fillColor={fill};strokeColor={stroke};horizontalStack=0;resizeParent=1;"
    "resizeParentMax=0;html=1;verticalAlign=middle;align=center;"
    "whiteSpace=wrap;collapsible=0;marginBottom=0;"
)
ROW_STYLE = (
    "text;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
    "spacingLeft=6;spacingRight=6;overflow=hidden;rotatable=0;points=[[0,0.5],[1,0.5]];"
    "portConstraint=eastwest;whiteSpace=wrap;html=1;fontSize=11;"
)
DIVIDER_STYLE = (
    "line;strokeWidth=1;fillColor=none;align=left;verticalAlign=middle;"
    "spacingTop=-1;spacingLeft=3;spacingRight=3;rotatable=0;labelPosition=right;"
    "points=[];portConstraint=eastwest;strokeColor=inherit;"
)
NOTE_STYLE = (
    "shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;darkOpacity=0.05;"
    "fillColor=#ffffff;strokeColor=#b0b0b0;align=left;verticalAlign=top;"
    "spacingLeft=8;spacingTop=4;fontSize=11;size=14;"
)
HEADING_STYLE = "text;html=1;align=left;verticalAlign=middle;fontSize=20;fontStyle=1;"
SUBHEADING_STYLE = "text;html=1;align=left;verticalAlign=top;fontSize=12;fontColor=#555555;"

EDGE_STYLES = {
    # dashed open arrow: «use» dependency (calls / imports)
    "uses": "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;dashed=1;endArrow=open;"
            "endFill=0;endSize=12;strokeColor=#6c8ebf;fontSize=10;",
    # dashed open arrow, orange: a shell script launching a python entry point
    "invokes": "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;dashed=1;endArrow=open;"
               "endFill=0;endSize=12;strokeColor=#d79b00;fontSize=10;",
    # solid filled arrow, green: data flow (reads / writes an artifact)
    "flow": "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;"
            "endSize=8;strokeWidth=2;strokeColor=#82b366;fontSize=10;",
    # hollow triangle: generalization (subclass -> superclass)
    "extends": "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=0;"
               "endSize=16;strokeColor=#9673a6;fontSize=10;",
    # filled diamond: composition (whole -> part)
    "owns": "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;startArrow=diamondThin;"
            "startFill=1;startSize=14;endArrow=open;endFill=0;endSize=10;"
            "strokeColor=#9673a6;fontSize=10;",
    # plain open arrow: association
    "assoc": "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=open;endFill=0;"
             "endSize=12;strokeColor=#666666;fontSize=10;",
}


class Page:
    """One draw.io page: a column-stacked set of UML-ish class boxes plus edges."""

    def __init__(self, name, heading, subheading=""):
        self.name = name
        self.heading = heading
        self.subheading = subheading
        self.cells = []          # (id, value, style, x, y, w, h, parent, is_edge, src, tgt)
        self.columns = []        # list of dicts: {x, width, cursor}
        self._uid = 0

    # -- layout -------------------------------------------------------------
    def add_column(self, width, top=150):
        x = 40
        for col in self.columns:
            x = col["x"] + col["width"] + COL_GAP
        self.columns.append({"x": x, "width": width, "cursor": top})
        return len(self.columns) - 1

    def _next_id(self, prefix):
        self._uid += 1
        return f"{prefix}-{self._uid}"

    # -- shapes -------------------------------------------------------------
    def cls(self, cid, col, stereotype, name, attrs=(), methods=(), kind="util",
            gap=BOX_GAP, y=None):
        """Add a UML class box to column `col`. Returns its id."""
        column = self.columns[col]
        x, w = column["x"], column["width"]
        y = column["cursor"] if y is None else y

        fill, stroke = PALETTE[kind]
        title = f"<i>«{stereotype}»</i><br><b>{name}</b>" if stereotype else f"<b>{name}</b>"
        has_divider = bool(attrs) and bool(methods)
        height = TITLE_H + len(attrs) * ROW_H + len(methods) * ROW_H + (DIVIDER_H if has_divider else 0)

        self.cells.append(dict(
            id=cid, value=title, style=CLASS_STYLE.format(title_h=TITLE_H, fill=fill, stroke=stroke),
            x=x, y=y, w=w, h=height, parent="1", edge=False))

        offset = TITLE_H
        for a in attrs:
            self.cells.append(dict(id=self._next_id(cid), value=a, style=ROW_STYLE,
                                   x=0, y=offset, w=w, h=ROW_H, parent=cid, edge=False))
            offset += ROW_H
        if has_divider:
            self.cells.append(dict(id=self._next_id(cid), value="", style=DIVIDER_STYLE,
                                   x=0, y=offset, w=w, h=DIVIDER_H, parent=cid, edge=False))
            offset += DIVIDER_H
        for m in methods:
            self.cells.append(dict(id=self._next_id(cid), value=m, style=ROW_STYLE,
                                   x=0, y=offset, w=w, h=ROW_H, parent=cid, edge=False))
            offset += ROW_H

        column["cursor"] = y + height + gap
        return cid

    def note(self, cid, col, text, height=90, gap=BOX_GAP, y=None):
        column = self.columns[col]
        y = column["cursor"] if y is None else y
        self.cells.append(dict(id=cid, value=text, style=NOTE_STYLE,
                               x=column["x"], y=y, w=column["width"], h=height,
                               parent="1", edge=False))
        column["cursor"] = y + height + gap
        return cid

    def free_note(self, cid, x, y, w, h, text):
        """A note placed at an explicit position, independent of the columns."""
        self.cells.append(dict(id=cid, value=text, style=NOTE_STYLE,
                               x=x, y=y, w=w, h=h, parent="1", edge=False))
        return cid

    def add_legend(self):
        """Place the legend below everything, spanning the full width of the page."""
        right = max((c["x"] + c["w"] for c in self.cells if not c["edge"] and c["parent"] == "1"),
                    default=1000)
        bottom = max((c["y"] + c["h"] for c in self.cells if not c["edge"] and c["parent"] == "1"),
                     default=400)
        return self.free_note("legend", 40, bottom + 60, right - 40, 130, LEGEND_TEXT)

    def edge(self, src, tgt, kind="uses", label="", exit_right=False):
        style = EDGE_STYLES[kind]
        if exit_right:
            style += "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
        self.cells.append(dict(id=self._next_id("e"), value=label, style=style,
                               x=0, y=0, w=0, h=0, parent="1", edge=True,
                               source=src, target=tgt))

    # -- serialization ------------------------------------------------------
    def to_xml(self, diagram_id):
        diagram = ET.Element("diagram", {"id": diagram_id, "name": self.name})
        model = ET.SubElement(diagram, "mxGraphModel", {
            "dx": "1600", "dy": "1000", "grid": "1", "gridSize": "10", "guides": "1",
            "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
            "pageScale": "1", "pageWidth": "1169", "pageHeight": "826",
            "math": "0", "shadow": "0",
        })
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

        # heading
        head = ET.SubElement(root, "mxCell", {
            "id": "heading", "value": self.heading, "style": HEADING_STYLE,
            "vertex": "1", "parent": "1"})
        ET.SubElement(head, "mxGeometry", {
            "x": "40", "y": "30", "width": "1000", "height": "30", "as": "geometry"})
        if self.subheading:
            sub = ET.SubElement(root, "mxCell", {
                "id": "subheading", "value": self.subheading, "style": SUBHEADING_STYLE,
                "vertex": "1", "parent": "1"})
            ET.SubElement(sub, "mxGeometry", {
                "x": "40", "y": "64", "width": "1000", "height": "60", "as": "geometry"})

        for c in self.cells:
            attrs = {"id": c["id"], "value": c["value"], "style": c["style"], "parent": c["parent"]}
            if c["edge"]:
                attrs["edge"] = "1"
                attrs["source"] = c["source"]
                attrs["target"] = c["target"]
                cell = ET.SubElement(root, "mxCell", attrs)
                ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
            else:
                attrs["vertex"] = "1"
                cell = ET.SubElement(root, "mxCell", attrs)
                ET.SubElement(cell, "mxGeometry", {
                    "x": str(c["x"]), "y": str(c["y"]),
                    "width": str(c["w"]), "height": str(c["h"]), "as": "geometry"})
        return diagram


def write_mxfile(path, pages):
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "type": "device", "version": "24.7.17"})
    for i, page in enumerate(pages):
        mxfile.append(page.to_xml(f"page-{i}"))
    raw = ET.tostring(mxfile, encoding="utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="UTF-8")
    with open(path, "wb") as f:
        f.write(pretty)
    print(f"wrote {path}")


LEGEND_TEXT = (
    "<b>Legend</b><br>"
    "Colours: yellow = SLURM/bash job script &#183; orange = python entry point &#183; "
    "blue = <i>src/</i> helper module &#183; purple = <i>eap/</i> circuit package &#183; "
    "green = on-disk artifact &#183; grey = third-party class.<br>"
    "Arrows: dashed open = «use» (calls/imports) &#183; solid green = data flow (reads/writes) &#183; "
    "hollow triangle = generalization &#183; filled diamond = composition.<br>"
    "Boxes stereotyped «module» are Python modules drawn as UML utility classes: the repo is mostly "
    "function-based, so module-level functions are shown as the module's operations."
)

# ============================================================== page 0 ======

def page_overview():
    p = Page("0. Pipeline Overview",
             "ABSA Circuit Fine-Tuning Pipeline — Overview",
             "Execution order is Full SFT &#8594; Dataset Creation &#8594; Circuit Discovery &#8594; "
             "Selective Circuit-Based SFT &#8594; Evaluation: dataset creation needs a fully fine-tuned "
             "model to filter on, and circuit discovery attributes through that same model.")
    for _ in range(5):
        p.add_column(300, top=150)

    # --- stage 2: Full SFT
    p.cls("ov-sh-sft", 0, "job script", "scripts/sft.sh",
          attrs=["SEEDS = 9584 123 2024 31415 777"],
          methods=["run_sft.py --train_full_model"], kind="script")
    p.cls("ov-sft", 0, "entry point", "run_sft.py",
          attrs=["+ train_json_path, model_name", "+ lr, batch_size, num_epochs, seed",
                 "+ train_full_model = True"],
          methods=["+ main(args)"], kind="entry")
    p.cls("ov-sft-out", 0, "artifact", "outputs/models/&#8230;/&lt;run&gt;/",
          attrs=["model.pt", "model_config.pkl", "tokenizer files"], kind="data")

    # --- stage 1: Dataset Creation
    p.cls("ov-sh-ds", 1, "job script", "scripts/create_dataset.sh",
          attrs=["SEEDS = 9584 123 2024 31415 777"],
          methods=["run_create_dataset.py"], kind="script")
    p.cls("ov-ds", 1, "entry point", "run_create_dataset.py",
          attrs=["+ finetuned_model, dataset_path", "+ method: mvp | mvp_aos | gas | gas_bar",
                 "+ filter_data: bool"],
          methods=["+ main(args)"], kind="entry")
    p.cls("ov-ds-out", 1, "artifact", "eap_dataset/&#8230;/*.csv",
          attrs=["clean, corrupted", "correct_label, incorrect_label",
                 "correct_idx, incorrect_idx"], kind="data")

    # --- stage 3: Circuit Discovery
    p.cls("ov-sh-cd", 2, "job script", "scripts/circuit_discovery.sh",
          attrs=["SEEDS, topks = 1000 2000 5000"],
          methods=["run_eap_multitokens.py"], kind="script")
    p.cls("ov-cd", 2, "entry point", "run_eap_multitokens.py",
          attrs=["+ finetuned_model, dataset", "+ ig_steps, topks, element",
                 "+ output_dir, batch_size"],
          methods=["+ main(args)", "+ log_results_csv(&#8230;)"], kind="entry")
    p.cls("ov-cd-out", 2, "artifact", "outputs/multitokens/seed_&lt;s&gt;/",
          attrs=["aos_circuit_topk-&lt;k&gt;.pt", "aos_circuit_topk-&lt;k&gt;.csv",
                 "faithfulness_log.csv"], kind="data")

    # --- stage 4: Selective Circuit SFT
    p.cls("ov-sh-csft", 3, "job script", "scripts/sft_circuit.sh",
          attrs=["SEEDS &#215; TOPKS = 1000 2000 5000"],
          methods=["run_sft.py --circuit_csv_path"], kind="script")
    p.cls("ov-csft", 3, "entry point", "run_sft.py",
          attrs=["+ circuit_csv_path", "+ random_circuit (control)",
                 "+ train_full_model = False"],
          methods=["+ main(args)"], kind="entry")
    p.cls("ov-csft-out", 3, "artifact", "outputs/models/&#8230;_topk-&lt;k&gt;/",
          attrs=["model.pt", "model_config.pkl", "tokenizer files"], kind="data")

    # --- stage 5: Evaluation
    p.cls("ov-sh-ev", 4, "job script", "scripts/eval.sh",
          attrs=["SEEDS, TEST_JSON"],
          methods=["run_eval.py (per model dir)"], kind="script")
    p.cls("ov-ev", 4, "entry point", "run_eval.py",
          attrs=["+ test_json_path, model_path", "+ prompt_type: mvp | gas",
                 "+ save_predictions"],
          methods=["+ main(args)"], kind="entry")
    p.cls("ov-ev-out", 4, "artifact", "outputs/evals/&#8230;/",
          attrs=["evaluation_results.json", "inference_results.json",
                 "raw_inference_results.json"], kind="data")

    # shared libraries
    shared_y = 640
    p.cls("ov-lib-utils", 0, "module", "src.utils", y=shared_y,
          attrs=["model loading &#183; dataset building", "circuit masking &#183; metrics"], kind="util")
    p.cls("ov-lib-train", 1, "module", "src.train", y=shared_y,
          attrs=["HookedTransformerTrainConfig", "train(&#8230;)"], kind="util")
    p.cls("ov-lib-eap", 2, "package", "eap", y=shared_y,
          attrs=["graph &#183; attribute &#183; evaluate", "attribute_node &#183; visualization"], kind="eap")
    p.cls("ov-lib-sampling", 3, "module", "src.sampling", y=shared_y,
          attrs=["difficulty annotation", "stratified sampling"], kind="util")
    p.cls("ov-lib-metric", 4, "module", "src.metric", y=shared_y,
          attrs=["logit_diff(&#8230;)", "get_logit_positions(&#8230;)"], kind="util")

    for sh, entry, out in [("ov-sh-sft", "ov-sft", "ov-sft-out"),
                           ("ov-sh-ds", "ov-ds", "ov-ds-out"),
                           ("ov-sh-cd", "ov-cd", "ov-cd-out"),
                           ("ov-sh-csft", "ov-csft", "ov-csft-out"),
                           ("ov-sh-ev", "ov-ev", "ov-ev-out")]:
        p.edge(sh, entry, "invokes", "sbatch")
        p.edge(entry, out, "flow", "writes")

    # cross-stage data dependencies
    p.edge("ov-sft-out", "ov-ds", "flow", "fine-tuned model")
    p.edge("ov-sft-out", "ov-cd", "flow", "model under study")
    p.edge("ov-ds-out", "ov-cd", "flow", "clean/corrupted pairs")
    p.edge("ov-cd-out", "ov-csft", "flow", "circuit edge list")
    p.edge("ov-csft-out", "ov-ev", "flow", "checkpoints")
    p.edge("ov-sft-out", "ov-ev", "flow", "baseline checkpoint")

    p.edge("ov-sft", "ov-lib-utils", "uses")
    p.edge("ov-sft", "ov-lib-train", "uses")
    p.edge("ov-sft", "ov-lib-sampling", "uses")
    p.edge("ov-ds", "ov-lib-utils", "uses")
    p.edge("ov-cd", "ov-lib-eap", "uses")
    p.edge("ov-cd", "ov-lib-metric", "uses")
    p.edge("ov-csft", "ov-lib-utils", "uses")
    p.edge("ov-ev", "ov-lib-utils", "uses")

    p.add_legend()
    return p


# ============================================================== page 1 ======

def page_dataset_creation():
    p = Page("1. Dataset Creation",
             "1. Dataset Creation — run_create_dataset.py",
             "Turns raw counterfactual pairs into an EAP-IG dataset of (clean, corrupted) prompt pairs "
             "with matching-length correct/incorrect label token ids. Two variants: the MVP/AOS pipeline "
             "(<i>[A] [O] [S]</i> markers, five tag orders) and the simpler GAS pipeline "
             "(<i>sentence =&gt; ( a | o | s )</i>).")
    p.add_column(300, top=170)   # 0: driver
    p.add_column(430, top=170)   # 1: MVP steps
    p.add_column(430, top=170)   # 2: GAS steps + helpers
    p.add_column(370, top=170)   # 3: artifacts

    p.cls("dc-sh", 0, "job script", "scripts/create_dataset.sh",
          attrs=["SEEDS = 9584 123 2024 31415 777",
                 "MODEL_PARENT_DIR",
                 "FINETUNED_MODEL (latest full run)"],
          methods=["python run_create_dataset.py &#8230;"], kind="script")

    p.cls("dc-entry", 0, "entry point", "run_create_dataset.py",
          attrs=["+ finetuned_model: str",
                 "+ dataset_path: str",
                 "+ filtered_data_path: str",
                 "+ full_aos_path: str",
                 "+ sequence_variants_path: str",
                 "+ eap_output_path: str",
                 "+ method: mvp|mvp_aos|gas|gas_bar",
                 "+ filter_data: bool"],
          methods=["+ main(args)"], kind="entry")

    p.cls("dc-load", 0, "module", "src.utils  (model loading)",
          methods=["+ load_finetuned_model_lens_from_dir(dir, device)",
                   "&#160;&#160;&#160;&#160;: HookedTransformer"], kind="util")

    p.cls("dc-model", 0, "external", "HookedTransformer",
          attrs=["+ cfg: HookedTransformerConfig", "+ tokenizer"],
          methods=["+ generate(input, max_new_tokens, &#8230;)",
                   "+ to_tokens(text) : Tensor",
                   "+ to_string(tokens) : str"], kind="external")

    # --- MVP branch (column 1)
    p.cls("dc-mvp", 1, "module", "src.utils  —  MVP / AOS pipeline",
          methods=[
              "+ format_counterfactuals(input_path) : DataFrame",
              "&#160;&#160;&#160;&#160;step 0 — split 'original_pair' on [A] [O] [S]",
              "+ filter_correct_data(model, data, sentence_col, label_col,",
              "&#160;&#160;&#160;&#160;max_tokens=150, filter_only_correct=True,",
              "&#160;&#160;&#160;&#160;filter_mode='AOS', save_path) : DataFrame",
              "&#160;&#160;&#160;&#160;step 1 — keep rows the model already predicts correctly",
              "+ create_full_AOS_dataset(dataset_path) : DataFrame",
              "&#160;&#160;&#160;&#160;step 2 — aspect of cf1 + opinion/sentiment of cf3",
              "&#160;&#160;&#160;&#160;&#8594; counterfact4_replaced (simultaneous AOS)",
              "+ create_aos_sequence_variant(dataset_path) : DataFrame",
              "&#160;&#160;&#160;&#160;step 3 — one row per tag order",
              "+ build_eap_dataset(model, df, sentence_col, triplet_col,",
              "&#160;&#160;&#160;&#160;corrupted_col, corrupted_triplet_col, suffix,",
              "&#160;&#160;&#160;&#160;filer_same_length_counterfactuals=True) : DataFrame",
              "&#160;&#160;&#160;&#160;step 4 — tokenize + drop length-mismatched pairs",
          ], kind="util")

    p.cls("dc-helpers", 1, "module", "src.utils  —  triplet helpers",
          methods=["+ convert_triplet_string(triplet_str) : tuple",
                   "+ extract_triplet_fixed(text) : tuple | None",
                   "+ create_sequences(a, o, s) : Dict[str, str]",
                   "&#160;&#160;&#160;&#160;AOS &#183; ASO &#183; SAO &#183; OAS &#183; OSA",
                   "+ get_tag_suffix(order) : str",
                   "+ format_by_mode(aspect, opinion, sentiment, mode) : str",
                   "+ build_suffix_from_mode(mode) : str",
                   "+ extract_by_mode(text, mode) : str"], kind="util")

    # --- GAS branch (column 2)
    p.cls("dc-gas", 2, "module", "src.utils  —  GAS pipeline",
          methods=[
              "+ format_counterfactuals_gas(input_path, col_original,",
              "&#160;&#160;&#160;&#160;col_counter, index_col_name,",
              "&#160;&#160;&#160;&#160;coerce_index_to_int=True) : DataFrame",
              "+ filter_correct_data_gas(model, data, sentence_col,",
              "&#160;&#160;&#160;&#160;label_col, max_tokens=60,",
              "&#160;&#160;&#160;&#160;filter_only_correct=True, save_path) : DataFrame",
              "+ build_eap_dataset_gas(model, df, sentence_col, triplet_col,",
              "&#160;&#160;&#160;&#160;corrupted_col, corrupted_triplet_col,",
              "&#160;&#160;&#160;&#160;filter_same_length_counterfactuals=True) : DataFrame",
              "- _extract_first_triplet(text) : str | None",
              "- _normalize_triplet_str(s) : str | None",
              "&#160;&#160;&#160;&#160;normalises to '( a | o | s )'",
          ], kind="util")

    p.cls("dc-notused", 2, "module", "src.utils  —  also defined here",
          methods=["+ append_labels(model, clean, corrupted, labels)",
                   "&#160;&#160;&#160;&#160;multi-token label prefixing (currently unused)",
                   "+ safe_parse(raw) &#183; collate_EAP(batch)",
                   "&#160;&#160;&#160;&#160;consumed later by circuit discovery"], kind="util")

    p.note("dc-note", 2,
           "<b>Pipeline variants</b><br>"
           "<b>mvp</b> — all five tag orders kept.<br>"
           "<b>mvp_aos</b> — same, then filtered down to <i>order == 'AOS'</i>.<br>"
           "<b>gas / gas_bar</b> — no A/O/S modes; steps 2 and 3 are skipped and the whole "
           "triplet string is the label.<br>"
           "<b>--filter_data</b> — when omitted, every row is marked <i>is_match = True</i> "
           "and the generation-based filter is skipped.",
           height=150)

    # --- artifacts (column 3)
    p.cls("dc-a0", 3, "artifact", "hotel_dataset/&#8230;/counterfacts.csv",
          attrs=["original_pair | original_sentence",
                 "corrupted_pair | counterfact&#8230;",
                 "original_triplet"], kind="data")
    p.cls("dc-a1", 3, "artifact", "&#8230;_counterfactual_filtered_AOS.csv",
          attrs=["original_sentence, original_triplet",
                 "original_label, inference",
                 "is_match: bool"], kind="data")
    p.cls("dc-a2", 3, "artifact", "&#8230;_counterfactual_full_AOS.csv",
          attrs=["+ counterfact4_replaced",
                 "+ counterfact_triplet4_replaced"], kind="data")
    p.cls("dc-a3", 3, "artifact", "&#8230;_AOS_sequence_variants.csv",
          attrs=["order: AOS|ASO|SAO|OAS|OSA",
                 "original_label_variant",
                 "counterfact_label_variant"], kind="data")
    p.cls("dc-a4", 3, "artifact", "eap_dataset/&#8230;.csv",
          attrs=["clean, corrupted",
                 "correct_label, incorrect_label",
                 "correct_idx, incorrect_idx"], kind="data")

    p.edge("dc-sh", "dc-entry", "invokes", "sbatch")
    p.edge("dc-entry", "dc-load", "uses")
    p.edge("dc-load", "dc-model", "flow", "returns")
    p.edge("dc-entry", "dc-mvp", "uses", "method = mvp / mvp_aos")
    p.edge("dc-entry", "dc-gas", "uses", "method = gas / gas_bar")
    p.edge("dc-mvp", "dc-helpers", "uses")
    p.edge("dc-mvp", "dc-model", "uses", "generate / to_tokens")
    p.edge("dc-gas", "dc-model", "uses", "generate / to_tokens")
    p.edge("dc-a0", "dc-mvp", "flow", "read")
    p.edge("dc-mvp", "dc-a1", "flow", "filter_correct_data")
    p.edge("dc-a1", "dc-a2", "flow", "create_full_AOS_dataset")
    p.edge("dc-a2", "dc-a3", "flow", "create_aos_sequence_variant")
    p.edge("dc-a3", "dc-a4", "flow", "build_eap_dataset")

    p.add_legend()
    return p


# ============================================================== page 2 ======

def page_full_sft():
    p = Page("2. Full SFT",
             "2. Full SFT — run_sft.py --train_full_model",
             "Supervised fine-tuning of every parameter of the base model on the ABSA training set, "
             "producing the reference model that dataset creation filters with and that circuit "
             "discovery attributes through. Optional stratified sub-sampling controls the training budget.")
    p.add_column(300, top=170)   # 0: driver
    p.add_column(400, top=170)   # 1: model + data classes
    p.add_column(430, top=170)   # 2: training
    p.add_column(400, top=170)   # 3: sampling
    p.add_column(350, top=170)   # 4: artifacts

    p.cls("fs-sh", 0, "job script", "scripts/sft.sh",
          attrs=["SEEDS = 9584 123 2024 31415 777"],
          methods=["python run_sft.py --train_full_model"], kind="script")

    p.cls("fs-entry", 0, "entry point", "run_sft.py",
          attrs=["+ train_json_path, val_json_path",
                 "+ inference_train_json_path",
                 "+ model_name = 'Qwen/Qwen2.5-0.5B'",
                 "+ output_dir, save_mode",
                 "+ num_epochs, batch_size, lr",
                 "+ optimizer: AdamW|Adam|SGD",
                 "+ seed, sample_size, prompt_type",
                 "+ train_full_model = True",
                 "+ subtract_data_by",
                 "+ finetuned_model_path"],
          methods=["+ main(args)"], kind="entry")

    p.note("fs-note", 0,
           "<b>Output folder name</b> encodes the run: "
           "<i>&lt;timestamp&gt;_tflens_&lt;train file&gt;_model-&lt;name&gt;_lr-&lt;lr&gt;_bs-&lt;bs&gt;"
           "_epochs-&lt;n&gt;[_subtracted&lt;k&gt;][_n&lt;sample&gt;_&lt;prompt_type&gt;]</i>. "
           "Circuit runs append <i>topk-&lt;k&gt;</i> instead — that is how the shell scripts tell "
           "full runs from circuit runs.", height=140)

    # --- model + dataset
    p.cls("fs-load", 1, "module", "src.utils  —  model loading",
          methods=["+ get_cfg_dict(base_model_name, hf_config) : dict",
                   "&#160;&#160;&#160;&#160;Qwen2 and Bloom layouts only",
                   "+ load_model(base_model_name, fine_tuned_model_path,",
                   "&#160;&#160;&#160;&#160;device) : HookedTransformer",
                   "+ load_finetuned_model(base_model_name,",
                   "&#160;&#160;&#160;&#160;fine_tuned_model_path, device)",
                   "&#160;&#160;&#160;&#160;converts HF weights &#8594; TransformerLens",
                   "+ load_finetuned_model_lens_from_dir(dir, device)",
                   "&#160;&#160;&#160;&#160;reloads model.pt + model_config.pkl"], kind="util")

    p.cls("fs-ds", 1, "class", "ABSAAutoRegressiveDataset",
          attrs=["- data: List[dict]",
                 "- tokenizer",
                 "- max_len: int = 300"],
          methods=["+ __init__(data, tokenizer, max_len=300,",
                   "&#160;&#160;&#160;&#160;shuffle=False, seed=42, sample_size=None)",
                   "+ __len__() : int",
                   "+ __getitem__(idx) : {'tokens': Tensor}",
                   "&#160;&#160;&#160;&#160;input + ' ' + target + eos, padded to max_len"], kind="util")

    p.cls("fs-torchds", 1, "external", "torch.utils.data.Dataset", kind="external",
          attrs=["(PyTorch map-style dataset)"])

    p.cls("fs-model", 1, "external", "HookedTransformer",
          attrs=["+ cfg &#183; tokenizer &#183; blocks[]"],
          methods=["+ forward(tokens, return_type='loss')",
                   "+ state_dict() &#183; to(device)"], kind="external")

    # --- training
    p.cls("fs-cfg", 2, "dataclass", "HookedTransformerTrainConfig",
          attrs=["+ num_epochs: int",
                 "+ batch_size: int &#183; val_batch_size: int",
                 "+ lr: float = 1e-4 &#183; seed: int",
                 "+ momentum: float &#183; weight_decay: float",
                 "+ max_grad_norm: float",
                 "+ optimizer_name: str = 'Adam'",
                 "+ device: str &#183; warmup_steps: int",
                 "+ save_mode: 'best'|'steps'|None",
                 "+ save_every: int &#183; save_dir: str",
                 "+ validation_mode: 'epoch'|'steps'|None",
                 "+ validation_steps: int",
                 "+ wandb: bool &#183; wandb_project",
                 "+ wandb_run_name &#183; print_every &#183; max_steps",
                 "+ top_k &#183; sample_size &#183; subtract_data_amount",
                 "&#160;&#160;&#160;&#160;(last three are logging-only)"], kind="util")

    p.cls("fs-train", 2, "module", "src.train",
          methods=["+ train(model, config, dataset, val_dataset)",
                   "&#160;&#160;&#160;&#160;: HookedTransformer",
                   "&#160;&#160;&#160;&#160;&#183; builds Adam / AdamW / SGD + LambdaLR warm-up",
                   "&#160;&#160;&#160;&#160;&#183; per-step loss = model(tokens, return_type='loss')",
                   "&#160;&#160;&#160;&#160;&#183; save_mode='best' &#8594; save on best epoch loss",
                   "&#160;&#160;&#160;&#160;&#183; validation per epoch or per N steps",
                   "&#160;&#160;&#160;&#160;&#183; logs train_loss / val_loss to Weights &amp; Biases"], kind="util")

    p.cls("fs-wandb", 2, "external", "wandb", kind="external",
          methods=["+ init(project, config, name)", "+ log(dict)"])

    # --- sampling
    p.cls("fs-samp-mvp", 3, "module", "src.sampling  —  MVP",
          methods=["+ add_element_order_to_json(json_path, backup=True)",
                   "&#160;&#160;&#160;&#160;derives element_order from the [A][O][S] suffix",
                   "+ annotate_difficulty(input_path, output_path,",
                   "&#160;&#160;&#160;&#160;difficulty_method, bins=3, min_easy_f1)",
                   "&#160;&#160;&#160;&#160;: (annotated, thresholds)",
                   "+ get_difficulty_assigner(f1_scores, method, bins,",
                   "&#160;&#160;&#160;&#160;fixed_thresholds, min_easy_f1)",
                   "+ stratified_sample_by_factors(full_data_path,",
                   "&#160;&#160;&#160;&#160;output_path, factors, target_total_samples,",
                   "&#160;&#160;&#160;&#160;permutations_per_sentence, seed, plot)",
                   "+ extract_stratification_key(item, factors, config)",
                   "+ get_triplet_count &#183; get_input_length_bin",
                   "+ get_sentiment_bin &#183; get_sentiment_ratio_bin"], kind="util")

    p.cls("fs-samp-gas", 3, "module", "src.sampling  —  GAS",
          methods=["+ add_element_order_to_json_gas(json_path)",
                   "+ annotate_difficulty_gas(&#8230;)",
                   "+ get_difficulty_assigner_gas(&#8230;)",
                   "+ stratified_sample_by_factors_gas(&#8230;)",
                   "+ extract_stratification_key_gas(&#8230;)",
                   "+ get_triplet_count_gas &#183; get_input_length_bin_gas",
                   "+ get_sentiment_bin_gas &#183; get_sentiment_ratio_bin_gas",
                   "- _triplet_list_from_item &#183; _sentiments_from_item"], kind="util")

    p.cls("fs-samp-viz", 3, "module", "src.sampling  —  strata plots",
          methods=["+ visualize_strata_heatmap_from_meta(&#8230;)",
                   "+ visualize_strata_heatmap_from_meta_combo_xy(&#8230;)",
                   "+ visualize_strata_side_by_side(&#8230;)",
                   "&#160;&#160;&#160;&#160;(+ _gas twins; only when plot=True)"], kind="util")

    p.cls("fs-metrics", 3, "module", "src.utils  —  metrics",
          methods=["+ calculate_metrics(predictions, targets, task)",
                   "&#160;&#160;&#160;&#160;: {precision_&lt;task&gt;, recall_&lt;task&gt;, f1_&lt;task&gt;}",
                   "&#160;&#160;&#160;&#160;used to score difficulty bins"], kind="util")

    # --- artifacts
    p.cls("fs-in", 4, "artifact", "hotel_aste_train_augmented&#8230;.json",
          attrs=["sentence_id, input, target",
                 "task_elements, element_order"], kind="data")
    p.cls("fs-ann", 4, "artifact", "&lt;n&gt;_mvp_full_annotated.json",
          attrs=["+ precision, recall, f1",
                 "+ difficulty: easy|medium|hard"], kind="data")
    p.cls("fs-samp", 4, "artifact", "&lt;n&gt;_mvp_sample.json",
          attrs=["stratified subset of sentences"], kind="data")
    p.cls("fs-out", 4, "artifact", "outputs/models/&#8230;/&lt;run&gt;/",
          attrs=["model.pt",
                 "model_config.pkl",
                 "tokenizer.json, vocab.json, &#8230;"], kind="data")

    p.edge("fs-sh", "fs-entry", "invokes", "sbatch")
    p.edge("fs-entry", "fs-load", "uses")
    p.edge("fs-entry", "fs-ds", "uses", "creates")
    p.edge("fs-ds", "fs-torchds", "extends")
    p.edge("fs-load", "fs-model", "flow", "returns")
    p.edge("fs-entry", "fs-cfg", "uses", "builds")
    p.edge("fs-entry", "fs-train", "uses", "train(model, config, dataset)")
    p.edge("fs-train", "fs-cfg", "uses", "reads")
    p.edge("fs-train", "fs-ds", "uses", "DataLoader")
    p.edge("fs-train", "fs-model", "uses", "forward / backward")
    p.edge("fs-train", "fs-wandb", "uses")
    p.edge("fs-entry", "fs-samp-mvp", "uses", "prompt_type = mvp, sample_size set")
    p.edge("fs-entry", "fs-samp-gas", "uses", "prompt_type = gas")
    p.edge("fs-samp-mvp", "fs-samp-viz", "uses", "plot=True")
    p.edge("fs-samp-mvp", "fs-metrics", "uses", "F1 per item")
    p.edge("fs-samp-gas", "fs-metrics", "uses")
    p.edge("fs-in", "fs-samp-mvp", "flow", "read")
    p.edge("fs-samp-mvp", "fs-ann", "flow", "annotate_difficulty")
    p.edge("fs-ann", "fs-samp", "flow", "stratified_sample_by_factors")
    p.edge("fs-train", "fs-out", "flow", "writes")

    p.add_legend()
    return p


# ============================================================== page 3 ======

def page_circuit_discovery():
    p = Page("3. Circuit Discovery",
             "3. Circuit Discovery — run_eap_multitokens.py (EAP-IG)",
             "Edge Attribution Patching with Integrated Gradients over the fine-tuned model. Every "
             "candidate edge of the transformer's computational graph is scored by how much it moves "
             "<i>logit_diff</i> between the clean and corrupted prompt; the top-k edges form the circuit, "
             "whose faithfulness is then measured by re-running the model with all other edges patched "
             "to their corrupted activations.")
    p.add_column(310, top=180)   # 0: driver
    p.add_column(440, top=180)   # 1: graph model
    p.add_column(450, top=180)   # 2: attribution
    p.add_column(420, top=180)   # 3: evaluation + metric
    p.add_column(360, top=180)   # 4: artifacts

    p.cls("cd-sh", 0, "job script", "scripts/circuit_discovery.sh",
          attrs=["SEEDS = 9584 123 2024 31415 777",
                 "topks = 1000 2000 5000",
                 "ig_steps = 5, batch_size = 5",
                 "element = 'aos'"],
          methods=["python run_eap_multitokens.py &#8230;"], kind="script")

    p.cls("cd-entry", 0, "entry point", "run_eap_multitokens.py",
          attrs=["+ base_model, finetuned_model",
                 "+ dataset (EAP csv)",
                 "+ batch_size, ig_steps",
                 "+ topks: List[int]",
                 "+ output_dir, device, log_file",
                 "+ element: aspect|opinion|sentiment|aos"],
          methods=["+ main(args)",
                   "+ log_results_csv(log_path, element, metric,",
                   "&#160;&#160;&#160;&#160;total_edges, baseline_score, top_k,",
                   "&#160;&#160;&#160;&#160;score, faithfulness)"], kind="entry")

    p.note("cd-note-cfg", 0,
           "<b>Required model flags</b> set before attribution:<br>"
           "<i>use_split_qkv_input</i>, <i>use_attn_result</i>, <i>use_hook_mlp_in</i>, "
           "<i>ungroup_grouped_query_attention</i>. <i>attribute()</i> asserts all four — "
           "without them the per-head hooks have nothing to attach to.", height=120)

    p.cls("cd-entry-old", 0, "entry point", "run_eap.py",
          attrs=["earlier single-element prototype",
                 "local EAPDataset / prob_diff_multitoken",
                 "hard-coded paths, mps device"],
          methods=["+ filter_dataset() &#183; main()"], kind="entry")

    # --- graph model
    p.cls("cd-graph", 1, "class", "eap.graph.Graph",
          attrs=["+ nodes: Dict[str, Node]",
                 "+ edges: Dict[str, Edge]",
                 "+ n_forward, n_backward: int",
                 "+ scores: Tensor[n_fwd, n_bwd]",
                 "+ in_graph: Tensor[n_fwd, n_bwd]",
                 "+ real_edge_mask: Tensor",
                 "+ nodes_in_graph, nodes_scores: Tensor",
                 "+ neurons_in_graph, neurons_scores: Tensor",
                 "+ cfg: GraphConfig"],
          methods=["+ from_model(model_or_config, neuron_level,",
                   "&#160;&#160;&#160;&#160;node_scores) : Graph  &lt;&lt;factory&gt;&gt;",
                   "+ from_pt(path) &#183; from_json(path)  &lt;&lt;factory&gt;&gt;",
                   "+ add_edge(parent, child, qkv)",
                   "+ forward_index(node) &#183; backward_index(node, qkv)",
                   "+ prev_index(node) &#183; get_dst_nodes()",
                   "+ apply_topn(n, absolute, level, reset, prune)",
                   "+ apply_threshold(threshold, &#8230;)",
                   "+ apply_greedy(n_edges, &#8230;)",
                   "+ prune() &#183; reset(empty=True)",
                   "+ count_included_edges() : int",
                   "+ count_included_nodes() : int",
                   "+ count_included_neurons() : int",
                   "+ weighted_edge_count() : float",
                   "+ to_pt(path) &#183; to_json(path)",
                   "+ to_graphviz(filename, &#8230;)"], kind="eap")

    p.cls("cd-node", 1, "class", "eap.graph.Node",
          attrs=["+ name, layer",
                 "+ in_hook, out_hook, index",
                 "+ qkv_inputs: List[str]",
                 "+ parents, children: Set[Node]",
                 "+ parent_edges, child_edges: Set[Edge]",
                 "/ in_graph, score  (views on Graph tensors)"], kind="eap")

    p.cls("cd-node-subs", 1, "classes", "Node subclasses",
          attrs=["InputNode&#160;&#160;&#8594; 'input', hook_embed",
                 "AttentionNode &#8594; 'a&lt;L&gt;.h&lt;H&gt;', per-head q/k/v hooks",
                 "MLPNode&#160;&#160;&#160;&#8594; 'm&lt;L&gt;', hook_mlp_in/out",
                 "LogitNode&#160;&#160;&#8594; 'logits', always in graph"], kind="eap")

    p.cls("cd-edge", 1, "class", "eap.graph.Edge",
          attrs=["+ name: '&lt;parent&gt;-&gt;&lt;child&gt;&lt;qkv&gt;'",
                 "+ parent, child: Node",
                 "+ qkv: 'q' | 'k' | 'v' | None",
                 "+ hook, index, matrix_index",
                 "/ score, in_graph  (views on Graph tensors)"], kind="eap")

    p.cls("cd-gcfg", 1, "class", "eap.graph.GraphConfig(dict)",
          attrs=["n_layers, n_heads, d_model",
                 "parallel_attn_mlp"], kind="eap")

    # --- attribution
    p.cls("cd-attr", 2, "module", "eap.attribute",
          methods=["+ attribute(model, graph, dataloader, metric, method,",
                   "&#160;&#160;&#160;&#160;intervention='patching', aggregation='sum',",
                   "&#160;&#160;&#160;&#160;ig_steps, is_absa=False, batch_size, device)",
                   "&#160;&#160;&#160;&#160;&lt;&lt;dispatcher&gt;&gt; writes into graph.scores",
                   "+ modified_get_scores_eap_ig(model, graph, df, metric,",
                   "&#160;&#160;&#160;&#160;batch_size, steps, quiet, device) : Tensor",
                   "&#160;&#160;&#160;&#160;ABSA path — groups rows by label length and",
                   "&#160;&#160;&#160;&#160;attributes one target token at a time",
                   "+ get_scores_eap(&#8230;)  — plain EAP",
                   "+ get_scores_eap_ig(&#8230;) — non-ABSA EAP-IG",
                   "+ get_scores_ig_activations(&#8230;)",
                   "+ get_scores_clean_corrupted(&#8230;)",
                   "+ make_hooks_and_matrices(model, graph, batch_size,",
                   "&#160;&#160;&#160;&#160;n_pos, scores)",
                   "&#160;&#160;&#160;&#160;&#8594; (fwd_corrupted, fwd_clean, bwd), act_diff",
                   "&#160;&#160;&#160;&#160;- activation_hook(index, acts, hook, add)",
                   "&#160;&#160;&#160;&#160;- gradient_hook(prev_index, bwd_index, grads, hook)",
                   "+ tokenize_plus(model, inputs, max_length)",
                   "&#160;&#160;&#160;&#160;&#8594; tokens, attention_mask, input_lengths, n_pos",
                   "+ compute_mean_activations(model, graph, dataloader,",
                   "&#160;&#160;&#160;&#160;per_position)  — for mean interventions"], kind="eap")

    p.cls("cd-attr-node", 2, "module", "eap.attribute_node",
          methods=["+ attribute_node(model, graph, dataloader, metric,",
                   "&#160;&#160;&#160;&#160;method, intervention, aggregation, ig_steps,",
                   "&#160;&#160;&#160;&#160;neuron=False)",
                   "+ get_scores_eap / _eap_ig / _ig_activations /",
                   "&#160;&#160;&#160;&#160;_clean_corrupted  (node- and neuron-level twins)",
                   "&#160;&#160;&#160;&#160;not used by the current pipeline scripts"], kind="eap")

    p.cls("cd-eapds", 2, "class", "src.utils.EAPDataset",
          attrs=["- df: DataFrame"],
          methods=["+ __len__() &#183; __getitem__(idx)",
                   "&#160;&#160;&#160;&#160;&#8594; (clean, corrupted, [correct_idx, incorrect_idx])",
                   "+ shuffle() &#183; head(n)",
                   "+ to_dataloader(batch_size) : DataLoader",
                   "&#160;&#160;&#160;&#160;collate_fn = collate_EAP"], kind="util")

    p.cls("cd-eaphelp", 2, "module", "src.utils  —  EAP helpers",
          methods=["+ collate_EAP(batch) : (clean, corrupted, labels)",
                   "+ safe_parse(raw) : list",
                   "+ edge_merging(graph_paths) : DataFrame",
                   "&#160;&#160;&#160;&#160;in-graph edges &#8594; parent_node, child_node, child_type",
                   "&#160;&#160;&#160;&#160;this CSV is the hand-off to selective SFT"], kind="util")

    # --- evaluation of the circuit
    p.cls("cd-eval", 3, "module", "eap.evaluate",
          methods=["+ evaluate_baseline_multitoken(model, df, metrics,",
                   "&#160;&#160;&#160;&#160;run_corrupted=False, quiet, batch_size)",
                   "&#160;&#160;&#160;&#160;unpatched score, token-step by token-step",
                   "+ evaluate_graph_multitoken(model, graph, df, metrics,",
                   "&#160;&#160;&#160;&#160;batch_size, intervention='patching',",
                   "&#160;&#160;&#160;&#160;intervention_dataloader, skip_clean)",
                   "&#160;&#160;&#160;&#160;rebuilds every node input from in-graph edges only",
                   "&#160;&#160;&#160;&#160;- make_input_construction_hook(s)(&#8230;)",
                   "+ evaluate_baseline(model, dataloader, metrics, &#8230;)",
                   "+ evaluate_graph(model, graph, dataloader, metrics, &#8230;)",
                   "&#160;&#160;&#160;&#160;single-token originals"], kind="eap")

    p.cls("cd-metric", 3, "module", "src.metric",
          methods=["+ logit_diff(logits, clean_logits, input_length,",
                   "&#160;&#160;&#160;&#160;labels, mean=True, loss=False) : Tensor",
                   "&#160;&#160;&#160;&#160;logit(correct) &#8722; logit(incorrect) at the answer position",
                   "+ get_logit_positions(logits, input_length) : Tensor"], kind="util")

    p.cls("cd-viz", 3, "module", "eap.visualization",
          methods=["+ get_color(qkv, score) : str",
                   "+ generate_random_color(colorscheme) : str",
                   "+ cmap(cmap_name, rgb_order) &#183; color(cmap_name, index)",
                   "&#160;&#160;&#160;&#160;used by Graph.to_graphviz()"], kind="eap")

    p.cls("cd-model", 3, "external", "HookedTransformer",
          attrs=["+ cfg.use_split_qkv_input = True",
                 "+ cfg.use_attn_result = True",
                 "+ cfg.use_hook_mlp_in = True",
                 "+ cfg.ungroup_grouped_query_attention = True"],
          methods=["+ hooks(fwd_hooks, bwd_hooks)",
                   "+ to_tokens / to_string"], kind="external")

    # --- artifacts
    p.cls("cd-in", 4, "artifact", "eap_dataset/&#8230;.csv",
          attrs=["clean, corrupted",
                 "correct_idx, incorrect_idx"], kind="data")
    p.cls("cd-pt", 4, "artifact", "&lt;element&gt;_circuit_topk-&lt;k&gt;.pt",
          attrs=["serialized Graph:",
                 "cfg, scores, in_graph masks"], kind="data")
    p.cls("cd-csv", 4, "artifact", "&lt;element&gt;_circuit_topk-&lt;k&gt;.csv",
          attrs=["parent_node",
                 "child_node   (e.g. a12.h3, m7)",
                 "child_type   (q | k | v | None)"], kind="data")
    p.cls("cd-log", 4, "artifact", "faithfulness log csv",
          attrs=["element, metric, total_edges",
                 "baseline_score, top_k, edge_percentage",
                 "circuit_score, faithfulness"], kind="data")

    p.edge("cd-sh", "cd-entry", "invokes", "sbatch")
    p.edge("cd-entry", "cd-graph", "uses", "Graph.from_model(model)")
    p.edge("cd-entry", "cd-attr", "uses", "attribute(&#8230;, method='EAP-IG-inputs', is_absa=True)")
    p.edge("cd-entry", "cd-eval", "uses", "baseline &amp; top-k scores")
    p.edge("cd-entry", "cd-metric", "uses", "partial(logit_diff, &#8230;)")
    p.edge("cd-entry", "cd-eaphelp", "uses", "edge_merging")
    p.edge("cd-graph", "cd-node", "owns", "nodes")
    p.edge("cd-graph", "cd-edge", "owns", "edges")
    p.edge("cd-graph", "cd-gcfg", "owns", "cfg")
    p.edge("cd-node-subs", "cd-node", "extends")
    p.edge("cd-edge", "cd-node", "assoc", "parent / child")
    p.edge("cd-attr", "cd-graph", "uses", "writes graph.scores")
    p.edge("cd-attr", "cd-eapds", "uses", "batches")
    p.edge("cd-attr", "cd-metric", "uses", "metric(&#8230;).backward()")
    p.edge("cd-attr", "cd-model", "uses", "fwd / bwd hooks")
    p.edge("cd-eval", "cd-graph", "uses", "reads in_graph")
    p.edge("cd-eval", "cd-eapds", "uses")
    p.edge("cd-eapds", "cd-eaphelp", "uses", "collate_EAP / safe_parse")
    p.edge("cd-graph", "cd-viz", "uses", "to_graphviz")
    p.edge("cd-in", "cd-eapds", "flow", "read")
    p.edge("cd-graph", "cd-pt", "flow", "to_pt")
    p.edge("cd-pt", "cd-csv", "flow", "edge_merging")
    p.edge("cd-entry", "cd-log", "flow", "log_results_csv")

    p.add_legend()
    return p


# ============================================================== page 4 ======

def page_selective_sft():
    p = Page("4. Selective Circuit-Based SFT",
             "4. Selective Circuit-Based SFT — run_sft.py --circuit_csv_path",
             "The same training loop as Full SFT, but every parameter outside the discovered circuit is "
             "held fixed. Freezing is done with backward hooks that zero the gradient of inactive "
             "attention heads, so the optimizer sees updates only for the heads named in the circuit CSV. "
             "A random-circuit control of the same size is available via --random_circuit.")
    p.add_column(310, top=180)   # 0: driver
    p.add_column(470, top=180)   # 1: masking
    p.add_column(430, top=180)   # 2: reused training stack
    p.add_column(360, top=180)   # 3: artifacts

    p.cls("cs-sh", 0, "job script", "scripts/sft_circuit.sh",
          attrs=["SEEDS = 9584 123 2024 31415 777",
                 "TOPKS = 1000 2000 5000"],
          methods=["python run_sft.py --circuit_csv_path &#8230;"], kind="script")

    p.cls("cs-entry", 0, "entry point", "run_sft.py  (circuit mode)",
          attrs=["+ circuit_csv_path: str  (required)",
                 "+ train_full_model = False",
                 "+ random_circuit: bool",
                 "+ random_circuit_sample_n: int",
                 "+ random_circuit_strict: bool",
                 "+ random_circuit_sample_like_topk: int",
                 "+ seed &#8594; random_state",
                 "+ num_epochs, batch_size, lr, optimizer"],
          methods=["+ main(args)"], kind="entry")

    p.note("cs-note", 0,
           "<b>top_k</b> for the run name and for W&amp;B logging is parsed straight out of the "
           "circuit CSV filename (<i>&#8230;_topk-2000.csv</i> &#8594; <i>topk-2000</i>), so the "
           "filename convention from circuit discovery is load-bearing here.", height=110)

    p.cls("cs-random", 0, "module", "src.utils  —  random-circuit control",
          methods=["+ get_random_nodes(model, df, include_mlp=False,",
                   "&#160;&#160;&#160;&#160;include_logits=False, sample_n, random_state,",
                   "&#160;&#160;&#160;&#160;strict=False) : DataFrame",
                   "&#160;&#160;&#160;&#160;all a&lt;L&gt;.h&lt;H&gt; &#215; {q,k,v}, anti-joined against",
                   "&#160;&#160;&#160;&#160;the real circuit, then sampled",
                   "&#160;&#160;&#160;&#160;strict=True also drops whole heads in the circuit"], kind="util")

    p.cls("cs-mask", 1, "module", "src.utils  —  circuit masking",
          methods=["+ apply_active_edge_unfreezing(model, csv_path,",
                   "&#160;&#160;&#160;&#160;random_circuit=False, sample_n=None,",
                   "&#160;&#160;&#160;&#160;random_state=None, strict=False,",
                   "&#160;&#160;&#160;&#160;sample_like_topk=None) : HookedTransformer",
                   "&#160;&#160;&#160;&#160;1. read circuit CSV (or sample a random one)",
                   "&#160;&#160;&#160;&#160;2. parse child_node &#8594; (layer, head, q|k|v)",
                   "&#160;&#160;&#160;&#160;&#160;&#160;&#160;and m&lt;L&gt; &#8594; mlp_layers",
                   "&#160;&#160;&#160;&#160;3. buffer.requires_grad = False for all buffers",
                   "&#160;&#160;&#160;&#160;4. register gradient masks on W_Q / W_K / W_V",
                   "&#160;&#160;&#160;&#160;&#160;&#160;&#160;per (layer, projection)",
                   "&#160;&#160;&#160;&#160;5. register a mask on W_O for every head that",
                   "&#160;&#160;&#160;&#160;&#160;&#160;&#160;appears in any q/k/v role",
                   "- _derive_like_path(path, like_topk) : str",
                   "- _count_unique_pairs_from_csv(path) : int",
                   "- register_head_mask(weight_tensor, active_heads)",
                   "&#160;&#160;&#160;&#160;- mask_hook(grad) : grad * mask"], kind="util")

    p.note("cs-note2", 1,
           "<b>What is actually frozen.</b> No parameter is detached: the whole model still receives "
           "gradients, but <i>Tensor.register_hook</i> multiplies the incoming gradient of "
           "<i>W_Q/W_K/W_V/W_O</i> by a per-head 0/1 mask, so inactive heads get a zero update. "
           "Embeddings, layer norms, MLP weights and the unembedding are <b>not</b> masked and keep "
           "training — <i>mlp_layers</i> is parsed and reported but never turned into a mask.", height=150)

    p.cls("cs-attn", 1, "external", "HookedTransformer.blocks[L].attn",
          attrs=["+ W_Q, W_K, W_V : [n_heads, d_model, d_head]",
                 "+ W_O : [n_heads, d_head, d_model]"],
          methods=["Tensor.register_hook(mask_hook)"], kind="external")

    # --- reused training stack
    p.cls("cs-ds", 2, "class", "ABSAAutoRegressiveDataset",
          attrs=["(identical to Full SFT)"],
          methods=["+ __getitem__(idx) : {'tokens': Tensor}"], kind="util")

    p.cls("cs-cfg", 2, "dataclass", "HookedTransformerTrainConfig",
          attrs=["+ top_k = 'topk-&lt;k&gt;'  (from the CSV name)",
                 "+ sample_size, subtract_data_amount",
                 "+ optimizer_name, lr, num_epochs, &#8230;",
                 "+ wandb_run_name = seed-&lt;s&gt;_topk-&lt;k&gt;_&#8230;"], kind="util")

    p.cls("cs-train", 2, "module", "src.train",
          methods=["+ train(model, config, dataset, val_dataset)",
                   "&#160;&#160;&#160;&#160;: HookedTransformer",
                   "&#160;&#160;&#160;&#160;unchanged — the masks live on the tensors,",
                   "&#160;&#160;&#160;&#160;not in the loop"], kind="util")

    p.cls("cs-load", 2, "module", "src.utils  —  model loading",
          methods=["+ load_model(model_name, device)",
                   "+ load_finetuned_model_lens_from_dir(dir, device)",
                   "&#160;&#160;&#160;&#160;--finetuned_model_path continues SFT from a",
                   "&#160;&#160;&#160;&#160;checkpoint instead of the base model"], kind="util")

    p.cls("cs-samp", 2, "module", "src.sampling",
          methods=["+ annotate_difficulty(_gas) &#183; stratified_sample_by_factors(_gas)",
                   "&#160;&#160;&#160;&#160;same optional budget control as Full SFT"], kind="util")

    # --- artifacts
    p.cls("cs-in-csv", 3, "artifact", "&#8230;/aos_circuit_topk-&lt;k&gt;.csv",
          attrs=["parent_node",
                 "child_node",
                 "child_type (q | k | v | None)"], kind="data")
    p.cls("cs-in-json", 3, "artifact", "hotel_aste_train_augmented&#8230;.json",
          attrs=["sentence_id, input, target",
                 "task_elements, element_order"], kind="data")
    p.cls("cs-out", 3, "artifact", "outputs/models/&#8230;_topk-&lt;k&gt;/",
          attrs=["model.pt",
                 "model_config.pkl",
                 "tokenizer files"], kind="data")
    p.cls("cs-wandb", 3, "external", "Weights &amp; Biases",
          attrs=["project = 'absa-eap'",
                 "run = seed-&lt;s&gt;_topk-&lt;k&gt;_&#8230;",
                 "train_loss, val_loss, samples"], kind="external")

    p.edge("cs-sh", "cs-entry", "invokes", "sbatch")
    p.edge("cs-entry", "cs-mask", "uses", "apply_active_edge_unfreezing(model, csv)")
    p.edge("cs-mask", "cs-random", "uses", "random_circuit = True")
    p.edge("cs-random", "cs-attn", "uses", "enumerates heads from model.cfg")
    p.edge("cs-mask", "cs-attn", "uses", "register_head_mask")
    p.edge("cs-entry", "cs-ds", "uses")
    p.edge("cs-entry", "cs-cfg", "uses")
    p.edge("cs-entry", "cs-train", "uses")
    p.edge("cs-entry", "cs-load", "uses")
    p.edge("cs-entry", "cs-samp", "uses", "sample_size set")
    p.edge("cs-train", "cs-cfg", "uses")
    p.edge("cs-train", "cs-ds", "uses")
    p.edge("cs-in-csv", "cs-mask", "flow", "read")
    p.edge("cs-in-json", "cs-ds", "flow", "read")
    p.edge("cs-train", "cs-out", "flow", "writes")
    p.edge("cs-train", "cs-wandb", "flow", "logs")

    p.add_legend()
    return p


# ============================================================== page 5 ======

def page_evaluation():
    p = Page("5. Evaluation",
             "5. Evaluation — run_eval.py",
             "Greedy batched generation over the held-out test set, followed by micro precision / recall / "
             "F1 per element order. Predictions and targets are split on the prompt-type separator "
             "(<i>[SSEP]</i> for MVP, <i>;</i> for GAS) and compared as sets of triplet strings.")
    p.add_column(320, top=170)   # 0: driver
    p.add_column(450, top=170)   # 1: eval flow
    p.add_column(430, top=170)   # 2: metrics
    p.add_column(370, top=170)   # 3: artifacts

    p.cls("ev-sh", 0, "job script", "scripts/eval.sh",
          attrs=["SEEDS = 9584 123 2024 31415 777",
                 "TEST_JSON = hotel_aste_test_augmented.json",
                 "MODEL_DIR, OUTPUT_DIR",
                 "skips model dirs already evaluated"],
          methods=["python run_eval.py &#8230; (per model dir)"], kind="script")

    p.cls("ev-entry", 0, "entry point", "run_eval.py",
          attrs=["+ test_json_path: str",
                 "+ model_path: str",
                 "+ prompt_type: mvp | gas",
                 "+ output_dir: str",
                 "+ batch_size: int",
                 "+ save_predictions: bool"],
          methods=["+ main(args)",
                   "&#160;&#160;&#160;&#160;1. load model, pick device",
                   "&#160;&#160;&#160;&#160;2. batched greedy generate (max 300 new tokens)",
                   "&#160;&#160;&#160;&#160;3. strip the prompt prefix from each output",
                   "&#160;&#160;&#160;&#160;4. split target / prediction into triplet lists",
                   "&#160;&#160;&#160;&#160;5. group by element_order, score each group"], kind="entry")

    p.cls("ev-franken", 0, "entry point", "run_eval_franken.py",
          attrs=["variant for Franken-adapter checkpoints",
                 "loads base Qwen + raw state_dict",
                 "uses postprocess_absa_outputs (voting)"],
          methods=["+ main(args)"], kind="entry")

    p.cls("ev-load", 1, "module", "src.utils  —  model loading",
          methods=["+ load_finetuned_model_lens_from_dir(dir, device)",
                   "&#160;&#160;&#160;&#160;: HookedTransformer"], kind="util")

    p.cls("ev-model", 1, "external", "HookedTransformer",
          methods=["+ generate(input, max_new_tokens=300,",
                   "&#160;&#160;&#160;&#160;stop_at_eos=True, do_sample=False,",
                   "&#160;&#160;&#160;&#160;return_type='str', padding_side='left')",
                   "&#160;&#160;&#160;&#160;: str | List[str]",
                   "+ eval() &#183; to(device)"], kind="external")

    p.note("ev-note", 1,
           "<b>Per-order scoring.</b> Test items carry an <i>element_order</i> "
           "(<i>aos</i>, <i>aso</i>, <i>sao</i>, &#8230;). Predictions are bucketed by that key and "
           "<i>calculate_metrics</i> runs once per bucket, so the result file holds "
           "<i>precision_&lt;order&gt;</i> / <i>recall_&lt;order&gt;</i> / <i>f1_&lt;order&gt;</i> "
           "triples, scaled by 100.", height=140)

    p.cls("ev-metrics", 2, "module", "src.utils  —  scoring",
          methods=["+ calculate_metrics(predictions, targets, task)",
                   "&#160;&#160;&#160;&#160;: {precision_&lt;task&gt;, recall_&lt;task&gt;, f1_&lt;task&gt;}",
                   "&#160;&#160;&#160;&#160;micro TP / FP / FN over triplet strings",
                   "+ parse_absa_string(text) : List[Dict[str, str]]",
                   "&#160;&#160;&#160;&#160;'[A] a [O] o [S] s [SSEP] &#8230;' &#8594; [{A,O,S}, &#8230;]",
                   "+ postprocess_absa_outputs(preds, labels,",
                   "&#160;&#160;&#160;&#160;sentence_id, task)",
                   "&#160;&#160;&#160;&#160;majority-vote aggregation across the permutations",
                   "&#160;&#160;&#160;&#160;of one sentence (MvP-style)"], kind="util")

    p.cls("ev-circuit-eval", 2, "module", "eap.evaluate  (cross-reference)",
          methods=["+ evaluate_baseline_multitoken(&#8230;)",
                   "+ evaluate_graph_multitoken(&#8230;)",
                   "&#160;&#160;&#160;&#160;circuit faithfulness is measured in step 3,",
                   "&#160;&#160;&#160;&#160;not here — this step scores task quality"], kind="eap")

    p.cls("ev-in", 3, "artifact", "hotel_aste_test_augmented.json",
          attrs=["sentence_id, input, target",
                 "task_elements, element_order"], kind="data")
    p.cls("ev-raw", 3, "artifact", "raw_inference_results.json",
          attrs=["list of decoded generations",
                 "(debugging copy, always written)"], kind="data")
    p.cls("ev-res", 3, "artifact", "evaluation_results.json",
          attrs=["precision_&lt;order&gt;",
                 "recall_&lt;order&gt;",
                 "f1_&lt;order&gt;   (&#215;100)"], kind="data")
    p.cls("ev-inf", 3, "artifact", "inference_results.json",
          attrs=["sentence_id, task_elements",
                 "element_order, input, target",
                 "prediction, target_list, prediction_list",
                 "written only with --save_predictions"], kind="data")

    p.edge("ev-sh", "ev-entry", "invokes", "sbatch")
    p.edge("ev-entry", "ev-load", "uses")
    p.edge("ev-load", "ev-model", "flow", "returns")
    p.edge("ev-entry", "ev-model", "uses", "generate")
    p.edge("ev-entry", "ev-metrics", "uses", "calculate_metrics")
    p.edge("ev-franken", "ev-metrics", "uses", "postprocess_absa_outputs")
    p.edge("ev-in", "ev-entry", "flow", "read")
    p.edge("ev-entry", "ev-raw", "flow", "writes")
    p.edge("ev-entry", "ev-res", "flow", "writes")
    p.edge("ev-entry", "ev-inf", "flow", "writes")

    p.add_legend()
    return p


# ==================================================================== main ==

PAGES = [
    ("00_pipeline_overview.drawio", page_overview),
    ("01_dataset_creation.drawio", page_dataset_creation),
    ("02_full_sft.drawio", page_full_sft),
    ("03_circuit_discovery.drawio", page_circuit_discovery),
    ("04_selective_circuit_sft.drawio", page_selective_sft),
    ("05_evaluation.drawio", page_evaluation),
]


def main():
    combined = []
    for filename, builder in PAGES:
        write_mxfile(os.path.join(OUT_DIR, filename), [builder()])
        combined.append(builder())
    write_mxfile(os.path.join(OUT_DIR, "absa_eap_class_diagrams.drawio"), combined)


if __name__ == "__main__":
    main()
