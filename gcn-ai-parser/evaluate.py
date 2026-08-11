"""
evaluate.py
Compute event-level precision P, recall R, F1, and GCN-level recall GCN-R
(as defined in Patel et al. 2025) for each model across multiple run files.
Reports mean ± std across runs and a per-GCN error breakdown.
"""

import json
import re
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS = {
    "GPT-5.5": [
        "result/gpt55_run1.json",
        "result/gpt55_run2.json",
        "result/gpt55_run3.json",
    ],
    "Claude Opus 4.7": [
        "result/claude_run1.json",
        "result/claude_run2.json",
        "result/claude_run3.json",
    ],
    "Claude Opus 4.8": [
        "result/claude_48_run1.json",
        "result/claude_48_run2.json",
        "result/claude_48_run3.json",
    ],
    "Gemini 3.5 Flash": [
        "result/gemini_run1.json",
        "result/gemini_run2.json",
        "result/gemini_run3.json",
    ],
}
GROUND_TRUTH_FILE = "result/ground_truth.json"

# Fractional tolerance for numeric field matching
FREQ_TOL = 0.01   # 1% — frequency in GHz
FLUX_TOL = 0.01   # 1% — flux density in mJy
# ──────────────────────────────────────────────────────────────────────────────


# ── Unit parsing ──────────────────────────────────────────────────────────────

def parse_freq_ghz(val):
    """Parse LLM frequency string to float GHz. Returns None on failure."""
    if val is None:
        return None
    if isinstance(val, list):
        val = val[0]  # take first element if model returned a list
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).lower().strip()
    m = re.search(r"([\d.]+(?:e[+-]?\d+)?)", s)
    if not m:
        return None
    num = float(m.group(1))
    if "mhz" in s:
        return num / 1000
    if "hz" in s and "ghz" not in s and "mhz" not in s:
        return num / 1e9
    return num  # assume GHz


def parse_flux_mjy(val):
    """Parse LLM flux density string to float mJy. Returns None on failure."""
    if val is None:
        return None
    if isinstance(val, list):
        val = val[0]
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).lower().strip()
    s = s.lstrip("<~").strip()
    s = re.sub(r"(/beam|rms|\(.*?\))", "", s).strip()
    m = re.search(r"([\d.]+(?:e[+-]?\d+)?)", s)
    if not m:
        return None
    num = float(m.group(1))
    if any(u in s for u in ["ujy", "microjy", "μjy"]):
        return num / 1000
    if "jy" in s and "mjy" not in s and "millijy" not in s and "milli jy" not in s:
        return num * 1000
    return num  # assume mJy


def approx_eq(a, b, tol):
    if a is None or b is None:
        return False
    return abs(a - b) / max(abs(b), 1e-12) <= tol


def extract_datetime_str(time_string):
    """Extract 'YY/MM/DD HH:MM:SS' from 'DATE:    YY/MM/DD HH:MM:SS GMT'."""
    if time_string is None:
        return None
    m = re.search(r"(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", str(time_string))
    return m.group(1) if m else None


# ── Event matching ────────────────────────────────────────────────────────────

def match_event(llm, gt):
    """
    Compare one LLM event against one GT event on all required fields.
    Returns (is_match: bool, mismatches: list[str]).
    """
    mm = []

    # Frequency
    lf = parse_freq_ghz(llm.get("frequency"))
    gf = gt["frequency"]
    if lf is None:
        mm.append("frequency: missing in LLM output")
    elif not approx_eq(lf, gf, FREQ_TOL):
        mm.append(f"frequency: LLM={lf:.4f} GHz  GT={gf:.4f} GHz")

    # Flux density
    lx = parse_flux_mjy(llm.get("flux_density"))
    gx = gt["flux_density"]
    if lx is None:
        mm.append("flux_density: missing in LLM output")
    elif not approx_eq(lx, gx, FLUX_TOL):
        mm.append(f"flux_density: LLM={lx:.6g} mJy  GT={gx:.6g} mJy")

    # Name — only checked when GT has a name
    gn = gt.get("name")
    ln = llm.get("name")
    if gn is not None:
        if ln is None:
            mm.append(f"name: LLM=None  GT={gn!r}")
        elif gn.lower() not in ln.lower() and ln.lower() not in gn.lower():
            mm.append(f"name: LLM={ln!r}  GT={gn!r}")

    # RA / Dec — only checked when GT has them
    for coord, alt_key in (("ra", "right_ascension"), ("dec", "declination")):
        gt_val = gt.get(coord)
        if gt_val is not None:
            llm_val = llm.get(coord) or llm.get(alt_key)
            if llm_val != gt_val:
                mm.append(f"{coord}: LLM={llm_val!r}  GT={gt_val!r}")

    return len(mm) == 0, mm


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(llm_events, gt_events, loose=False):
    """
    Compute P, R, F1, GCN-R for one run.
    llm_events must already be filtered to GCNs present in ground truth.
    Returns (metrics dict, error_details dict).
    """
    gt_gcns = sorted(set(d["GCN_number"] for d in gt_events))

    gt_by_gcn  = defaultdict(list)
    llm_by_gcn = defaultdict(list)
    for d in gt_events:
        gt_by_gcn[d["GCN_number"]].append(d)
    for d in llm_events:
        llm_by_gcn[d["GCN_number"]].append(d)

    total_matched = 0
    total_llm     = 0
    total_gt      = 0
    gcn_matched   = 0
    error_details = {}

    for gcn in gt_gcns:
        gt_list  = gt_by_gcn[gcn]
        llm_list = llm_by_gcn.get(gcn, [])

        total_gt  += len(gt_list)
        total_llm += len(llm_list)

        # Greedy one-to-one matching: for each GT event find the first exact LLM match
        used_llm     = set()
        unmatched_gt = []

        for gt_ev in gt_list:
            matched_idx = None
            for li, llm_ev in enumerate(llm_list):
                if li in used_llm:
                    continue
                ok, _ = match_event(llm_ev, gt_ev)
                if ok:
                    matched_idx = li
                    break
            if matched_idx is not None:
                used_llm.add(matched_idx)
                total_matched += 1
            else:
                # Find closest LLM event for diagnostics
                best_mm = None
                for li, llm_ev in enumerate(llm_list):
                    _, mm = match_event(llm_ev, gt_ev)
                    if best_mm is None or len(mm) < len(best_mm):
                        best_mm = mm
                unmatched_gt.append((gt_ev, best_mm or ["no LLM output for this GCN"]))

        hallucinated = [llm_list[i] for i in range(len(llm_list)) if i not in used_llm]
        gcn_ok = (len(unmatched_gt) == 0) if loose else (len(unmatched_gt) == 0 and len(hallucinated) == 0)
        if gcn_ok:
            gcn_matched += 1
        else:
            error_details[gcn] = {
                "unmatched_gt": unmatched_gt,
                "hallucinated": hallucinated,
            }

    P    = total_matched / total_llm   if total_llm   > 0 else 0.0
    R    = total_matched / total_gt    if total_gt    > 0 else 0.0
    F1   = 2 * P * R / (P + R)        if (P + R)     > 0 else 0.0
    GCNR = gcn_matched  / len(gt_gcns) if gt_gcns    else 0.0

    return {"P": P, "R": R, "F1": F1, "GCN-R": GCNR}, error_details


# ── Error reporting ───────────────────────────────────────────────────────────

def print_errors(error_details):
    print("\n── Per-GCN error breakdown ─────────────────────────────────────────")
    # Count mismatch types across all GCNs for a summary
    mismatch_counter = defaultdict(int)

    for gcn in sorted(error_details, key=lambda x: int(x)):
        info = error_details[gcn]
        missed  = info["unmatched_gt"]
        extra   = info["hallucinated"]
        print(f"\n  GCN {gcn}  (missed GT={len(missed)}, extra LLM={len(extra)})")

        for gt_ev, mm in missed:
            print(f"    [MISSED]  freq={gt_ev['frequency']} GHz  "
                  f"flux={gt_ev['flux_density']} mJy  name={gt_ev['name']!r}")
            for m in mm:
                print(f"             → {m}")
                field = m.split(":")[0].strip()
                mismatch_counter[field] += 1

        for llm_ev in extra:
            freq = llm_ev.get("frequency")
            flux = llm_ev.get("flux_density")
            name = llm_ev.get("name")
            print(f"    [EXTRA]   freq={freq!r}  flux={flux!r}  name={name!r}")
            mismatch_counter["extra_llm_event"] += 1

    print("\n── Most common mismatch types ──────────────────────────────────────")
    for field, count in sorted(mismatch_counter.items(), key=lambda x: -x[1]):
        print(f"  {field:<30} {count}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loose", action="store_true",
                        help="Loose GCN-R: a GCN matches if all GT events are covered, "
                             "even if the LLM produced extra predictions.")
    args = parser.parse_args()

    gt_all     = json.load(open(GROUND_TRUTH_FILE))
    gt_gcn_set = set(d["GCN_number"] for d in gt_all)

    if args.loose:
        print("GCN-R mode: LOOSE (extras do not penalise)\n")
    else:
        print("GCN-R mode: STRICT (extras penalise — use --loose to relax)\n")

    for model, files in RESULTS.items():
        print(f"\n{'='*62}")
        print(f"  Model: {model}  ({len(files)} run{'s' if len(files) > 1 else ''})")
        print(f"{'='*62}")

        per_run_metrics = []
        # Aggregate errors across runs (last run shown for diagnostics)
        last_errors = {}

        for i, fpath in enumerate(files):
            llm_all      = json.load(open(fpath))
            llm_filtered = [d for d in llm_all if d["GCN_number"] in gt_gcn_set]
            dropped      = len(llm_all) - len(llm_filtered)
            if dropped:
                print(f"  Run {i+1}: dropped {dropped} event(s) not in GT "
                      f"({Path(fpath).name})")
            metrics, errors = compute_metrics(llm_filtered, gt_all, loose=args.loose)
            per_run_metrics.append(metrics)
            last_errors = errors

        # Summary table
        print(f"\n  {'Metric':<28} {'Mean':>8}  {'Std':>8}")
        print(f"  {'-'*46}")
        for key in ("P", "R", "F1", "GCN-R"):
            vals = [m[key] for m in per_run_metrics]
            mean = np.mean(vals)
            std  = np.std(vals, ddof=0)
            label = {"P": "Event match precision, P",
                     "R": "Event match recall, R",
                     "F1": "Event match F1",
                     "GCN-R": "GCN match recall, GCN-R"}[key]
            print(f"  {label:<28} {mean:>8.3f}  {std:>8.3f}")

        if len(files) == 1:
            print("\n  (Single run — std = 0 by definition)")

        print_errors(last_errors)


if __name__ == "__main__":
    main()
