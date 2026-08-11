"""
llm_judge.py -- Fully automated SB evaluation: mechanical checks in code
(score_sb.py) PLUS an LLM judge for the checks that need external knowledge.

The mechanical/computed checks remain authoritative and deterministic. Only the
four `needs_external_data` rubric checks are handed to the lab's Argo LLM:

    phase_cal_visible     -- is the phase cal observable in this band + config?
    phase_cal_separation  -- target<->phase-cal angular separation vs threshold
    coord_fwhm_match      -- do the SB coords match the named target's true position?
    sensitivity_depth     -- does the on-source time reach the requested sigma depth?

The LLM returns strict JSON verdicts (PASS / FAIL / UNSURE + reason); these fold
into the same +1/-1 score. UNSURE stays a manual flag (0 points).

Usage:
    python llm_judge.py <sb_file> [<sb_file> ...] [--model GPT-5.5] [--no-llm]
"""

import os
import re
import sys
import json
import argparse

import score_sb
from score_sb import (
    SB, load_rubric, score, load_targets, total_on_source_seconds,
    SEPARATION_THRESHOLDS, Result, PASS, FAIL, REVIEW, FLAG,
)

BASE_URL = "https://apps.inside.anl.gov/argoapi/v1"
DEFAULT_USER = "zilinghan.li"
MODEL = "GPT-5.5"
TEMPERATURE = 0.0
MAX_TOKENS = 2000
MAX_RETRY = 3

LLM_CHECKS = [
    "phase_cal_visible",
    "phase_cal_separation",
    "coord_fwhm_match",
    "sensitivity_depth",
]


# --------------------------------------------------------------------------- #
#  Pull the facts the judge needs out of the SB                               #
# --------------------------------------------------------------------------- #

def extract_facts(sb):
    """Collect target/cal/band/config/on-source facts for the LLM prompt."""
    comment = sb.comment or ""

    # Target RA/Dec are typically written in the SCHED-BLOCK comment, e.g.
    # "target RA 21h03m10.107s Dec +20d45m07.58s".
    ra = _search(r"RA\s*([0-9]{1,2}[hH:][0-9dms.:+\- ]*?)(?=\s*Dec|\s*,|;|$)", comment)
    dec = _search(r"Dec\s*([+\-]?[0-9]{1,2}[dD:][0-9dms.:+\- ]*?)(?=\s*,|;|phase|flux|$)", comment)

    # Requested depth / sigma, if stated (e.g. "3 sigma", "5-sigma", "10 uJy rms").
    depth = _search(r"([0-9.]+\s*(?:sigma|σ|uJy|mJy|micro).*?)(?=[,;]|$)", comment)

    # Phase calibrator = source of the CalGain scans that are NOT the flux cal.
    phase_cals = _unique(s.source for s in sb.scans
                         if s.has("CalGain") and "=" not in s.source)
    flux_cals = _unique(s.source for s in sb.scans if s.has("CalFlux"))
    targets = _unique(s.source for s in sb.scans if s.has("ObsTgt"))

    return {
        "sb_name": sb.name,
        "config": sb.config,
        "science_bands": sorted(sb.science_bands),
        "target_names": targets,
        "target_ra": ra,
        "target_dec": dec,
        "phase_calibrators": phase_cals,
        "flux_calibrators": flux_cals,
        "requested_depth": depth or "not stated",
        "total_on_source_min": round(total_on_source_seconds(sb) / 60.0, 1),
        "sched_comment": comment,
    }


def _search(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None

def _unique(it):
    out = []
    for x in it:
        if x not in out:
            out.append(x)
    return out


# --------------------------------------------------------------------------- #
#  Ground-truth verifier -- resolve external checks against the known-good SB  #
# --------------------------------------------------------------------------- #

# Tolerance for the on-source-time (sensitivity) comparison against ground truth.
ONSOURCE_TOL = 0.10


def _target_cores(names):
    return {re.search(r"(\d{4}[a-z]+)", n, re.I).group(1).lower()
            for n in names if re.search(r"(\d{4}[a-z]+)", n, re.I)}


def ground_verdicts(ai_sb, ground_sb):
    """Resolve external-knowledge checks against a ground-truth SB where possible.

    Returns {check_id: Result}. Only includes checks the ground truth can actually
    decide; everything else is left out (so the LLM/catalog fallback still runs).
    """
    out = {}

    def sources(sb, intent, exclude_flux=False):
        return {s.source for s in sb.scans
                if s.has(intent) and not (exclude_flux and "=" in s.source)}

    ai_bands = ai_sb.science_bands
    gt_bands = ground_sb.science_bands

    # sensitivity_depth: on-source time vs ground truth (same bands only).
    ai_os = total_on_source_seconds(ai_sb) / 60.0
    gt_os = total_on_source_seconds(ground_sb) / 60.0
    if gt_os > 0 and ai_bands == gt_bands:
        tol_pct = f"{ONSOURCE_TOL * 100:.0f}%"
        if ai_os >= gt_os * (1 - ONSOURCE_TOL):
            rel = ">=" if ai_os >= gt_os else f"within {tol_pct} of"
            out["sensitivity_depth"] = Result(
                PASS, f"on-source {ai_os:.1f}m {rel} ground {gt_os:.1f}m -> reaches intended depth",
                source="GROUND")
        else:
            out["sensitivity_depth"] = Result(
                FAIL, f"on-source {ai_os:.1f}m < ground {gt_os:.1f}m by >{tol_pct} -> below intended depth",
                source="GROUND")

    # phase_cal_visible / _separation: ground can CONFIRM (not refute) when the AI
    # uses the same phase calibrator(s) as the known-good SB.
    ai_pc = sources(ai_sb, "CalGain", exclude_flux=True)
    gt_pc = sources(ground_sb, "CalGain", exclude_flux=True)
    if ai_pc and ai_pc == gt_pc:
        out["phase_cal_visible"] = Result(
            PASS, f"phase cal {sorted(gt_pc)} matches ground truth -> known-valid",
            source="GROUND")
        ai_tgt = _target_cores({s.source for s in ai_sb.scans if s.has("ObsTgt")})
        gt_tgt = _target_cores({s.source for s in ground_sb.scans if s.has("ObsTgt")})
        if ai_tgt and ai_tgt == gt_tgt:
            out["phase_cal_separation"] = Result(
                PASS, "same phase cal & target as ground -> separation known-good",
                source="GROUND")
    return out


# --------------------------------------------------------------------------- #
#  LLM judge                                                                   #
# --------------------------------------------------------------------------- #

def build_prompt(facts):
    bands = ", ".join(facts["science_bands"]) or "unknown"
    sep_lines = "\n".join(
        f"  {b}: PASS if <= {g} deg, REVIEW/UNSURE if <= {w} deg, FAIL if > {w} deg"
        for b, (g, w) in SEPARATION_THRESHOLDS.items()
    )
    return f"""You are an expert NRAO/VLA observing validator. Judge the following
Scheduling Block on FOUR criteria that require external astronomical knowledge.
Use your knowledge of the VLA calibrator list, standard source positions (TNS/NED/
SIMBAD), VLA sky coverage by configuration, and the VLA sensitivity/exposure
calculator. If you genuinely cannot determine an answer, use "UNSURE".

SB facts:
  file: {facts['sb_name']}
  band(s): {bands}
  array configuration: {facts['config']}
  target name(s): {facts['target_names']}
  target RA (from SB): {facts['target_ra']}
  target Dec (from SB): {facts['target_dec']}
  phase calibrator(s): {facts['phase_calibrators']}
  flux calibrator(s): {facts['flux_calibrators']}
  requested depth: {facts['requested_depth']}
  total on-source time: {facts['total_on_source_min']} min
  SB comment: {facts['sched_comment']}

Phase-cal angular-separation thresholds (RADAR guidance):
{sep_lines}

Judge each of these:
1. phase_cal_visible    -- Is the phase calibrator a real VLA calibrator that is
                           observable in the stated band and array configuration?
2. phase_cal_separation -- Estimate the angular separation between the target and
                           the phase calibrator, then apply the threshold above.
3. coord_fwhm_match     -- Do the target RA/Dec in the SB match the true catalog
                           position of the named transient (within a beam FWHM)?
4. sensitivity_depth    -- Given band, configuration, and total on-source time,
                           does the SB plausibly reach the requested depth? If no
                           depth was requested, judge whether the on-source time is
                           reasonable and return UNSURE if it cannot be assessed.

Respond with STRICT JSON only, no other text, of exactly this shape:
{{
  "phase_cal_visible":    {{"verdict": "PASS|FAIL|UNSURE", "reason": "<=25 words"}},
  "phase_cal_separation": {{"verdict": "PASS|FAIL|UNSURE", "reason": "<=25 words, include est. degrees"}},
  "coord_fwhm_match":     {{"verdict": "PASS|FAIL|UNSURE", "reason": "<=25 words"}},
  "sensitivity_depth":    {{"verdict": "PASS|FAIL|UNSURE", "reason": "<=25 words"}}
}}"""


def judge(sb, client, model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
          max_retry=MAX_RETRY):
    facts = extract_facts(sb)
    prompt = build_prompt(facts)
    for _ in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                # temperature=temperature,
                max_tokens=max_tokens,
            )
            data = _extract_json(resp.choices[0].message.content)
            if all(k in data for k in LLM_CHECKS):
                return data
        except Exception as e:  # noqa: BLE001 -- surface and retry
            print(f"  (LLM error, retrying: {e})")
    return None


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = re.sub(r"\n```$", "", text.rstrip())
    start, depth, end = text.find("{"), 0, -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(text[start:end])


def verdict_to_result(entry):
    v = str(entry.get("verdict", "UNSURE")).upper()
    reason = entry.get("reason", "")
    if v == "PASS":
        return Result(PASS, reason, source="LLM")
    if v == "FAIL":
        return Result(FAIL, reason, source="LLM")
    return Result(REVIEW, f"UNSURE: {reason}", source="LLM")


# --------------------------------------------------------------------------- #
#  Merge mechanical score + LLM verdicts                                       #
# --------------------------------------------------------------------------- #

def evaluate(sb, rubric, client, model, use_llm=True, ground_sb=None):
    _, rows = score(sb, rubric)

    # Priority for the external-knowledge checks:
    #   ground truth (oracle)  ->  code/catalog result  ->  LLM estimate  ->  flag
    gverdicts = ground_verdicts(sb, ground_sb) if ground_sb is not None else {}
    lverdicts = judge(sb, client, model) if (use_llm and client is not None) else None

    # Point values come from rubric.yaml (see score_sb.award_points): PASS awards
    # +points; FAIL awards the rule's fail_points / the rubric's fail_default / -points;
    # anything unresolved (flag/review/waived) awards 0.
    checks_by_id = {c["id"]: c for c in rubric["checks"]}
    meta = rubric.get("meta", {})

    merged = []
    total = 0
    for row in rows:
        cid, title, res, awarded, tier = row
        if tier == "needs_external_data":
            if cid in gverdicts:
                # Ground truth can decide -> authoritative.
                res = gverdicts[cid]
            elif res.status == FLAG and lverdicts and cid in lverdicts:
                # Code/catalog left it unresolved -> fall back to the LLM estimate.
                res = verdict_to_result(lverdicts[cid])
        awarded = score_sb.award_points(checks_by_id.get(cid, {}), meta, res.status)
        total += awarded
        merged.append((cid, title, res, awarded, tier))
    return total, merged


def find_ground(ai_path):
    """Locate the ground-truth SB paired with an AI SB (same folder)."""
    import compare_sb
    folder = os.path.dirname(os.path.abspath(ai_path)) or "."
    for ai, gt in compare_sb.auto_pairs(folder)[0]:
        if os.path.abspath(ai) == os.path.abspath(ai_path):
            return gt
    return None


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="SB file(s) to evaluate")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--user", default=os.environ.get("ARGO_USER", DEFAULT_USER))
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the LLM judge (flagged checks stay unresolved)")
    parser.add_argument("--targets", default=None,
                        help="CSV of authoritative target positions (default: targets.csv)")
    parser.add_argument("--ground", default=None,
                        help="Ground-truth SB to verify external checks against "
                             "(single-file mode). Use --ground-auto for many files.")
    parser.add_argument("--ground-auto", action="store_true",
                        help="Auto-find each file's *_ground.optSB and verify against it")
    args = parser.parse_args(argv[1:])

    root = score_sb.data_dir()
    rubric = load_rubric(os.path.join(root, "rubric.yaml"))
    load_targets(args.targets or os.path.join(root, "targets.csv"))

    client = None
    if not args.no_llm:
        from openai import OpenAI  # lazy: --no-llm works without the package/network
        client = OpenAI(api_key=args.user, base_url=BASE_URL)

    for path in args.files:
        if not os.path.isfile(path):
            print(f"skip (not a file): {path}")
            continue
        sb = SB(path)
        ground_sb = None
        gpath = args.ground if args.ground else (find_ground(path) if args.ground_auto else None)
        if gpath and os.path.isfile(gpath):
            ground_sb = SB(gpath)
            print(f"(verifying external checks against ground truth: {os.path.basename(gpath)})")
        elif args.ground_auto:
            print(f"(no ground-truth match found for {os.path.basename(path)}; "
                  f"external checks fall back to catalog/LLM)")
        total, rows = evaluate(sb, rubric, client, args.model,
                               use_llm=not args.no_llm, ground_sb=ground_sb)
        score_sb.print_report(sb, total, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
