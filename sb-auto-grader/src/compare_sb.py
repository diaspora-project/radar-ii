"""
compare_sb.py -- Compare an AI-generated Scheduling Block against its ground-truth
SB and report where they agree and differ.

The ground-truth *_ground.optSB files are the authoritative "correct" answers.
This does NOT replace the rule-based grader (score_sb.py / llm_judge.py); it is a
second, complementary view: "how close is the AI SB to the known-good SB?"

Differences are split into two buckets, because an AI SB can differ from ground
truth and still be valid:
  * SIGNIFICANT  -- band, array config, flux calibrator, reference pointing, target.
                    A mismatch here usually means the AI got something wrong.
  * INFORMATIONAL-- phase-cal choice, resource variant, scan counts, durations,
                    on-source time. These often differ between valid SBs.

Usage:
    python compare_sb.py <ai_sb> <ground_truth_sb>
    python compare_sb.py --auto # automatically pair *_AItest.txt with *_ground.optSB in cwd
    python compare_sb.py --auto --dir <folder>
"""

import os
import re
import sys
import glob
import argparse

from score_sb import (SB, total_on_source_seconds,
                      STANDARD_FLUX_CALS, FLARING_FLUX_CALS)

# Which comparison fields are treated as significant vs informational.
SIGNIFICANT = ["target", "config", "bands", "flux_cal", "ref_pointing"]
INFORMATIONAL = ["phase_cal", "resources", "n_loops", "n_target_scans", "on_source_min"]

LABELS = {
    "target": "Target(s)",
    "config": "Array config",
    "bands": "Band(s)",
    "flux_cal": "Flux calibrator",
    "ref_pointing": "Reference pointing",
    "phase_cal": "Phase calibrator(s)",
    "resources": "Hardware resources",
    "n_loops": "# loops",
    "n_target_scans": "# target scans",
    "on_source_min": "On-source (min)",
}

# On-source time within this fraction counts as a match.
ONSOURCE_TOL = 0.10


def summarize(sb):
    """Extract the comparable attributes of an SB."""
    scans = sb.scans
    return {
        "target": sorted({s.source for s in scans if s.has("ObsTgt")}),
        "config": sb.config,
        "bands": sorted(sb.science_bands),
        "flux_cal": sorted({s.source for s in scans if s.has("CalFlux")}),
        "ref_pointing": any(s.is_refpointing for s in scans),
        "phase_cal": sorted({s.source for s in scans
                             if s.has("CalGain") and "=" not in s.source}),
        "resources": sorted({s.resource for s in scans}),
        "n_loops": sum(1 for _, k, _ in sb.lines if k == "LOOP-START"),
        "n_target_scans": sum(1 for s in scans if s.has("ObsTgt")),
        "on_source_min": round(total_on_source_seconds(sb) / 60.0, 1),
    }


def _core(name):
    """Core transient designation, e.g. 'SN2024rjw_2as' / 'AT2024rjw' -> '2024rjw'."""
    m = re.search(r"(\d{4}[a-z]+)", name, re.I)
    return m.group(1).lower() if m else name.lower()


def _acceptable_flux_substitution(ai_flux):
    """A flux-cal difference is acceptable if every AI flux cal is a standard
    calibrator that is not currently flaring (per NRAO fdscale). Position ('not too
    far from the source') still needs a human/expert check."""
    if not ai_flux:
        return False
    for f in ai_flux:
        std = any(s in f for s in STANDARD_FLUX_CALS)
        flaring = any(fl in f for fl in FLARING_FLUX_CALS)
        if not std or flaring:
            return False
    return True


def is_match(field, a, b):
    if field == "on_source_min":
        hi = max(a, b, 1e-9)
        return abs(a - b) / hi <= ONSOURCE_TOL
    if field == "target":
        # Compare core designations so name suffixes / SN-vs-AT prefixes don't
        # count, but a real number difference (2026ulz vs 2025ulz) still shows.
        return {_core(x) for x in a} == {_core(x) for x in b}
    return a == b


def fmt(val):
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else "(none)"
    return str(val)


def compare(ai_path, ground_path):
    ai, gt = SB(ai_path), SB(ground_path)
    sa, sg = summarize(ai), summarize(gt)

    print("=" * 90)
    print(f"AI SB     : {os.path.basename(ai_path)}")
    print(f"Ground SB : {os.path.basename(ground_path)}")
    print("-" * 90)
    print(f"  {'Attribute':<20} {'AI':<30} {'Ground':<26} Result")
    print("  " + "-" * 86)

    sig_diffs, info_diffs = [], []
    for field in SIGNIFICANT + INFORMATIONAL:
        a, b = sa[field], sg[field]
        match = is_match(field, a, b)
        if match:
            mark = "match"
        elif field == "flux_cal" and _acceptable_flux_substitution(a):
            # A different standard, non-flaring flux cal is an acceptable substitution.
            mark = "differs (ok: standard substitution; verify position)"
            info_diffs.append(field)
        elif field in SIGNIFICANT:
            mark = ">> DIFFERS <<"
            sig_diffs.append(field)
        else:
            mark = "differs (info)"
            info_diffs.append(field)
        print(f"  {LABELS[field]:<20} {fmt(a):<30} {fmt(b):<26} {mark}")

    print("-" * 90)
    n_sig = len(SIGNIFICANT)
    sig_match = n_sig - len(sig_diffs)
    verdict = "MATCH" if not sig_diffs else "SIGNIFICANT DIFFERENCES"
    print(f"  Significant fields matching ground truth: {sig_match}/{n_sig}   ->  {verdict}")
    if sig_diffs:
        print("  Significant differences: " + ", ".join(LABELS[f] for f in sig_diffs))
    if info_diffs:
        print("  Informational differences (may still be valid): "
              + ", ".join(LABELS[f] for f in info_diffs))
    print("=" * 90)
    print()
    return sig_diffs, info_diffs

_DESIG = re.compile(r"((?:SN|AT)\d{4}[a-zA-Z]+)")
_BANDS = ["Ku", "Ka", "P", "L", "S", "C", "X", "K", "Q"]


def _designation(name):
    m = _DESIG.search(name)
    return m.group(1) if m else None


def _band_set(name):
    """Set of band letters from the token that directly follows the designation.

    Only that one token is parsed (e.g. 'Xband', 'Kuband', 'SXKu', 'S,X,Ku'), so
    stray letters elsewhere in the filename (e.g. the 'P' in 'noRefPoint') are
    ignored.
    """
    stub = re.sub(r"\.(optSB|txt)$", "", name)
    parts = stub.split("_")
    token = ""
    for i, p in enumerate(parts):
        if _DESIG.search(p):
            token = parts[i + 1] if i + 1 < len(parts) else ""
            break
    token = token.replace("band", "")
    found = set()
    for grp in re.split(r"[,\s]+", token):   # handle comma-separated "S,X,Ku"
        i = 0
        while i < len(grp):
            if grp[i:i + 2] in ("Ku", "Ka"):
                found.add(grp[i:i + 2])
                i += 2
            elif grp[i] in "PLSCXKQ":
                found.add(grp[i])
                i += 1
            else:
                i += 1
    return found


def auto_pairs(folder):
    ai_files = sorted(glob.glob(os.path.join(folder, "*_AItest.txt")))
    ground_files = sorted(glob.glob(os.path.join(folder, "*_ground.optSB")))
    pairs, unmatched = [], []
    for ai in ai_files:
        an = os.path.basename(ai)
        adesig, abands = _designation(an), _band_set(an)
        best = None
        for gt in ground_files:
            gn = os.path.basename(gt)
            if _designation(gn) == adesig and _band_set(gn) == abands:
                best = gt
                break
        if best:
            pairs.append((ai, best))
        else:
            unmatched.append(ai)
    return pairs, unmatched


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ai_sb", nargs="?", help="AI-generated SB file")
    parser.add_argument("ground_sb", nargs="?", help="Ground-truth SB file")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-pair *_AItest.txt with *_ground.optSB")
    parser.add_argument("--dir", default=".", help="Folder for --auto (default: cwd)")
    args = parser.parse_args(argv[1:])

    if args.auto:
        pairs, unmatched = auto_pairs(args.dir)
        for ai, gt in pairs:
            compare(ai, gt)
        if unmatched:
            print("No ground-truth match found for (pair these manually):")
            for ai in unmatched:
                print(f"  - {os.path.basename(ai)}  "
                      f"(designation {_designation(os.path.basename(ai))}, "
                      f"bands {sorted(_band_set(os.path.basename(ai))) or '?'})")
        return 0

    if not args.ai_sb or not args.ground_sb:
        print(__doc__)
        return 1
    compare(args.ai_sb, args.ground_sb)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
