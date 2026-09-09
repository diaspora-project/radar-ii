"""
score_sb.py -- Score a VLA Scheduling Block against rubric.yaml.

Usage:
    python score_sb.py <sb_file> [<sb_file> ...]
    python score_sb.py *.txt
    python score_sb.py *.txt --review --review-file reviews.csv

Scoring convention:
    every rule is worth 1 point -- 1/1 when met, 0/1 when not met. Nothing scores
    negative. A rule that no code path could decide raises a HUMAN REVIEW FLAG:
    it takes the else branch (0) provisionally and a human rules it 1 or 0, so in
    the final score every rule is either 1/1 or 0/1.
Checks in the `expert` tier need astronomical judgement; they are answered from a
ground-truth SB or the LLM judge, and raised for human review when neither can decide.

The check registry (which checks to run, points, tier, provenance) lives in
rubric.yaml.  The REFERENCE DATA below (hardware allowlist, config schedule,
cycle-time table, ...) is the ground truth copied from the RADAR document and is
meant to be edited here.
"""

import os
import re
import sys
import csv
import math

# --------------------------------------------------------------------------- #
#  REFERENCE DATA  (ground truth -- edit here)                                 #
# --------------------------------------------------------------------------- #

# RADAR sec2 -- every acceptable hardware setup name.
HARDWARE_ALLOWLIST = {
    "C-point", "X-point",
    "Q64f2A", "Q64f3DCB",
    "Ka64f2A", "Ka64f3DCB",
    "K64f3DCB", "K64f2A",
    "Ku48f3DCB", "Ku48f2A", "Ku48f3DCBalt", "Ku48f2Aalt", "Ku-slew",
    "X16f3DCBalt", "X16f2A", "X16f3DCB", "X32f3DCBalt", "X32f2A", "X32f3DCB",
    "X16f2Aalt", "X-slew", "X32f2Aalt",
    "C16f5DCalt", "C32f3Bmixalt-blank", "C16f5DC", "C16f2A",
    "C32f2Amixalt-blank", "C32f5DC", "C32f2A", "C16f2Aalt", "C32f3B", "C16f3B",
    "C16f3Balt", "C32f5DCalt", "C32f3Balt", "C32f2Amixalt", "C-slew",
    "C32f3Bmixalt", "C32f2Aalt",
    "S16f2A", "S16f3B", "S16f5DC", "S-slew", "S14f2Ashiftalt-blank",
    "S14f2Ashiftalt", "S16f2Aalt", "S16f5DCalt", "S16f3Balt",
    "L16f3B", "L16f5DCalt", "L16f2A", "L16f3Balt", "L16f2Aalt", "L-slew",
    "L16f5DC",
    "4P19f2DCBA", "P16f2DCBA", "P-slew", "4_3f2DCBA",
    "eLWA",
}

# Notes comment 6 -- allowed scan intents (token -> human name).
VALID_INTENTS = {
    "SetAtnGain": "Setup Intent",
    "CalGain":    "Calibrate Complex Gain (A and P)",
    "CalFlux":    "Calibrate Flux Density Scale",
    "CalBP":      "Calibrate Bandpass",
    "CalDelay":   "Calibrate Delay",
    "ObsTgt":     "Observe Target",
}

# RADAR sec6 -- standard flux calibrators.
STANDARD_FLUX_CALS = {"3C286", "3C48", "3C147"}
# Flaring standard calibrators to avoid. This is TIME-SENSITIVE -- update it from the
# NRAO flux-density-scale page, which lists calibrators currently undergoing a flare:
#   https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/fdscale
# A different standard calibrator is an acceptable substitution as long as it is not
# in this list (and is not too far from the target position).
FLARING_FLUX_CALS = {"3C138"}

# RADAR sec4 -- VLA array-configuration schedule (start, end inclusive, config).
CONFIG_SCHEDULE = [
    ("2026-02-20", "2026-06-22", "A"),
    ("2026-07-10", "2026-10-19", "D"),
    ("2026-10-29", "2027-02-15", "C"),
    ("2027-03-03", "2027-06-14", "B"),
    ("2027-06-25", "2027-10-25", "A"),
]

# RADAR sec8 -- recommended complex-gain cycle time (minutes) by band group / config.
CYCLE_TIME_MIN = {
    "L": {"A": 15, "B": 15, "C": 15, "D": 15},
    "S": {"A": 15, "B": 15, "C": 15, "D": 15},
    "C": {"A": 8,  "B": 10, "C": 10, "D": 10},
    "X": {"A": 8,  "B": 10, "C": 10, "D": 10},
    "Ku": {"A": 6, "B": 7,  "C": 8,  "D": 8},
    "K": {"A": 4,  "B": 5,  "C": 6,  "D": 6},
    "Ka": {"A": 3, "B": 4,  "C": 5,  "D": 6},
    "Q": {"A": 2,  "B": 3,  "C": 4,  "D": 5},
}

HIGH_FREQ_BANDS = {"Ku", "K", "Ka", "Q"}

# RADAR sec5 -- phase-cal angular-separation thresholds (degrees) by band group.
# (good_max, warn_max): <=good_max PASS, <=warn_max REVIEW, else FAIL.
SEPARATION_THRESHOLDS = {
    "L": (10, 15), "S": (10, 15),
    "C": (7, 10),  "X": (7, 10),
    "Ku": (3, 10), "K": (3, 10), "Ka": (3, 10), "Q": (3, 10),
}

# Approximate VLA synthesized-beam FWHM (arcsec) by band and array config.
# Source: NRAO VLA observing guide "resolution" table. Used by coord_fwhm_match:
# the target pointing must land within about one beam of its true position.
BEAM_FWHM_ARCSEC = {
    "P":  {"A": 5.6,   "B": 18.5,  "C": 65.0, "D": 200.0},
    "L":  {"A": 1.3,   "B": 4.3,   "C": 14.0, "D": 46.0},
    "S":  {"A": 0.65,  "B": 2.1,   "C": 7.0,  "D": 23.0},
    "C":  {"A": 0.33,  "B": 1.0,   "C": 3.5,  "D": 12.0},
    "X":  {"A": 0.20,  "B": 0.60,  "C": 2.1,  "D": 7.2},
    "Ku": {"A": 0.13,  "B": 0.42,  "C": 1.4,  "D": 4.6},
    "K":  {"A": 0.089, "B": 0.28,  "C": 0.95, "D": 3.1},
    "Ka": {"A": 0.059, "B": 0.19,  "C": 0.63, "D": 2.1},
    "Q":  {"A": 0.043, "B": 0.14,  "C": 0.47, "D": 1.5},
}

# Authoritative target positions, loaded from targets.csv (name,ra,dec) if present.
# Populated by load_targets(); empty means coord_fwhm_match falls back to the LLM.
TARGETS = {}

# Phrases in the SB comment that waive the reference-pointing requirement.
REFPOINT_WAIVERS = ["not required", "not needed", "no reference pointing",
                    "by request", "pointing observations are not required"]

# Tolerance on total SB duration vs the observing time the user asked for.
#
# The check is ONE-SIDED: an SB that overruns its allocation cannot be executed
# as scheduled and fails, while one that comes in under is valid (it merely
# leaves time unused) and passes, with the shortfall shown in the report.
#
# Raise this if scoring the summed DUR fields, which ignore slew and settle time
# and therefore understate the duration the OPT reports -- most for SBs built
# from many short scans. Leave it at 0 when scoring an OPT-reported total (the
# optional 'opt_total' column of requests.csv), which already includes overheads.
TOTAL_TIME_TOL_SEC = 0

# Requested total durations in seconds, keyed by SB basename (see load_requests).
#
# IMPORTANT: the requested time is the observer's input to the LLM and must come
# from outside the SB. Do NOT read it from the SCHED-BLOCK comment -- the
# generator writes its own "total duration approximately ..." there, so checking
# against that would be circular and would pass unconditionally.
REQUESTED = {}

# Band prefixes, longest first so "Ka"/"Ku" win over "K".
_BAND_PREFIXES = ["Ka", "Ku", "4P", "4_", "P", "L", "S", "C", "X", "K", "Q"]


# --------------------------------------------------------------------------- #
#  Minimal YAML loader (supports exactly the subset used by rubric.yaml)       #
# --------------------------------------------------------------------------- #

def load_rubric(path):
    """Parse rubric.yaml -> {"meta": {...}, "checks": [ {...}, ... ]}."""
    meta, checks = {}, []
    section = None
    item = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()

            if indent == 0 and stripped.endswith(":"):
                section = stripped[:-1]
                item = None
                continue

            if section == "meta" and indent == 2:
                k, v = _split_kv(stripped)
                meta[k] = v
            elif section == "checks":
                if stripped.startswith("- "):
                    item = {}
                    checks.append(item)
                    k, v = _split_kv(stripped[2:])
                    item[k] = v
                elif item is not None:
                    k, v = _split_kv(stripped)
                    item[k] = v
    return {"meta": meta, "checks": checks}


def _split_kv(text):
    key, _, val = text.partition(":")
    key = key.strip()
    val = val.strip().strip('"').strip("'")
    if re.fullmatch(r"-?\d+", val):
        val = int(val)
    return key, val


# --------------------------------------------------------------------------- #
#  SB file parsing                                                             #
# --------------------------------------------------------------------------- #

class Scan:
    """One STD / PTG / IP scan line."""
    def __init__(self, fields, lineno):
        self.lineno = lineno
        self.type = fields[0]
        self.name = fields[1] if len(fields) > 1 else ""
        self.source = fields[2] if len(fields) > 2 else ""
        self.resource = fields[3] if len(fields) > 3 else ""
        self.duration = _parse_dur(fields[5]) if len(fields) > 5 else 0
        self.intents = [f for f in fields
                        if f.rstrip(",") in VALID_INTENTS
                        or "," in f and any(t.strip() in VALID_INTENTS
                                            for t in f.split(","))]
        # flatten intent tokens
        toks = []
        for f in fields:
            for t in f.split(","):
                t = t.strip()
                if t in VALID_INTENTS:
                    toks.append(t)
        self.intent_tokens = toks
        self.raw_intent_field = _raw_intent_field(fields)

    def has(self, intent):
        return intent in self.intent_tokens

    @property
    def is_refpointing(self):
        return self.type in ("PTG", "IP")


class SB:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.lines = []           # (lineno, kind, fields)
        self.scans = []           # list[Scan]
        self.sched_block = None   # fields of SCHED-BLOCK
        self.version = None
        self.has_src_cat = False
        self.has_hdwr_cat = False
        self._parse()

    def _parse(self):
        with open(self.path, encoding="utf-8") as fh:
            for i, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                fields = [f.strip() for f in raw.strip().rstrip(";").split(";")]
                kind = fields[0]
                self.lines.append((i, kind, fields))
                if kind == "VERSION" and self.version is None:
                    self.version = fields[1] if len(fields) > 1 else ""
                elif kind == "SRC-CAT":
                    self.has_src_cat = True
                elif kind == "HDWR-CAT":
                    self.has_hdwr_cat = True
                elif kind == "SCHED-BLOCK":
                    self.sched_block = fields
                elif kind in ("STD", "PTG", "IP"):
                    self.scans.append(Scan(fields, i))

    # -- SCHED-BLOCK accessors (see RADAR sec7 example) --
    @property
    def config(self):
        return self.sched_block[7] if self.sched_block and len(self.sched_block) > 7 else None

    @property
    def start_date(self):
        # field 4 = "YYYY-MM-DD HH:MM:SS,YYYY-MM-DD HH:MM:SS"
        if not self.sched_block or len(self.sched_block) < 5:
            return None
        m = re.search(r"\d{4}-\d{2}-\d{2}", self.sched_block[4])
        return m.group(0) if m else None

    @property
    def comment(self):
        return self.sched_block[-1] if self.sched_block else ""

    @property
    def science_bands(self):
        """Bands of science resources (exclude *-point and *-slew helpers)."""
        bands = set()
        for s in self.scans:
            r = s.resource
            if r.endswith("-point") or r.endswith("-slew"):
                continue
            b = band_of_resource(r)
            if b:
                bands.add(b)
        return bands


def _parse_dur(text):
    """'0h 6m 30s' -> seconds."""
    h = m = s = 0
    for val, unit in re.findall(r"(\d+)\s*([hms])", text):
        if unit == "h":
            h = int(val)
        elif unit == "m":
            m = int(val)
        elif unit == "s":
            s = int(val)
    return h * 3600 + m * 60 + s


def _raw_intent_field(fields):
    for f in fields:
        if any(t.strip() in VALID_INTENTS for t in f.split(",")):
            return f
    return ""


def band_of_resource(res):
    for p in _BAND_PREFIXES:
        if res.startswith(p):
            return "P" if p in ("4P", "4_") else p
    return None


# -- coordinates -------------------------------------------------------------

def parse_ra_deg(text):
    """'21h03m10.107s' or '21:03:10.107' -> degrees, or None."""
    if not text:
        return None
    m = re.search(r"(\d+)[h:]\s*(\d+)[m:]\s*([\d.]+)s?", text)
    if not m:
        return None
    h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return (h + mi / 60 + s / 3600) * 15.0

def parse_dec_deg(text):
    """'+20d45m07.58s' or '-06:22:10' -> degrees, or None."""
    if not text:
        return None
    m = re.search(r"([+\-]?)(\d+)[d:]\s*(\d+)[m:]\s*([\d.]+)s?", text)
    if not m:
        return None
    sign = -1.0 if m.group(1) == "-" else 1.0
    d, mi, s = float(m.group(2)), float(m.group(3)), float(m.group(4))
    return sign * (d + mi / 60 + s / 3600)

def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation of two (RA, Dec) points in degrees -> arcsec."""
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    dra, ddec = r2 - r1, d2 - d1
    a = math.sin(ddec / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(dra / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a)))) * 3600.0


# --------------------------------------------------------------------------- #
#  The observing request, parsed from the prompt given to the LLM              #
# --------------------------------------------------------------------------- #
#
# examples/<target>_<band>_prompt.txt holds the natural-language request the SB
# was generated from. It is the authoritative statement of what the observer
# asked for -- the target name, its position, and the total time -- so it, not
# the SB, is what the SB should be checked against.

_PROMPT_TARGET = re.compile(r"targeting source\s+([A-Za-z0-9+.\-]+)", re.I)
_PROMPT_RA = re.compile(r"\bRA[:\s]\s*(\d{1,2}h\s*\d{1,2}m\s*[\d.]+s)", re.I)
_PROMPT_DEC = re.compile(r"\bDec[:\s]\s*([+\-]?\d{1,2}d\s*\d{1,2}m\s*[\d.]+s)", re.I)
_PROMPT_TIME = re.compile(r"total time of strictly\s+(.+?)(?=,|\.|\s+attempt)", re.I)

# Parsed prompts, keyed by SB basename. None means "no prompt file found".
PROMPTS = {}


def parse_prompt(text):
    """Pull target name, position, and requested total time out of a prompt."""
    out = {"target": None, "ra": None, "dec": None, "total_seconds": None}
    m = _PROMPT_TARGET.search(text)
    if m:
        out["target"] = m.group(1).rstrip(".,")
    m = _PROMPT_RA.search(text)
    if m:
        out["ra"] = parse_ra_deg(m.group(1))
    m = _PROMPT_DEC.search(text)
    if m:
        out["dec"] = parse_dec_deg(m.group(1))
    m = _PROMPT_TIME.search(text)
    if m:
        t = m.group(1).lower()
        for word, sym in (("hours", "h"), ("hour", "h"), ("minutes", "m"),
                          ("minute", "m"), ("mins", "m"), ("min", "m")):
            t = t.replace(word, sym)
        out["total_seconds"] = parse_duration_token(t)
    return out


def find_prompt(sb_path):
    """Locate the prompt paired with an SB: <target>_<band>_prompt.txt beside it."""
    folder = os.path.dirname(os.path.abspath(sb_path)) or "."
    toks = os.path.basename(sb_path).split("_")
    if len(toks) < 2:
        return None
    cand = os.path.join(folder, f"{toks[0]}_{toks[1]}_prompt.txt")
    return cand if os.path.isfile(cand) else None


def prompt_for(sb):
    """Parsed prompt for this SB, or None. Cached per SB name."""
    if sb.name not in PROMPTS:
        path = find_prompt(sb.path)
        data = None
        if path:
            with open(path, encoding="utf-8") as fh:
                data = parse_prompt(fh.read())
        PROMPTS[sb.name] = data
    return PROMPTS[sb.name]


# --------------------------------------------------------------------------- #
#  Source catalog (SCT) generated alongside the SB                             #
# --------------------------------------------------------------------------- #
#
# The SB references sources by NAME only; the positions the OPT actually uses
# live in the source catalog the observer loads alongside it. The generator now
# emits that catalog too, so the declared position is checkable. Format is the
# VLA SCT/PST source list, one source per line:
#
#   NAME;alias;Equatorial;J2000;HH:MM:SS.sss;+DD:MM:SS.ss;;;;N;
#
# '*' names the catalog, '#' is a comment.
SOURCE_CATALOG = {}


def load_source_catalog(path):
    """Parse a VLA SCT source list into {name: (ra_deg, dec_deg)}."""
    SOURCE_CATALOG.clear()
    if not path or not os.path.isfile(path):
        return SOURCE_CATALOG
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(("*", "#")):
                continue
            f = [x.strip() for x in line.split(";")]
            if len(f) < 6 or not f[0]:
                continue
            ra, dec = parse_ra_deg(f[4]), parse_dec_deg(f[5])
            if ra is not None and dec is not None:
                SOURCE_CATALOG[f[0]] = (ra, dec)
    return SOURCE_CATALOG


def find_source_catalog(folder):
    """Locate a *source_catalog*.txt beside the SBs, or None."""
    if not os.path.isdir(folder):
        return None
    for name in sorted(os.listdir(folder)):
        if "source_catalog" in name.lower() and name.lower().endswith(".txt"):
            return os.path.join(folder, name)
    return None


def catalog_position(name):
    """Position of `name` in the source catalog, tolerating _Nas suffixes."""
    if name in SOURCE_CATALOG:
        return SOURCE_CATALOG[name]
    for cat_name, pos in SOURCE_CATALOG.items():
        if _same_source(cat_name, name):
            return pos
    return None


def load_targets(path):
    """Load authoritative target positions from a CSV with columns name,ra,dec."""
    TARGETS.clear()
    if not path or not os.path.isfile(path):
        return TARGETS
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("name") or "").strip()
            if not name or name.startswith("#"):
                continue
            TARGETS[name] = (parse_ra_deg(row.get("ra")), parse_dec_deg(row.get("dec")))
    return TARGETS


def sb_target_coords(sb):
    """RA/Dec the SB intends to point at, read from the SCHED-BLOCK comment."""
    c = sb.comment or ""
    ra = re.search(r"RA\s*([0-9]{1,2}[hH:][0-9dms.:+\- ]*?)(?=\s*Dec|,|;|$)", c, re.I)
    dec = re.search(r"Dec\s*([+\-]?[0-9]{1,2}[dD:][0-9dms.:+\- ]*?)(?=,|;|phase|flux|$)", c, re.I)
    return (parse_ra_deg(ra.group(1)) if ra else None,
            parse_dec_deg(dec.group(1)) if dec else None)


def _loop_repeat(fields):
    """Repeat count of a LOOP-START line, e.g. 'LOOP-START; name; 17; Y; ;' -> 17."""
    if len(fields) > 2:
        digits = re.sub(r"\D", "", fields[2])
        return int(digits) if digits else 1
    return 1


def total_on_source_seconds(sb):
    """Total target (ObsTgt) time, counting each loop's repeat count."""
    total = 0
    loop_stack = []
    for lineno, kind, fields in sb.lines:
        if kind == "LOOP-START":
            loop_stack.append([_loop_repeat(fields), 0])   # [repeat, per-iter secs]
        elif kind == "LOOP-END":
            if loop_stack:
                rep, secs = loop_stack.pop()
                total += rep * secs
        elif kind in ("STD", "PTG", "IP") and loop_stack:
            sc = next((s for s in sb.scans if s.lineno == lineno), None)
            if sc and sc.has("ObsTgt"):
                loop_stack[-1][1] += sc.duration
    return total


def total_sb_seconds(sb):
    """Total duration of every scan, counting each loop's repeat count.

    This is the wall-clock length of the SB: setup, slews, calibration, and
    target scans alike. Nested loops multiply.
    """
    total = 0
    loop_stack = []
    for lineno, kind, fields in sb.lines:
        if kind == "LOOP-START":
            loop_stack.append([_loop_repeat(fields), 0])   # [repeat, per-iter secs]
        elif kind == "LOOP-END":
            if loop_stack:
                rep, secs = loop_stack.pop()
                sub = rep * secs
                if loop_stack:
                    loop_stack[-1][1] += sub
                else:
                    total += sub
        elif kind in ("STD", "PTG", "IP"):
            sc = next((s for s in sb.scans if s.lineno == lineno), None)
            if sc is None:
                continue
            if loop_stack:
                loop_stack[-1][1] += sc.duration
            else:
                total += sc.duration
    return total


def parse_duration_token(text):
    """'2hr' / '3h25m' / '1h' / '90m' -> seconds, or None if unrecognised."""
    if not text:
        return None
    t = str(text).strip().lower().replace(" ", "")
    m = re.fullmatch(r"(?:(\d+)h(?:r|rs)?)?(?:(\d+)m(?:in)?)?(?:(\d+)s)?", t)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s


def requested_seconds(sb):
    """Requested total duration for this SB, or None if unknown.

    Priority: the prompt the SB was generated from ("total time of strictly
    ..."), then the requests catalog (load_requests), then a duration token in
    the filename, e.g. 'SN2026gzf_Cband_Aconfig_1hr_AItest.txt'. Never the SB's
    own comment (see the REQUESTED note above).
    """
    pr = prompt_for(sb)
    if pr and pr.get("total_seconds"):
        return pr["total_seconds"]
    if sb.name in REQUESTED:
        return REQUESTED[sb.name]
    stem = os.path.splitext(sb.name)[0]
    for tok in stem.split("_"):
        secs = parse_duration_token(tok)
        if secs:
            return secs
    return None


def _duration_field(raw):
    """A duration token ('2hr', '3h25m') or plain minutes -> seconds, else None."""
    raw = (raw or "").strip()
    if not raw:
        return None
    secs = parse_duration_token(raw)
    if secs is not None:
        return secs
    try:
        return float(raw) * 60.0
    except ValueError:
        return None


def load_requests(path):
    """Load requested durations from a CSV of 'name,requested' rows.

    'name' is the SB filename and 'requested' the time the observer asked for,
    as a duration token ('2hr', '3h25m') or plain minutes. Only used when no
    ground-truth SB is available. Blank lines and '#' comments are skipped.
    """
    REQUESTED.clear()
    if not path or not os.path.isfile(path):
        return REQUESTED
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("name") or "").strip()
            if not name or name.startswith("#"):
                continue
            req = _duration_field(row.get("requested"))
            if req is not None:
                REQUESTED[name] = req
    return REQUESTED


# --------------------------------------------------------------------------- #
#  Check result                                                               #
# --------------------------------------------------------------------------- #

PASS, FAIL, WAIVED, REVIEW, FLAG = "PASS", "FAIL", "WAIVED", "REVIEW", "FLAG"

class Result:
    def __init__(self, status, detail="", source=""):
        self.status = status
        self.detail = detail
        self.source = source   # who decided: AUTO/CALC/CATALOG/GROUND/LLM/MANUAL


# --------------------------------------------------------------------------- #
#  Check functions  (registry keyed by rubric id)                             #
# --------------------------------------------------------------------------- #

def c_version_present(sb):
    first = sb.lines[0] if sb.lines else None
    if first and first[1] == "VERSION" and sb.version == "7":
        return Result(PASS, "VERSION; 7;")
    return Result(FAIL, f"first line is {first[1] if first else 'missing'}, version={sb.version}")

def c_src_cat_present(sb):
    return Result(PASS) if sb.has_src_cat else Result(FAIL, "no SRC-CAT line")

def c_hdwr_cat_present(sb):
    return Result(PASS) if sb.has_hdwr_cat else Result(FAIL, "no HDWR-CAT line")

def c_loop_closure(sb):
    starts = sum(1 for _, k, _ in sb.lines if k == "LOOP-START")
    ends = sum(1 for _, k, _ in sb.lines if k == "LOOP-END")
    if starts == ends:
        return Result(PASS, f"{starts} loops closed")
    return Result(FAIL, f"{starts} LOOP-START vs {ends} LOOP-END")

def c_dummy_setup_first(sb):
    if sb.scans and sb.scans[0].has("SetAtnGain"):
        return Result(PASS, f"first scan '{sb.scans[0].name}'")
    first = sb.scans[0].name if sb.scans else "none"
    return Result(FAIL, f"first scan '{first}' has no SetAtnGain intent")

def c_hardware_in_allowlist(sb):
    bad = sorted({s.resource for s in sb.scans if s.resource not in HARDWARE_ALLOWLIST})
    if not bad:
        return Result(PASS, "all resources valid")
    return Result(FAIL, "invalid: " + ", ".join(bad))

def c_intents_valid(sb):
    bad = set()
    for s in sb.scans:
        if not s.raw_intent_field:
            continue
        for t in s.raw_intent_field.split(","):
            t = t.strip()
            if t and t not in VALID_INTENTS:
                bad.add(t)
    if not bad:
        return Result(PASS)
    return Result(FAIL, "unknown intents: " + ", ".join(sorted(bad)))

def c_flux_present(sb):
    return Result(PASS) if any(s.has("CalFlux") for s in sb.scans) else Result(FAIL, "no CalFlux scan")

def c_bandpass_present(sb):
    return Result(PASS) if any(s.has("CalBP") for s in sb.scans) else Result(FAIL, "no CalBP scan")

def c_flux_cal_standard(sb):
    flux_scans = [s for s in sb.scans if s.has("CalFlux")]
    if not flux_scans:
        return Result(FAIL, "no flux calibrator scan")
    for s in flux_scans:
        if any(std in s.source for std in STANDARD_FLUX_CALS):
            return Result(PASS, s.source)
        if any(fl in s.source for fl in FLARING_FLUX_CALS):
            return Result(REVIEW, f"{s.source} is flaring (3C138)")
    return Result(FAIL, "non-standard flux cal: " + flux_scans[0].source)

def c_phase_present(sb):
    return Result(PASS) if any(s.has("CalGain") for s in sb.scans) else Result(FAIL, "no CalGain scan")

def c_target_present(sb):
    return Result(PASS) if any(s.has("ObsTgt") for s in sb.scans) else Result(FAIL, "no ObsTgt scan")

def c_one_phase_per_target_loop(sb):
    """Within each LOOP block that has ObsTgt scans, exactly one CalGain line."""
    problems = []
    loop_stack = []
    loops = []
    for lineno, kind, fields in sb.lines:
        if kind == "LOOP-START":
            loop_stack.append({"name": fields[1] if len(fields) > 1 else "?",
                               "gain": 0, "tgt": 0})
        elif kind == "LOOP-END":
            if loop_stack:
                loops.append(loop_stack.pop())
        elif kind in ("STD", "PTG", "IP") and loop_stack:
            sc = next((s for s in sb.scans if s.lineno == lineno), None)
            if sc:
                if sc.has("ObsTgt"):
                    loop_stack[-1]["tgt"] += 1
                if sc.has("CalGain"):
                    loop_stack[-1]["gain"] += 1
    target_loops = [l for l in loops if l["tgt"] > 0]
    if not target_loops:
        return Result(FAIL, "no target loop found")
    for l in target_loops:
        if l["gain"] != 1:
            problems.append(f"{l['name']} has {l['gain']} phase lines")
    if problems:
        return Result(FAIL, "; ".join(problems))
    return Result(PASS, f"{len(target_loops)} target loop(s), 1 phase line each")

def c_template_order(sb):
    def first_idx(pred):
        for i, s in enumerate(sb.scans):
            if pred(s):
                return i
        return None
    i_setup = first_idx(lambda s: s.has("SetAtnGain"))
    i_flux = first_idx(lambda s: s.has("CalFlux"))
    i_gain = first_idx(lambda s: s.has("CalGain"))
    i_tgt = first_idx(lambda s: s.has("ObsTgt"))
    if None in (i_flux, i_gain, i_tgt):
        return Result(FAIL, "missing flux/phase/target scan")
    ok = (i_setup == 0) and (i_flux < i_gain < i_tgt)
    if ok:
        return Result(PASS, "setup -> flux -> phase -> target")
    return Result(FAIL, f"order idx setup={i_setup} flux={i_flux} gain={i_gain} tgt={i_tgt}")

def c_config_matches_date(sb):
    date, cfg = sb.start_date, sb.config
    if not date or not cfg:
        return Result(FAIL, "cannot read date/config")
    for start, end, config in CONFIG_SCHEDULE:
        if start <= date <= end:
            if config == cfg:
                return Result(PASS, f"{date} -> {config}")
            return Result(FAIL, f"{date} should be config {config}, SB says {cfg}")
    return Result(REVIEW, f"{date} outside known config schedule (SB says {cfg})")

def c_highfreq_refpointing(sb):
    hf = sb.science_bands & HIGH_FREQ_BANDS
    if not hf:
        return Result(PASS, "not a high-frequency SB")
    has_rp = any(s.is_refpointing for s in sb.scans)
    if has_rp:
        return Result(PASS, f"reference pointing present ({', '.join(sorted(hf))})")
    comment = sb.comment.lower()
    if any(w in comment for w in REFPOINT_WAIVERS):
        return Result(WAIVED, "no ref pointing, but waived in SB comment")
    return Result(FAIL, f"{', '.join(sorted(hf))} SB missing X-band reference pointing")

def c_target_matches_prompt(sb):
    """Target named and positioned as the prompt asked for.

    The SB references its target by NAME only; the position the OPT uses comes
    from the source catalog generated alongside the SB. So this check resolves
    the SB's target name in that catalog and compares the catalogued position
    with the one the prompt specified, passing when the two agree to within the
    synthesized-beam FWHM for the SB's highest-frequency band and configuration.

    The RA/Dec written into the SCHED-BLOCK comment is decorative -- it has no
    effect on where the telescope points -- so a disagreement there is reported
    in the detail but never decides the verdict.
    """
    pr = prompt_for(sb)
    if not pr:
        return Result(FLAG, "no prompt file found for this SB")
    if not SOURCE_CATALOG:
        return Result(FLAG, "no source catalog loaded (use --catalog)")

    want = pr.get("target")
    got = [s_.source for s_ in sb.scans if s_.has("ObsTgt")]
    tname = got[0] if got else None
    if not tname:
        return Result(FAIL, "SB has no target scan", source="PROMPT")
    if want and not _same_source(want, tname):
        return Result(FAIL, f"prompt asked for '{want}', SB targets '{tname}'",
                      source="PROMPT")

    pos = catalog_position(tname)
    if pos is None:
        return Result(FAIL, f"target '{tname}' has no source-catalog entry",
                      source="PROMPT")
    if pr.get("ra") is None or pr.get("dec") is None:
        return Result(FLAG, f"'{tname}' in catalog, but prompt states no position")

    sep = angular_sep_arcsec(pr["ra"], pr["dec"], pos[0], pos[1])
    fwhms = [BEAM_FWHM_ARCSEC[b][sb.config]
             for b in sb.science_bands
             if b in BEAM_FWHM_ARCSEC and sb.config in BEAM_FWHM_ARCSEC.get(b, {})]

    # The SCHED-BLOCK comment position is advisory; note it, never score it.
    note = ""
    sb_ra, sb_dec = sb_target_coords(sb)
    if None in (sb_ra, sb_dec):
        note = "; SB comment states no position"
    else:
        csep = angular_sep_arcsec(pr["ra"], pr["dec"], sb_ra, sb_dec)
        if not fwhms or csep > min(fwhms):
            note = f"; note: SCHED-BLOCK comment position differs from prompt by {csep:.2f}\""

    if not fwhms:
        return Result(REVIEW,
                      f"{tname}: catalog offset {sep:.3f}\" but no beam size "
                      f"for band/config{note}", source="PROMPT")
    fwhm = min(fwhms)
    if sep <= fwhm:
        return Result(PASS,
                      f"{tname}: catalog position matches prompt, offset "
                      f"{sep:.3f}\" <= beam {fwhm}\"{note}", source="PROMPT")
    return Result(FAIL,
                  f"{tname}: catalog position off by {sep:.2f}\" > beam "
                  f"{fwhm}\"{note}", source="PROMPT")


def _same_source(a, b):
    """Compare source names ignoring case, the SN/AT prefix, and _Nas suffixes.

    'AT2025ulz' == 'SN2025ulz' == 'SN2025ulz_1as'; 'GW170817' == 'GW170817_7as'.
    """
    def norm(n):
        n = re.sub(r"_\d+as$", "", n.strip(), flags=re.I)
        n = re.sub(r"^(SN|AT|GW)", "", n, flags=re.I)
        return n.lower()
    return norm(a) == norm(b)


def c_total_time_matches_request(sb):
    """Total SB duration vs the observing time the user requested.

    The observer derives a total allocation from a sensitivity calculation and
    asks the LLM for it; the SB has to fit. How that block is divided between
    overheads and target scans is a strategy choice and is reported, not scored
    (see onsource_note in llm_judge.py).

    One-sided: overrunning the allocation fails, since the SB cannot then run as
    scheduled; coming in under passes, since the SB is still executable and only
    leaves time unused.

    FLAGs when the requested duration is unknown, so the check falls through to
    the LLM/manual path rather than silently passing.
    """
    req = requested_seconds(sb)
    if not req:
        return Result(FLAG, "requested duration unknown (use --requested or requests.csv)")
    # Sum of the declared DUR fields. This is a LOWER BOUND: it omits the slew
    # and settle time the OPT adds at every source change, and the gap grows
    # with the number of scans.
    got = total_sb_seconds(sb)
    if not got:
        return Result(FLAG, "cannot measure SB duration")
    delta = got - req
    summary = f"SB {got/60:.1f}m vs requested {req/60:.1f}m ({delta:+.0f}s)"
    if delta > TOTAL_TIME_TOL_SEC:
        over = f" by >{TOTAL_TIME_TOL_SEC:.0f}s" if TOTAL_TIME_TOL_SEC else ""
        return Result(FAIL, f"{summary}, overruns allocation{over}", source="REQUEST")
    if delta < 0:
        return Result(PASS, f"{summary}, fits allocation with {-delta:.0f}s unused",
                      source="REQUEST")
    return Result(PASS, f"{summary}, fits allocation", source="REQUEST")


def c_cycle_time(sb):
    """Advisory: total time per phase->phase loop iteration vs recommended cycle."""
    cfg = sb.config
    # find each target loop's single-iteration duration
    loop_stack = []
    reports = []
    for lineno, kind, fields in sb.lines:
        if kind == "LOOP-START":
            loop_stack.append({"name": fields[1] if len(fields) > 1 else "?",
                               "dur": 0, "tgt": 0, "band": None})
        elif kind == "LOOP-END":
            if loop_stack:
                l = loop_stack.pop()
                if l["tgt"] > 0 and l["band"]:
                    reports.append(l)
        elif kind in ("STD", "PTG", "IP") and loop_stack:
            sc = next((s for s in sb.scans if s.lineno == lineno), None)
            if sc:
                loop_stack[-1]["dur"] += sc.duration
                if sc.has("ObsTgt"):
                    loop_stack[-1]["tgt"] += 1
                    if not loop_stack[-1]["band"]:
                        loop_stack[-1]["band"] = band_of_resource(sc.resource)
    if not reports or not cfg:
        return Result(REVIEW, "no measurable target loop")
    msgs = []
    ok = True
    for l in reports:
        rec = CYCLE_TIME_MIN.get(l["band"], {}).get(cfg)
        if rec is None:
            continue
        got_min = l["dur"] / 60.0
        flag = "" if got_min <= rec * 1.2 else "  (>guidance)"
        if got_min > rec * 1.2:
            ok = False
        msgs.append(f"{l['name']}: {got_min:.1f}m loop vs {rec}m rec{flag}")
    if not msgs:
        return Result(REVIEW, "band/config not in cycle-time table")
    return Result(PASS if ok else REVIEW, "; ".join(msgs))


CHECKS = {
    "version_present": c_version_present,
    "src_cat_present": c_src_cat_present,
    "hdwr_cat_present": c_hdwr_cat_present,
    "loop_closure": c_loop_closure,
    "dummy_setup_first": c_dummy_setup_first,
    "hardware_in_allowlist": c_hardware_in_allowlist,
    "intents_valid": c_intents_valid,
    "flux_present": c_flux_present,
    "bandpass_present": c_bandpass_present,
    "flux_cal_standard": c_flux_cal_standard,
    "phase_present": c_phase_present,
    "target_present": c_target_present,
    "one_phase_per_target_loop": c_one_phase_per_target_loop,
    "template_order": c_template_order,
    "config_matches_date": c_config_matches_date,
    "highfreq_refpointing": c_highfreq_refpointing,
    "cycle_time": c_cycle_time,
    "target_matches_prompt": c_target_matches_prompt,
    "total_time_matches_request": c_total_time_matches_request,
}

# Symbols for the report.
SYMBOL = {PASS: "PASS", FAIL: "FAIL", WAIVED: "WAIV", REVIEW: "REVW", FLAG: "FLAG"}


def award_points(chk, meta, status):
    """Points a check contributes to the score, given its verdict.

    Configured entirely from rubric.yaml:
      PASS  -> chk['points']                       (default 1, so 1/1)
      FAIL  -> chk['fail_points'] if set,
               else meta['fail_default'] if set,   (the rubric ships 0, so 0/1)
               else -chk['points']                 (legacy +1/-1 behaviour)
      other -> 0   provisionally: the verdict is unresolved, so the check raises a
               HUMAN REVIEW FLAG and a human rules it 1 or 0 before the score is
               final (see apply_review_decisions / prompt_human_reviews).

    The shipped rubric sets 'fail_default: 0'. Setting a rule's own 'fail_points'
    overrides it; dropping both restores the old +1/-1 scoring.
    """
    pts = chk.get("points", 1)
    if status == PASS:
        return pts
    if status == FAIL:
        if "fail_points" in chk:
            return chk["fail_points"]
        if "fail_default" in meta:
            return meta["fail_default"]
        return -pts
    return 0


def score(sb, rubric):
    total = 0
    rows = []
    meta = rubric.get("meta", {})
    for chk in rubric["checks"]:
        cid = chk["id"]
        tier = chk.get("tier", "mechanical")
        fn = CHECKS.get(cid)
        if fn is None:
            res = Result(FLAG, "needs expert judgement / manual review")
        else:
            res = fn(sb)
        awarded = award_points(chk, meta, res.status)
        total += awarded
        rows.append((cid, chk.get("title", cid), res, awarded, tier,
                     is_informational(chk, meta), chk.get("points", 1)))
    return total, rows


def is_informational(chk, meta):
    """True when a check scores zero whether it is met or not.

    Such checks are reported for a human to read but are not one of the scored
    criteria, so they are listed separately and excluded from the pass/fail/
    flagged tallies.
    """
    if chk.get("points", 1) != 0:
        return False
    fail = chk.get("fail_points", meta.get("fail_default", -chk.get("points", 1)))
    return fail == 0


# --------------------------------------------------------------------------- #
#  Human review                                                               #
# --------------------------------------------------------------------------- #
#
# A verdict that is neither PASS nor FAIL is not a third score. The check raises a
# HUMAN REVIEW FLAG, takes the else branch (0) for now, and waits for a human to
# rule it 1 or 0. Rulings can be recorded in a CSV (--review-file) so a rerun
# reproduces the same final score without asking again.

HUMAN = "HUMAN"


def needs_human_review(res):
    """True when a verdict is neither met nor unmet, so a human has to decide."""
    return res.status not in (PASS, FAIL)


def human_result(decision, note=""):
    """Turn a human 1/0 ruling into an ordinary met/not-met verdict."""
    status = PASS if int(decision) == 1 else FAIL
    return Result(status, note or "ruled by human review", source=HUMAN)


def load_review_decisions(path):
    """Read recorded rulings from CSV: sb_name,check_id,score,note."""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 3 or row[0].lstrip().startswith("#"):
                continue
            sb_name, cid, dec = row[0].strip(), row[1].strip(), row[2].strip()
            if sb_name.lower() in ("sb_name", "sb"):        # header line
                continue
            note = row[3].strip() if len(row) > 3 else ""
            if dec in ("0", "1"):
                out[(sb_name, cid)] = (int(dec), note)
    return out


def save_review_decision(path, sb_name, cid, decision, note):
    """Append one ruling so the next run reuses it instead of re-asking."""
    if not path:
        return
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow(["sb_name", "check_id", "score", "note"])
        w.writerow([sb_name, cid, decision, note])


def apply_review_decisions(sb_name, rows, rubric, decisions):
    """Fold recorded human rulings into rows. Returns (total, rows)."""
    if not decisions:
        return sum(r[3] for r in rows), rows
    meta = rubric.get("meta", {})
    checks_by_id = {c["id"]: c for c in rubric["checks"]}
    out = []
    for row in rows:
        cid, title, res, awarded, tier = row[:5]
        rest = tuple(row[5:])
        if needs_human_review(res) and (sb_name, cid) in decisions:
            dec, note = decisions[(sb_name, cid)]
            res = human_result(dec, note)
            awarded = award_points(checks_by_id.get(cid, {}), meta, res.status)
        out.append((cid, title, res, awarded, tier) + rest)
    return sum(r[3] for r in out), out


def pending_reviews(rows):
    """Scored checks still waiting on a human ruling."""
    return [r for r in rows
            if not (len(r) > 5 and r[5]) and needs_human_review(r[2])]


def prompt_human_reviews(sb_name, rows, rubric, save_path=None, stream=None):
    """Ask a human to rule 1/0 on each unresolved check. Returns the rulings."""
    stream = stream or sys.stdin
    pending = pending_reviews(rows)
    if not pending:
        return {}
    print("=" * 78)
    print(f"HUMAN REVIEW -- {sb_name}: {len(pending)} check(s) need a ruling")
    print("  Answer 1 (met) or 0 (not met); text after the digit is kept as a note.")
    print("  Enter alone leaves the check unresolved, which scores 0.")
    print("-" * 78)
    ruled = {}
    for i, row in enumerate(pending, 1):
        cid, title, res = row[0], row[1], row[2]
        print(f"  [{i}/{len(pending)}] {cid}")
        print(f"        {title}")
        print(f"        automatic verdict: {SYMBOL[res.status]}"
              + (f" -- {res.detail}" if res.detail else ""))
        while True:
            sys.stdout.write("        score 1 or 0 (+ optional note): ")
            sys.stdout.flush()
            line = stream.readline()
            if not line:                       # EOF: nothing more to read
                print()
                print("=" * 78)
                print()
                return ruled
            line = line.strip()
            if not line:                       # left unresolved on purpose
                break
            tok, _, note = line.partition(" ")
            if tok in ("0", "1"):
                note = note.strip()
                ruled[(sb_name, cid)] = (int(tok), note)
                save_review_decision(save_path, sb_name, cid, tok, note)
                break
            print("        please answer 1 or 0 (or Enter to leave it unresolved)")
    print("=" * 78)
    print()
    return ruled


def review_sb(sb_name, total, rows, rubric, decisions=None, interactive=False,
              save_path=None, stream=None):
    """Apply recorded rulings, then optionally ask about whatever is still open."""
    total, rows = apply_review_decisions(sb_name, rows, rubric, decisions or {})
    if interactive and pending_reviews(rows):
        new = prompt_human_reviews(sb_name, rows, rubric, save_path, stream)
        if new:
            total, rows = apply_review_decisions(sb_name, rows, rubric, new)
    return total, rows


def _source_tag(tier, res):
    """Where this verdict came from, shown at the start of each report line."""
    if getattr(res, "source", ""):
        return res.source            # CATALOG / GROUND / LLM set explicitly
    if tier == "mechanical":
        return "AUTO"                # deterministic code check
    if tier == "computed":
        return "CALC"               # deterministic comparison (cycle time, duration)
    return "MANUAL"                  # expert check left unresolved


def print_report(sb, total, rows, extra_info=None, stage=""):
    print("=" * 78)
    print(f"SB: {sb.name}" + (f"   [{stage}]" if stage else ""))
    meta = []
    if sb.config:
        meta.append(f"config {sb.config}")
    if sb.start_date:
        meta.append(f"start {sb.start_date}")
    if sb.science_bands:
        meta.append("band " + "/".join(sorted(sb.science_bands)))
    if meta:
        print("    " + ", ".join(meta))
    print("    source: AUTO=code  CALC=computed  CATALOG=targets.csv  "
          "REQUEST=requested time  GROUND=ground-truth  LLM=model  HUMAN=human review  "
          "MANUAL=needs human")
    print("-" * 78)
    scored = [r for r in rows if not (len(r) > 5 and r[5])]
    info = [r for r in rows if len(r) > 5 and r[5]]
    for row in scored:
        title, res, awarded, tier = row[1], row[2], row[3], row[4]
        pts = row[6] if len(row) > 6 else 1
        mark = "  <-- HUMAN REVIEW" if needs_human_review(res) else ""
        detail = f"  -- {res.detail}" if res.detail else ""
        src = _source_tag(tier, res)
        print(f"  {src:<7} [{SYMBOL[res.status]}] {awarded}/{pts}  {title}{detail}{mark}")
    print("-" * 78)
    passes = sum(1 for r in scored if r[2].status == PASS)
    fails = sum(1 for r in scored if r[2].status == FAIL)
    pending = pending_reviews(rows)
    max_total = sum((r[6] if len(r) > 6 else 1) for r in scored)
    label = "SCORE (provisional)" if pending else "SCORE"
    print(f"  {label}: {total} / {max_total}   ({passes} met, {fails} not met, "
          f"{len(pending)} awaiting human review)")
    if pending:
        print("-" * 78)
        print(f"  HUMAN REVIEW FLAG -- {len(pending)} check(s) could not be decided "
              f"automatically.")
        print("  Each scores 0 until a human rules it 1 (met) or 0 (not met):")
        for row in pending:
            print(f"    - {row[0]}: {row[1]}")
        print("  Resolve them with:  --review   (record them with --review-file FILE)")
    extra_info = [x for x in (extra_info or []) if x]
    if info or extra_info:
        print("-" * 78)
        print("  INFORMATIONAL (not scored, not counted above):")
        for row in info:
            title, res = row[1], row[2]
            detail = f"  -- {res.detail}" if res.detail else ""
            print(f"    [{SYMBOL[res.status]}] {title}{detail}")
        for line in extra_info:
            print(f"    {line}")
    print("=" * 78)
    print()


def data_dir():
    """Directory holding rubric.yaml / targets.csv.

    Supports both a flat layout (data files next to this script) and the repo
    layout (this script in src/, data files one level up at the repo root).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.dirname(here)):
        if os.path.exists(os.path.join(cand, "rubric.yaml")):
            return cand
    return here


def main(argv):
    root = data_dir()
    rubric = load_rubric(os.path.join(root, "rubric.yaml"))

    # Optional "--targets PATH"; otherwise auto-load targets.csv from the data dir.
    args = argv[1:]
    targets_path = os.path.join(root, "targets.csv")
    if "--targets" in args:
        i = args.index("--targets")
        targets_path = args[i + 1]
        del args[i:i + 2]
    load_targets(targets_path)

    # Optional "--requests PATH"; otherwise auto-load requests.csv if present.
    requests_path = os.path.join(root, "requests.csv")
    if "--requests" in args:
        i = args.index("--requests")
        requests_path = args[i + 1]
        del args[i:i + 2]
    load_requests(requests_path)

    # Optional "--review" / "--review-file PATH": human rulings on flagged checks.
    interactive = "--review" in args
    if interactive:
        args.remove("--review")
    review_path = None
    if "--review-file" in args:
        i = args.index("--review-file")
        review_path = args[i + 1]
        del args[i:i + 2]
    decisions = load_review_decisions(review_path)

    # Optional "--catalog PATH"; otherwise look for one beside the SB files.
    catalog_path = None
    if "--catalog" in args:
        i = args.index("--catalog")
        catalog_path = args[i + 1]
        del args[i:i + 2]
    if catalog_path is None and args:
        catalog_path = find_source_catalog(os.path.dirname(os.path.abspath(args[0])) or ".")
    load_source_catalog(catalog_path)

    files = args
    if not files:
        print(__doc__)
        return 1
    for path in files:
        if not os.path.isfile(path):
            print(f"skip (not a file): {path}")
            continue
        sb = SB(path)
        total, rows = score(sb, rubric)
        total, rows = apply_review_decisions(sb.name, rows, rubric, decisions)
        print_report(sb, total, rows)
        if interactive and pending_reviews(rows):
            new_rulings = prompt_human_reviews(sb.name, rows, rubric, review_path)
            if new_rulings:
                total, rows = apply_review_decisions(sb.name, rows, rubric, new_rulings)
                print_report(sb, total, rows, stage="after human review")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
