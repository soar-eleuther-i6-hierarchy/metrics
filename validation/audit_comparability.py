"""Audit: which result files may be compared with which, and which may not.

Why this exists
---------------
On 2026-09-11 a Temporal SAE result was compared against a Matryoshka baseline that had
been graded before BOS positions were excluded from the co-firing counts. Both files
looked complete. Both loaded without error. The ratio they produced, 1,760x, was wrong by
a factor of more than two, and nothing in the pipeline objected.

Reading the files did not catch it. It surfaced by accident, during an unrelated run that
happened to regrade the same pairs. That is the problem this script addresses: a wrong
basis for a comparison leaves no trace in the numbers it produces.

What it checks
--------------
Every result file records the settings it was produced under. Two files may be compared
only when those settings agree. The script reads the settings, never the prose, and
reports three things:

1. Files that predate a guard. A report whose `config` omits `bos_excluded` or
   `min_joint` was written before that guard existed. The check is on absence, not on
   value, because a key that did not exist cannot carry a value.

2. Groups of files that share a corpus. Files with the same source, layer, document count
   and context size should report the same token count. One that does not was produced by
   different code.

3. Every finding's stated sources, resolved to files, with the verdict for each.

Exit status is non-zero when anything is flagged, so this can gate a build.

Run:  python3 -m validation.audit_comparability
      python3 -m validation.audit_comparability --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = HERE.parent
OUT = METRICS / "outputs"
ROOT = METRICS.parent
FINDINGS = ROOT / "findings"

# Keys that entered the `config` block together with the guard they name. A report that
# omits one was graded before that guard existed. Absence is the signal; a value of
# `false` would be a deliberate choice and is reported separately.
GUARD_KEYS = ("bos_excluded", "min_joint")

# Settings that must agree before two files describe the same corpus. The run directory
# is part of the key: PCFG's `fmt_0000`, `fmt_1667`, `fmt_2308` and `fmt_2400` all record
# `sae_source: pcfg, layer 2, 2000 docs, context 512`, yet they are four different corpora
# at four formatting densities, and their token counts differ for that reason. Keying on
# the config alone reported them as a conflict, which is the audit making the same class
# of mistake it exists to catch.
CORPUS_KEYS = ("sae_source", "layer", "n_docs", "context_size")


def is_metrics_report(d) -> bool:
    """A graded report, as opposed to a sweep, a label cache or a checkpoint dump."""
    return (isinstance(d, dict) and isinstance(d.get("config"), dict)
            and "block_ranges" in d["config"]
            and isinstance(d.get("pairs"), list)
            and all(isinstance(q, dict) and "pair" in q for q in d["pairs"]))


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return {"__unreadable__": str(e)}


def scan_reports():
    """Every graded report under outputs/, with the settings that decide comparability."""
    rows = []
    for p in sorted(OUT.rglob("*.json")):
        d = load(p)
        if not is_metrics_report(d):
            continue
        cfg = d["config"]
        rows.append({
            "path": str(p.relative_to(METRICS)),
            "withdrawn": "withdrawn" in p.parts,
            "tokens": d.get("total_tokens"),
            "missing_guards": [k for k in GUARD_KEYS if k not in cfg],
            "guards_off": [k for k in GUARD_KEYS
                           if cfg.get(k) is False or cfg.get(k) == 0],
            # The run directory distinguishes corpora that the config block does not.
            "corpus": tuple(cfg.get(k) for k in CORPUS_KEYS) + (p.parent.name,),
            "pairs": [q.get("pair") for q in d["pairs"]],
        })
    return rows


def check_guards(rows):
    """Files graded before a guard existed. These may not baseline anything current."""
    bad = [r for r in rows if r["missing_guards"] and not r["withdrawn"]]
    quarantined = [r for r in rows if r["missing_guards"] and r["withdrawn"]]
    return bad, quarantined


def check_corpora(rows):
    """Files claiming the same corpus but reporting different token counts."""
    groups = defaultdict(list)
    for r in rows:
        if r["withdrawn"]:
            continue
        groups[r["corpus"]].append(r)
    split = []
    for corpus, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        counts = {m["tokens"] for m in members}
        if len(counts) > 1:
            split.append((corpus, members, sorted(counts, key=lambda x: (x is None, x))))
    return split


# A finding names its sources in fenced paths and inline code spans. Resolve those that
# look like result files; ignore scripts, directories and prose.
_PATHY = re.compile(r"`([^`\n]*?\.json)`")


def finding_sources():
    """For each finding, the result files it names, and whether each is comparable."""
    out = []
    if not FINDINGS.is_dir():
        return out
    for f in sorted(FINDINGS.glob("I6-F*.md")):
        text = f.read_text()
        named = sorted(set(_PATHY.findall(text)))
        resolved = []
        for n in named:
            # Prefer an exact path suffix. Matching on the basename alone resolved
            # `outputs/gemma-2-2b/layer_12/metrics_report.json` to every layer's report,
            # then graded the wrong one, which is the error this audit exists to find.
            # Findings write paths from the repository root; this script resolves from
            # metrics/. Strip the prefix so an exact match is possible at all.
            want = n.lstrip("./")
            if want.startswith("metrics/"):
                want = want[len("metrics/"):]
            cands = list(OUT.rglob(Path(n).name))
            exact = [p for p in cands if str(p.relative_to(METRICS)).endswith(want)]
            hits, how = (exact, "") if exact else (cands, " (basename only)")
            resolved.append((n, [str(h.relative_to(METRICS)) + how for h in hits]))
        out.append((f.name, resolved))
    return out


def check_second_pass():
    """A `second_pass.json` carries no config, so it is audited against its neighbour.

    The probe stage shortlists the candidate edges that have positive PMI, so its
    shortlist is a subset of the candidate set in the `metrics_report.json` sitting in the
    same directory. A shortlist larger than that candidate set means the two files came
    from different runs, which is exactly the condition that produced the 2026-09-11
    error. The check is deliberately loose: it catches a mismatched pair, not a small
    difference in how many of the shortlisted edges were then scored.
    """
    flags, checked = [], 0
    for sp_path in sorted(OUT.rglob("second_pass.json")):
        rep_path = sp_path.parent / "metrics_report.json"
        if not rep_path.is_file():
            flags.append((str(sp_path.relative_to(METRICS)), None,
                          "no metrics_report.json beside it, so it cannot be checked"))
            continue
        sp, rep = load(sp_path), load(rep_path)
        if not is_metrics_report(rep) or not isinstance(sp, dict):
            continue
        cand = {q["pair"]: q.get("n_candidate_edges") for q in rep["pairs"]}
        for pair, body in sp.items():
            if not isinstance(body, dict) or "sres" not in body:
                continue
            checked += 1
            n_short = body["sres"].get("n_shortlist_edges")
            n_cand = cand.get(pair)
            if n_cand is None:
                flags.append((str(sp_path.relative_to(METRICS)), pair,
                              f"pair absent from the report beside it"))
            elif n_short is not None and n_short > n_cand:
                flags.append((str(sp_path.relative_to(METRICS)), pair,
                              f"shortlist {n_short:,} exceeds the {n_cand:,} candidate "
                              f"edges in the report beside it"))
    return flags, checked


def check_sweeps():
    """A threshold sweep must say which cache it ran on, and it must agree with its neighbour.

    The gemma layer-12 sweep was first run on a pre-BOS cache. Its file held only rows, so
    nothing recorded the token count, and this audit could not see that the sweep and the
    report beside it came from different corpora. Sweep files now carry a `provenance` block;
    a file without one is flagged, and one whose token count differs from the neighbouring
    `metrics_report.json` is flagged.
    """
    flags, checked = [], 0
    for sw in sorted(OUT.rglob("threshold_sweep*.json")):
        if "withdrawn" in sw.parts:
            continue
        d = load(sw)
        rel = str(sw.relative_to(METRICS))
        if isinstance(d, dict) and "rows" not in d:
            # The trained-toy sweep scores against ground truth and reads no cache. It has
            # `default_point` and `true_edges` rather than `rows`, and nothing to be
            # provenance-checked against.
            continue
        prov = d.get("provenance") if isinstance(d, dict) else None
        if not prov:
            flags.append((rel, "no provenance block: the cache it ran on is unknown"))
            continue
        checked += 1
        rep = sw.parent / "metrics_report.json"
        if rep.is_file():
            r = load(rep)
            for pr in prov:
                if pr.get("total_tokens") != r.get("total_tokens"):
                    flags.append((rel, f"sweep ran on {pr.get('total_tokens')} tokens, the report beside it "
                                       f"has {r.get('total_tokens')}: different caches"))
                if pr.get("bos_excluded") is not True:
                    flags.append((rel, f"bos_excluded={pr.get('bos_excluded')!r} in the sweep's cache"))
    return flags, checked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None, help="also write the result here")
    args = ap.parse_args()

    rows = scan_reports()
    bad, quarantined = check_guards(rows)
    split = check_corpora(rows)

    print(f"[audit] {len(rows)} graded reports under outputs/\n")

    print("=" * 78)
    print("1. Reports graded before a guard existed")
    print("=" * 78)
    if bad:
        for r in bad:
            print(f"  FLAG  {r['path']}")
            print(f"        omits {r['missing_guards']}, reports {r['tokens']} tokens")
        print("\n  These cannot baseline a current result. Quarantine or regrade them.")
    else:
        print("  none in active directories")
    if quarantined:
        print(f"\n  {len(quarantined)} already quarantined under withdrawn/:")
        for r in quarantined:
            print(f"        {r['path']}  (omits {r['missing_guards']})")

    off = [r for r in rows if r["guards_off"] and not r["withdrawn"]]
    if off:
        print("\n  Guards present but disabled, which is a choice rather than an era:")
        for r in off:
            print(f"        {r['path']}  {r['guards_off']}")

    print("\n" + "=" * 78)
    print("2. Files claiming one corpus but reporting different token counts")
    print("=" * 78)
    if split:
        for corpus, members, counts in split:
            src, layer, docs, ctx, run = corpus
            print(f"  FLAG  {src}, layer {layer}, {docs} docs, context {ctx}, run {run}")
            print(f"        token counts seen: {counts}")
            for m in members:
                print(f"          {m['tokens']!s:>8}  {m['path']}")
    else:
        print("  none: every group agrees on its token count")

    sw_flags, sw_checked = check_sweeps()
    sp_flags, sp_checked = check_second_pass()
    print("\n" + "=" * 78)
    print("3. Probe results against the report beside them")
    print("=" * 78)
    if sp_flags:
        for path, pair, why in sp_flags:
            print(f"  FLAG  {path}" + (f"  [{pair}]" if pair else ""))
            print(f"        {why}")
    else:
        print(f"  none: {sp_checked} probe results are consistent with their reports")

    print("\n" + "=" * 78)
    print("3b. Threshold sweeps: which cache, and does it match the report beside it")
    print("=" * 78)
    if sw_flags:
        for path, why in sw_flags:
            print(f"  FLAG  {path}\n        {why}")
    else:
        print(f"  none: {sw_checked} sweep file(s) carry provenance that matches their report")

    print("\n" + "=" * 78)
    print("4. Result files named by each finding")
    print("=" * 78)
    by_path = {r["path"]: r for r in rows}
    finding_flags = 0
    for name, resolved in finding_sources():
        if not resolved:
            continue
        print(f"\n  {name}")
        for named, hits in resolved:
            if not hits:
                print(f"      {named}  -> not found under outputs/ (may be a plan or an "
                      f"input)")
                continue
            for h in hits:
                r = by_path.get(h)
                if r is None:
                    verdict = "not a graded report"
                elif r["withdrawn"]:
                    verdict = "WITHDRAWN"
                    finding_flags += 1
                elif r["missing_guards"]:
                    verdict = f"FLAG: omits {r['missing_guards']}"
                    finding_flags += 1
                else:
                    verdict = f"ok, {r['tokens']} tokens"
                print(f"      {named}  -> {h}  [{verdict}]")

    print("\n" + "=" * 78)
    total = len(bad) + len(split) + len(sp_flags) + len(sw_flags) + finding_flags
    print(f"VERDICT: {total} flag(s)" if total else "VERDICT: nothing flagged")
    print("=" * 78)

    if args.json:
        args.json.write_text(json.dumps({
            "reports": rows,
            "graded_before_a_guard": bad,
            "quarantined": quarantined,
            "split_corpora": [{"corpus": list(c), "token_counts": t,
                               "files": [m["path"] for m in ms]}
                              for c, ms, t in split],
            "probe_flags": [{"file": f, "pair": p, "why": w} for f, p, w in sp_flags],
        }, indent=2, allow_nan=False))
        print(f"wrote {args.json}")

    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
