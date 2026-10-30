"""Collect the TL;DR dataset (val bpb curves and finals) from the bucket mirrors into logbook/tldr/data.json.

    python logbook/tldr/make_data.py --live $S2/live

Each experiment: a plain-language label, family (bpe | hnet), scale (small = L03 at 1e16 FLOPs, big = L05 at 1e17),
the runs (sweep/run dirs) it averages over, its mean curve per eval step and its final value per seed."""
import argparse, glob, json, os, statistics

SMALL, BIG = "small", "big"
# id: (label, family, scale, generation, note, [run dirs relative to --live])
E = {
 # references
 "bpe": ("BPE (standard tokenizer)", "bpe", SMALL, 0, "The usual Transformer with a fixed 32k-token vocabulary.",
         ["ladder_1e16_L03/bpe_s0", "evolve_g2/bpe_s1", "evolve_g2/bpe_s2"]),
 "bpe_copy": ("BPE + copying", "bpe", SMALL, 2, "BPE given the same copy-from-earlier-in-the-document trick our H-Nets use. The fair bar.",
         ["evolve_g2/g2_bpe_match_s1", "evolve_g2/g2_bpe_match_s2", "evolve_g4/g2_bpe_match_s3"]),
 "bpe_copy_mem": ("BPE + copying + extra memory", "bpe", SMALL, 5, "BPE + copying with the generation-5 memory bolted on (more parameters than allowed, to be generous).",
         ["evolve_g5/g5_output_bpe_mem_s1", "evolve_g5/g5_output_bpe_mem_s2"]),
 "hnet": ("H-Net as published", "hnet", SMALL, 0, "Byte-level model that learns its own chunks. No hints.",
         ["ladder_1e16_L03/hnet_s0", "evolve_g1/hnet_s1", "evolve_g1/hnet_s2"]),
 "hnet_ws": ("H-Net + space hint", "hnet", SMALL, 0, "Same, nudged early in training to cut words at spaces (the hint fades out).",
         ["ladder_1e16_L03/hnet_ws_s0", "evolve_g1/hnet_ws_s1", "evolve_g1/hnet_ws_s2"]),
 # generation 1
 "g1_wild_01": ("Round 1: copying", "hnet", SMALL, 1, "Copy the byte that followed the last time this context appeared in the document.", ["evolve_g1/g1_wild_01_s1", "evolve_g1/g1_wild_01_s2"]),
 "g1_mem_01": ("Round 1: word-completion memory (trie)", "hnet", SMALL, 1, "A lookup table that scores the next byte given the start of the current chunk.", ["evolve_g1/g1_mem_01_s1", "evolve_g1/g1_mem_01_s2"]),
 "g1_mem_02": ("Round 1: suffix memory", "hnet", SMALL, 1, "Chunk embeddings keyed by the chunk's last few bytes.", ["evolve_g1/g1_mem_02_s1", "evolve_g1/g1_mem_02_s2"]),
 "g1_wild_02": ("Round 1: prefix dictionary", "hnet", SMALL, 1, "A cheaper word-completion head over the top-16 bytes.", ["evolve_g1/g1_wild_02_s1", "evolve_g1/g1_wild_02_s2"]),
 "g1_arch_02": ("Round 1: position-aware read-back", "hnet", SMALL, 1, "Bytes inside a chunk read the chunk summary differently depending on their position.", ["evolve_g1/g1_arch_02_s1", "evolve_g1/g1_arch_02_s2"]),
 "g1_opt_01": ("Round 1: learning-rate fix", "hnet", SMALL, 1, "Remove a learning-rate multiplier that double-counted with the optimizer.", ["evolve_g1/g1_opt_01_s1", "evolve_g1/g1_opt_01_s2"]),
 "g1_router_02": ("Round 1: router tweak", "hnet", SMALL, 1, "A different way of deciding where chunks end.", ["evolve_g1/g1_router_02_s1", "evolve_g1/g1_router_02_s2"]),
 "g1_alloc_01": ("Round 1: compute reallocation A", "hnet", SMALL, 1, "Move compute between the byte layers and the chunk layers.", ["evolve_g1/g1_alloc_01_s1", "evolve_g1/g1_alloc_01_s2"]),
 "g1_alloc_02": ("Round 1: compute reallocation B", "hnet", SMALL, 1, "Another split of compute between levels.", ["evolve_g1/g1_alloc_02_s1", "evolve_g1/g1_alloc_02_s2"]),
 "g1_arch_01": ("Round 1: architecture tweak", "hnet", SMALL, 1, "A small architecture change to the chunk path.", ["evolve_g1/g1_arch_01_s1", "evolve_g1/g1_arch_01_s2"]),
 "g1_opt_02": ("Round 1: optimizer tweak", "hnet", SMALL, 1, "A second optimizer change.", ["evolve_g1/g1_opt_02_s1", "evolve_g1/g1_opt_02_s2"]),
 # generation 2
 "g2_match_mem": ("Round 2: copying + suffix memory", "hnet", SMALL, 2, "Two gen-1 winners combined.", ["evolve_g2/g2_match_mem_s1", "evolve_g2/g2_match_mem_s2"]),
 "g2_match_arch": ("Round 2: copying + read-back", "hnet", SMALL, 2, "Two gen-1 winners combined.", ["evolve_g2/g2_match_arch_s1", "evolve_g2/g2_match_arch_s2"]),
 "g2_match_opt": ("Round 2: copying + LR fix", "hnet", SMALL, 2, "Two gen-1 winners combined.", ["evolve_g2/g2_match_opt_s1", "evolve_g2/g2_match_opt_s2"]),
 "g2_match_pf": ("Round 2: copying, no space hint", "hnet", SMALL, 2, "Copying on the hint-free H-Net. Learned to cut before spaces instead of at them.", ["evolve_g2/g2_match_pf_s1", "evolve_g2/g2_match_pf_s2"]),
 "g2_combo_02": ("Round 2: suffix memory + read-back (no copying)", "hnet", SMALL, 2, "These two did not add up without copying.", ["evolve_g2/g2_combo_02_s1", "evolve_g2/g2_combo_02_s2"]),
 # generation 3
 "g3_mem_02": ("Round 3: + fast word-completion memory", "hnet", SMALL, 3, "Copying + suffix memory + the trie, made fast. First tie with BPE + copying.", ["evolve_g3/g3_mem_02_s1", "evolve_g3/g3_mem_02_s2", "evolve_g4/g3_mem_02_s3"]),
 "g3_combo_01": ("Round 3: + read-back + LR fix", "hnet", SMALL, 3, "Copying + suffix memory + read-back + LR fix. Second tie.", ["evolve_g3/g3_combo_01_s1", "evolve_g3/g3_combo_01_s2", "evolve_g4/g3_combo_01_s3"]),
 "g3_route_02": ("Round 3: self-taught chunking (no hint)", "hnet", SMALL, 3, "A small predictor teaches the chunker where text is surprising. Finds word boundaries without a space hint.", ["evolve_g3/g3_route_02_s1", "evolve_g3/g3_route_02_s2"]),
 "g3_copy_01": ("Round 3: whole-chunk copying", "hnet", SMALL, 3, "Copy whole chunks, not bytes. No gain.", ["evolve_g3/g3_copy_01_s1", "evolve_g3/g3_copy_01_s2"]),
 "g3_copy_02": ("Round 3: longer chunks inside copies", "hnet", SMALL, 3, "Spend fewer chunks on copied text. No gain.", ["evolve_g3/g3_copy_02_s1", "evolve_g3/g3_copy_02_s2"]),
 "g3_route_01": ("Round 3: more chunks on hard text", "hnet", SMALL, 3, "Shift chunks toward unpredictable regions. No gain.", ["evolve_g3/g3_route_01_s1", "evolve_g3/g3_route_01_s2"]),
 # generation 4
 "g4_combo_02": ("Round 4: best H-Net", "hnet", SMALL, 4, "Everything that worked, combined: copying, suffix memory, fast trie, read-back, LR fix.", ["evolve_g4/g4_combo_02_s1", "evolve_g4/g4_combo_02_s2", "evolve_g4/g4_combo_02_s3"]),
 "g4_combo_03": ("Round 4: best H-Net, no space hint", "hnet", SMALL, 4, "The same with self-taught chunking: fully tokenizer-free.", ["evolve_g4/g4_combo_03_s1", "evolve_g4/g4_combo_03_s2"]),
 # generation 5
 "g5_compute_02": ("Round 5: extra thinking on surprising chunks", "hnet", SMALL, 5, "Re-run the deep layers once a chunk's first byte is known, for the 50% most surprising chunks.", ["evolve_g5/g5_compute_02_s1", "evolve_g5/g5_compute_02_s2"]),
 "g5_compute_01": ("Round 5: extra thinking on every chunk", "hnet", SMALL, 5, "The same on every chunk, paid for with one fewer deep layer.", ["evolve_g5/g5_compute_01_s1", "evolve_g5/g5_compute_01_s2"]),
 "g5_output_02": ("Round 5: bigger word-completion memory", "hnet", SMALL, 5, "Move parameters into the trie.", ["evolve_g5/g5_output_02_s1", "evolve_g5/g5_output_02_s2"]),
 "g5_output_01": ("Round 5: memory inside the deep layers", "hnet", SMALL, 5, "A large lookup memory read by the chunk layers.", ["evolve_g5/g5_output_01_s1", "evolve_g5/g5_output_01_s2"]),
 # big scale (L05 at 1e17 FLOPs)
 "big_bpe": ("BPE (standard tokenizer)", "bpe", BIG, 0, "Bigger model, 10x the compute.", ["ladder_1e17_L05/bpe_s0"]),
 "big_bpe_copy": ("BPE + copying", "bpe", BIG, 4, "The fair bar at the bigger size.", ["evolve_g4x/g2_bpe_match_L05_s1", "evolve_g4x/g2_bpe_match_L05_s2"]),
 "big_hnet": ("H-Net as published", "hnet", BIG, 0, "", ["ladder_1e17_L05/hnet_s0"]),
 "big_hnet_ws": ("H-Net + space hint", "hnet", BIG, 0, "", ["ladder_1e17_L05/hnet_ws_s0"]),
 "big_g4_combo_02": ("Round 4: best H-Net", "hnet", BIG, 4, "The small-scale winner, scaled up.", ["evolve_g4x/g4_combo_02_L05_s1", "evolve_g4x/g4_combo_02_L05_s2"]),
 "big_g4_combo_03": ("Round 4: best H-Net, no space hint", "hnet", BIG, 4, "", ["evolve_g4x/g4_combo_03_L05_s1"]),
}
LADDER = {"1e16": ["L03", "L04", "L05", "L06"], "1e17": ["L05", "L06", "L07", "L08", "L09"]}


def curve(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(f):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if "val/bpb" in x:
                out[int(x["step"])] = float(x["val/bpb"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json"))
    a = ap.parse_args()
    exps = []
    for eid, (label, fam, scale, gen, note, runs) in E.items():
        cs = [curve(os.path.join(a.live, r)) for r in runs]
        cs = [c for c in cs if c]
        assert cs, eid
        last = max(max(c) for c in cs)
        finals = [c[max(c)] for c in cs if max(c) == last]
        steps = sorted(set.intersection(*[set(c) for c in cs]))
        steps = [s for s in steps if s == last or s < last - 10]  # the eval just before the final one (900, 1970) duplicates it
        pts = [[s, round(statistics.mean(c[s] for c in cs), 4)] for s in steps]
        exps.append(dict(id=eid, label=label, family=fam, scale=scale, gen=gen, note=note, seeds=len(finals),
                         final=round(statistics.mean(finals), 4),
                         sd=round(statistics.stdev(finals), 4) if len(finals) > 1 else None,
                         per_seed=[round(v, 4) for v in finals], steps_total=last, curve=pts))
    ladder = {}
    for bud, rungs in LADDER.items():
        for L in rungs:
            for arm in ("bpe", "hnet", "hnet_ws"):
                c = curve(os.path.join(a.live, f"ladder_{bud}_{L}", f"{arm}_s0"))
                if c:
                    ladder.setdefault(bud, {}).setdefault(L, {})[arm] = round(c[max(c)], 4)
    json.dump(dict(experiments=exps, ladder=ladder), open(a.out, "w"), indent=1)
    for e in exps:
        print(f"{e['id']:18s} {e['scale']:5s} {e['final']:.3f} ± {e['sd'] or 0:.3f}  n={e['seeds']}  pts={len(e['curve'])}")
    print(json.dumps(ladder))


if __name__ == "__main__":
    main()
