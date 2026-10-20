"""
Speech/evaluate_wer.py  (Owner: Salma Mohamed)

Measures Whisper Word Error Rate (WER) + latency on the audio test set, and the
wake-word activation / false-activation rates. Writes docs/speech_results.md.

Audio test set layout (made with Yousef Waleed, see speech/record_test_set.py):
  data/audio_test/transcripts.csv   columns: file,speaker,condition,style,transcript
  data/audio_test/wake/*.wav        clips that SAY "hey sound ..."  (should activate)
  data/audio_test/other/*.wav       unrelated speech               (should NOT activate)

Run:  python -m speech.evaluate_wer --models tiny base small
"""
import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

WER_TARGET = 0.15


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(ref: list, hyp: list) -> int:
    """Word-level Levenshtein distance (substitutions + insertions + deletions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def corpus_wer(pairs) -> float:
    """pairs = [(reference, hypothesis)]; total errors / total reference words."""
    errors = words = 0
    for ref, hyp in pairs:
        r, h = normalize_text(ref).split(), normalize_text(hyp).split()
        errors += edit_distance(r, h)
        words += len(r)
    return errors / words if words else 0.0


def load_rows(audio_dir: Path) -> list:
    csv_path = audio_dir / "transcripts.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Transcripts file not found at: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Transcripts file is empty: {csv_path}")
        return rows


def evaluate_model(model_size: str, audio_dir: Path) -> dict:
    from speech.transcriber import transcribe
    rows = load_rows(audio_dir)
    results = []
    
    first_file = audio_dir / rows[0]["file"]
    if first_file.exists():
        transcribe(str(first_file), model_size=model_size)   # warm-up
        
    for row in rows:
        file_path = audio_dir / row["file"]
        if not file_path.exists():
            print(f"Warning: Audio file not found: {file_path}")
            continue
            
        t0 = time.perf_counter()
        hyp = transcribe(str(file_path), model_size=model_size)
        results.append({
            **row, 
            "hyp": hyp, 
            "wer": wer(row["transcript"], hyp),
            "latency": time.perf_counter() - t0
        })
    return {"model": model_size, "results": results}


def group_wer(results: list, key: str) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[r[key]].append((r["transcript"], r["hyp"]))
    return {k: corpus_wer(v) for k, v in sorted(groups.items())}


def evaluate_wake(audio_dir: Path) -> dict:
    from speech.transcriber import transcribe_wake
    from speech.wake_word import is_wake_phrase
    
    wake_dir = audio_dir / "wake"
    other_dir = audio_dir / "other"
    
    wake = sorted(wake_dir.glob("*.wav")) if wake_dir.exists() else []
    other = sorted(other_dir.glob("*.wav")) if other_dir.exists() else []
    
    hits = sum(is_wake_phrase(transcribe_wake(str(p))) for p in wake)
    false = sum(is_wake_phrase(transcribe_wake(str(p))) for p in other)
    
    return {
        "wake_total": len(wake), 
        "wake_hits": hits,
        "other_total": len(other), 
        "false_activations": false
    }


def write_report(evals: list, wake: dict, out: Path) -> None:
    lines = [
        "# Speech Results (Whisper WER + Wake Word)", "",
        f"Target: WER <= {WER_TARGET:.0%} on clean speech.", "",
        "## Model comparison", "",
        "| Model | Overall WER | Clean WER | Noisy WER | Avg latency (s) | Meets target |",
        "|---|---|---|---|---|---|"
    ]
    
    for e in evals:
        res = e["results"]
        if not res:
            continue
        by_cond = group_wer(res, "condition")
        overall = corpus_wer([(r["transcript"], r["hyp"]) for r in res])
        lat = sum(r["latency"] for r in res) / len(res)
        
        clean = by_cond.get("clean", float("nan"))
        noisy = by_cond.get("noisy", float("nan"))
        
        # Protection against NaN formatting issues
        clean_str = f"{clean:.1%}" if (isinstance(clean, (int, float)) and clean == clean) else "N/A"
        noisy_str = f"{noisy:.1%}" if (isinstance(noisy, (int, float)) and noisy == noisy) else "N/A"
        
        meets_target = "yes" if (isinstance(clean, (int, float)) and clean == clean and clean <= WER_TARGET) else "no"
        
        lines.append(f"| {e['model']} | {overall:.1%} | {clean_str} | "
                     f"{noisy_str} | {lat:.2f} | {meets_target} |")
        
    best = evals[-1] if evals else None
    if best and best["results"]:
        for key, title in (("speaker", "Per speaker"), ("style", "Per command style")):
            lines += ["", f"## {title} (model: {best['model']})", "", "| Group | WER |", "|---|---|"]
            lines += [f"| {k} | {v:.1%} |" for k, v in group_wer(best["results"], key).items()]
            
        worst = sorted(best["results"], key=lambda r: r["wer"], reverse=True)[:10]
        lines += ["", f"## 10 worst clips (model: {best['model']})", "",
                  "| File | Expected | Heard | WER |", "|---|---|---|---|"]
        lines += [f"| {r['file']} | {r['transcript']} | {r['hyp']} | {r['wer']:.0%} |" for r in worst]
        
    if wake and wake["wake_total"] > 0:
        act = wake["wake_hits"] / wake["wake_total"]
        lines += ["", "## Wake word", "",
                  f"- Activation: {wake['wake_hits']}/{wake['wake_total']} = {act:.0%} (target >= 90%)",
                  f"- False activations on unrelated speech: "
                  f"{wake['false_activations']}/{wake['other_total']} clips"]
    elif wake:
        lines += ["", "## Wake word", "", "- No wake/other audio files found for evaluation."]
        
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tiny.en", "base.en", "small.en"])
    ap.add_argument("--audio-dir", default="data/audio_test")
    ap.add_argument("--out", default="docs/speech_results.md")
    ap.add_argument("--no-wake", action="store_true")
    args = ap.parse_args()
    
    audio_dir = Path(args.audio_dir)
    evals = [evaluate_model(m, audio_dir) for m in args.models]
    wake = None if args.no_wake else evaluate_wake(audio_dir)
    write_report(evals, wake, Path(args.out))
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()