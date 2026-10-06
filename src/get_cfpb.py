import json, csv, os

with open("data/complaints.json", encoding="utf-8") as f:
    d = json.load(f)

if isinstance(d, dict) and "detail" in d:
    raise SystemExit(f"API said: {d}")
if isinstance(d, list):
    rows = d
elif isinstance(d, dict) and "hits" in d:
    h = d["hits"]
    rows = h["hits"] if isinstance(h, dict) and "hits" in h else h
else:
    raise SystemExit(f"Unexpected shape, top keys: {list(d)[:10]}")

rows = [r.get("_source", r) for r in rows]

out = os.path.join("data", "complaints.csv")
n = 0
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Consumer complaint narrative", "Product", "Issue"])
    for r in rows:
        text = (r.get("complaint_what_happened") or "").strip()
        if text:
            w.writerow([text, r.get("product", ""), r.get("issue", "")])
            n += 1
print(f"wrote {n} rows to {out}")