# scripts/evaluate_calculator.py
import json, re, os
from tqdm import tqdm

def safe_eval(expr):
    # only allow digits, operators, parentheses, underscores, spaces and dots
    if not re.match(r"^[0-9\(\)\+\-\*/\s\._]+$", expr):
        raise ValueError("invalid characters in expr")
    # underscores should have been substituted before calling this
    return eval(expr, {"__builtins__": None}, {})

def evaluate_plan(plan_text):
    lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
    values = []
    logs = []
    for ln in lines:
        if ln == "<stop>" or re.match(r"^<stop>$", ln, flags=re.I):
            break
        m = re.match(r"^\d+\.\s*(.+)$", ln)
        if not m:
            logs.append({"line": ln, "error":"line_parse"})
            continue
        expr = m.group(1).split("#",1)[0].strip()
        # substitute placeholders
        def sub(m):
            idx = int(m.group(1)) - 1
            if idx < 0 or idx >= len(values):
                return "0"
            return str(values[idx])
        expr_sub = re.sub(r"_([0-9]+)", sub, expr)
        try:
            val = safe_eval(expr_sub)
            values.append(val)
            logs.append({"line":ln,"expr":expr_sub,"value":val})
        except Exception as e:
            logs.append({"line":ln,"expr":expr_sub,"error":str(e)})
            values.append(None)
    return values, logs

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("plans_file")
    p.add_argument("--out", default="data_intermediate/calculations.jsonl")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.plans_file, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            rec = json.loads(line)
            vals, logs = evaluate_plan(rec["plan"])
            outrec = {
                "id": rec["id"],
                "question": rec["question"],
                "gold": rec.get("answer"),
                "plan": rec["plan"],
                "values": vals,
                "logs": logs
            }
            fout.write(json.dumps(outrec, ensure_ascii=False)+"\n")
    print("done")
