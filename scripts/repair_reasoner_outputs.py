#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

import re
import ast
import math
from typing import Optional, List, Tuple

STRICT_NUMERIC_ONLY = False  # set by CLI

# import the same helpers or copy-paste repair_calc_markers_in_text, verify_calc_lines, safe_calc here
# For brevity, assume we paste those three functions here exactly as in A)

# Identify names inside expressions and decide if expression is symbolic (has variables/functions)
_IDENT_RE = re.compile(r"[A-Za-z_\\]+")  # includes LaTeX commands like \sin
_ALLOWED_IDENTIFIERS = {"sqrt", "pi", "e"}  # purely numeric-safe names


def _extract_identifiers(expr: str) -> List[str]:
    # remove allowed sqrt( by normalizer; we still allow 'sqrt' token
    return _IDENT_RE.findall(expr)

def _is_symbolic_expr(expr: str) -> bool:
    """
    True if expr contains identifiers other than allowed numeric-safe ones (sqrt, pi, e).
    This catches sin, cos, x, n, etc., and LaTeX commands (sin, cos, ...).
    """
    if not expr:
        return False
    ids = set(x.strip("\\") for x in _extract_identifiers(expr))
    # if any identifier besides allowed shows up, treat as symbolic
    return any((tok not in _ALLOWED_IDENTIFIERS) for tok in ids)


# ----------------------------
# Calculator (safe) - UPDATED
# ----------------------------

# --- numeric normalization helpers (safe to keep local to this block) ---
_UNIT_PATTERNS = [
    r"\^\s*\\?circ",   # ^\circ (LaTeX)
    r"°",              # degree symbol
    r"\\,|\\!|\\;|\\:",# LaTeX spacing
]
_SYMBOL_MAP = {
    "×": "*", "·": "*", "⋅": "*", "\\cdot": "*",
    "−": "-", "—": "-",
    "÷": "/",
}
def _latex_frac_to_ascii(s: str) -> str:
    # \frac{a}{b} -> (a)/(b)
    return re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)

def _strip_units_and_symbols(s: str) -> str:
    t = s
    for k, v in _SYMBOL_MAP.items():
        t = t.replace(k, v)
    for pat in _UNIT_PATTERNS:
        t = re.sub(pat, "", t)
    t = t.replace("^", "**")  # caret power -> python power
    return re.sub(r"\s+", " ", t).strip()

def normalize_numeric_text(s: str) -> str:
    return _strip_units_and_symbols(_latex_frac_to_ascii(s or ""))

# --- allow a tiny set of math functions/names if needed ---
ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
}
ALLOWED_NAMES = {"pi": math.pi, "e": math.e}

class SafeEval(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)

        # Python 3.8+ uses Constant for numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("bad const")

        if isinstance(node, ast.BinOp):
            l = self.visit(node.left)
            r = self.visit(node.right)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div): return l / r
            if isinstance(node.op, ast.Pow): return l ** r
            if isinstance(node.op, ast.Mod): return l % r
            raise ValueError("bad op")

        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.UAdd): return +v
            if isinstance(node.op, ast.USub): return -v
            raise ValueError("bad uop")

        if isinstance(node, ast.Call):
            # Only allow simple name calls like sqrt(9)
            if not isinstance(node.func, ast.Name):
                raise ValueError("bad call")
            fn = node.func.id
            if fn not in ALLOWED_FUNCS:
                raise ValueError("func not allowed")
            args = [self.visit(a) for a in node.args]
            return ALLOWED_FUNCS[fn](*args)

        if isinstance(node, ast.Name):
            if node.id in ALLOWED_NAMES:
                return ALLOWED_NAMES[node.id]
            raise ValueError("name not allowed")

        # Disallow everything else (attributes, subscripts, lambdas, etc.)
        raise ValueError("bad expr")

def safe_calc(expr: str) -> Optional[float]:
    expr = (expr or "").strip()
    try:
        tree = ast.parse(expr, mode="eval")
        return float(SafeEval().visit(tree))
    except Exception:
        return None
    
# --- verification of [[calc: ...]] -> ... lines ---
_CALC_PATTERN = re.compile(r"\[\[calc:\s*(.*?)\s*\]\]\s*->\s*([^\n]+)")
_SYMBOLIC_TOKENS_RE = re.compile(r"[A-Za-z√π]")  # treat letters/√/π as symbolic

def _is_symbolic(s: str) -> bool:
    return bool(_SYMBOLIC_TOKENS_RE.search(s or ""))

def _safe_num(s: str) -> Optional[float]:
    """
    Parse a RHS number; supports plain ints/floats and simple rationals like '3/2'.
    Returns float or None if not a plain numeric.
    """
    try:
        t = s.replace(",", "").strip()
        # simple rational
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?\s*/\s*[+-]?\d+(?:\.\d+)?", t):
            num, den = re.split(r"/", t)
            return float(num) / float(den)
        return float(t)
    except Exception:
        return None

def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s

def verify_calc_lines(text: str) -> Tuple[bool, List[str], str]:
    """
    Look for '[[calc: ...]] -> value' lines and verify.
    Returns (all_ok, messages, possibly_patched_text)
    """
    msgs = []
    patched = []
    ok = True
    for line in text.splitlines():
        m = re.search(r"\[\[calc:\s*(.*?)\s*\]\]\s*->\s*([+-]?\d+(?:\.\d+)?)", line)
        if not m:
            patched.append(line)
            continue
        expr = m.group(1)
        claimed = float(m.group(2))
        got = safe_calc(expr)
        if got is None:
            ok = False
            msgs.append(f"calc parse failed: {expr}")
            patched.append(line)
        else:
            if abs(got - claimed) > 1e-6:
                ok = False
                msgs.append(f"mismatch: {expr} = {got} != {claimed}")
                new_line = re.sub(r"->\s*[+-]?\d+(\.\d+)?", f"-> {got}", line)
                patched.append(new_line)
            else:
                patched.append(line)
    return ok, msgs, "\n".join(patched)

# ----------------------------
# Calc marker repair helpers
# ----------------------------

# Reuse your safe_calc (already imported/defined)
# from your earlier block, we assume safe_calc(expr) -> Optional[float]

_UNIT_RE = re.compile(r"\b(?:kg|g|lbs?|pounds?|mile?s?|km|m|cm|mm|ft|feet|inch(?:es)?|hrs?|hours?|mins?|minutes?|days?|years?|deg(?:rees)?|°|percent)\b", re.I)
_CURRENCY_RE = re.compile(r"[$£€¥]")
_NUMBER_WITH_COMMAS_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+)(?!\d)")
_SQRT_UNICODE_RE = re.compile(r"√\s*(\d+(?:\.\d+)?)")
_FRAC_LATEX_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_SUP_LATEX_RE = re.compile(r"\^\{([^{}]+)\}")        # ^{2} -> **2
_CDOT_RE = re.compile(r"(\\cdot|\\times|·|×)")
_LATEX_BRACES_RE = re.compile(r"(\\left|\\right)")
_SUP_PLAIN_RE = re.compile(r"(?<=\d)\s*\^\s*(\d+)")  # 3^2 -> 3**2

CALC_LINE_RE = re.compile(r"\[\[calc:\s*(.*?)\s*\]\]\s*->\s*([+-]?\d+(?:\.\d+)?)")
BLANK_OR_WS_RE = re.compile(r"^\s*$")
NUM_ONLY_RE = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?\s*$")


def _normalize_inline_math(s: str) -> str:
    # Remove currency
    s = _CURRENCY_RE.sub("", s)
    # Remove units
    s = _UNIT_RE.sub("", s)
    # Remove thousands commas
    s = _NUMBER_WITH_COMMAS_RE.sub(lambda m: m.group(1).replace(",", ""), s)
    # LaTeX \frac{a}{b} -> (a)/(b)
    s = _FRAC_LATEX_RE.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    # LaTeX/Unicode operators
    s = _CDOT_RE.sub("*", s)
    s = _LATEX_BRACES_RE.sub("", s)
    # Unicode sqrt -> sqrt()
    s = _SQRT_UNICODE_RE.sub(lambda m: f"sqrt({m.group(1)})", s)
    # LaTeX \sqrt{...} -> sqrt(...)
    s = s.replace("\\sqrt", "sqrt")
    s = s.replace("{", "(").replace("}", ")")
    # Exponent: ^{k} or ^k -> **k
    s = _SUP_LATEX_RE.sub(lambda m: f"**({m.group(1)})", s)
    s = _SUP_PLAIN_RE.sub(lambda m: f"**{m.group(1)}", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _balance_parens(expr: str) -> str:
    opens = expr.count("(")
    closes = expr.count(")")
    if opens > closes:
        expr += ")" * (opens - closes)
    elif closes > opens:
        # rare: drop extras from end if they’re dangling
        while closes > opens and expr.endswith(")"):
            expr = expr[:-1]
            closes -= 1
    return expr

def _clean_calc_expr(expr: str) -> str:
    if not expr:
        return expr
    expr = expr.strip()
    # normalize operators and inline math
    expr = _normalize_inline_math(expr)
    # Some stray tokens sometimes sneak in, strip brackets
    expr = expr.replace("]","").replace("[","")
    # common malformed chunks like '6)**2' -> '(6)**2' best-effort
    expr = re.sub(r"(^|\s)(\d+)\)\s*\*\*", r"\1(\2)**", expr)
    # ensure balanced parens
    expr = _balance_parens(expr)
    return expr

def _numbers_in_line(s: str) -> List[float]:
    # extract floats/ints (ignore inside calc markers)
    s_no_calc = re.sub(r"\[\[calc:.*?\]\]", "", s)
    return [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", s_no_calc)]

def _try_infer_binary_expr(nums: List[float], target: float, tol: float = 1e-6) -> Optional[str]:
    """
    Try to infer a simple binary expression a ∘ b (∘ in {+,-,*,/}) from numbers
    on the same line that evaluates to `target`. Returns the expression string
    (e.g., "600+300") or None.
    """
    if not nums:
        return None

    # Search only the last few numbers to keep it cheap & relevant
    window = nums[-6:] if len(nums) > 6 else nums[:]
    n = len(window)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = window[i], window[j]
            candidates = [
                (f"{a}+{b}", a + b),
                (f"{a}-{b}", a - b),
                (f"{b}-{a}", b - a),
                (f"{a}*{b}", a * b),
            ]
            if abs(b) > tol:
                candidates.append((f"{a}/{b}", a / b))
            if abs(a) > tol:
                candidates.append((f"{b}/{a}", b / a))

            for expr, val in candidates:
                if math.isfinite(val) and abs(val - target) <= tol:
                    return expr
    return None

def repair_calc_markers_in_text(text: str) -> Tuple[str, List[str]]:
    """
    Repairs [[calc: ...]] lines by:
      - Removing symbolic calc markers (contain variables/functions beyond sqrt, pi, e).
      - Optionally (STRICT_NUMERIC_ONLY) disabling inference when the *line* is symbolic.
      - Filling blank expressions when a plausible (a op b) can be inferred.
      - Replacing numeric-only expressions with inferred binary when possible.
      - Cleaning malformed expressions and re-validating.
      - If expression still doesn't compute, drop wrapper (or entire marker) depending on context.
    Returns (patched_text, list_of_notes)
    """
    notes = []
    out_lines = []

    def _repair_one(match: re.Match) -> str:
        raw_expr, val = match.group(1), match.group(2)
        expr = (raw_expr or "").strip()
        line_context = match.string
        symbolic_ctx = _is_symbolic_expr(line_context)  # check the whole line, not just expr

        # Parse arrow value; if non-numeric, drop marker entirely
        try:
            target = float(val)
        except Exception:
            notes.append("non-numeric arrow; removed calc wrapper")
            return ""

        cleaned = _clean_calc_expr(expr)

        # If the *expression* itself is symbolic, drop the whole marker
        if _is_symbolic_expr(cleaned):
            notes.append(f"removed symbolic calc marker: {expr}")
            return ""

        # If line is symbolic AND strict mode is on, do not infer/fill—just drop calc wrappers
        if STRICT_NUMERIC_ONLY and symbolic_ctx:
            if cleaned and safe_calc(cleaned) is not None:
                # keep a valid purely numeric calc if already present
                return f"[[calc: {cleaned}]] -> {val}"
            # otherwise drop the marker (avoid weird inferences like 2-1)
            notes.append("strict mode: dropped calc in symbolic context")
            return ""

        # Blank expression: try infer; else keep arrow only
        if BLANK_OR_WS_RE.match(cleaned or ""):
            nums = _numbers_in_line(line_context)
            inferred = _try_infer_binary_expr(nums, target)
            if inferred:
                fx = _clean_calc_expr(inferred)
                if safe_calc(fx) is not None:
                    notes.append(f"filled blank calc -> {fx} = {val}")
                    return f"[[calc: {fx}]] -> {val}"
            notes.append("dropped blank calc (kept arrow value)")
            return f"-> {val}"

        # Numeric-only expression: try to infer nicer binary; else keep if valid
        if NUM_ONLY_RE.match(cleaned):
            nums = _numbers_in_line(line_context)
            inferred = _try_infer_binary_expr(nums, target)
            if inferred:
                fx = _clean_calc_expr(inferred)
                if safe_calc(fx) is not None:
                    notes.append(f"replaced numeric-only calc {cleaned} -> {fx}")
                    return f"[[calc: {fx}]] -> {val}"
            if safe_calc(cleaned) is not None:
                return f"[[calc: {cleaned}]] -> {val}"
            notes.append(f"dropped non-evaluable numeric-only calc {cleaned}")
            return f"-> {val}"

        # Some expression present: validate/fix arrow; if not evaluable, attempt inference
        got = safe_calc(cleaned)
        if got is None:
            nums = _numbers_in_line(line_context)
            inferred = _try_infer_binary_expr(nums, target)
            if inferred:
                fx = _clean_calc_expr(inferred)
                if safe_calc(fx) is not None:
                    notes.append(f"repaired malformed calc '{expr}' -> {fx}")
                    return f"[[calc: {fx}]] -> {val}"
            notes.append(f"left malformed calc unchanged: {expr}")
            return f"[[calc: {expr}]] -> {val}"
        else:
            if abs(got - target) > 1e-6:
                notes.append(f"fixed arrow: {cleaned} = {got} (was {val})")
                return f"[[calc: {cleaned}]] -> {got}"
            return f"[[calc: {cleaned}]] -> {val}"

    for raw_line in text.splitlines():
        new_line = CALC_LINE_RE.sub(_repair_one, raw_line)
        out_lines.append(new_line)

    return "\n".join(out_lines), notes


def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                # skip or warn
                continue

def main():
    ap = argparse.ArgumentParser(description="Repair [[calc: ...]] markers in reasoner outputs.")
    ap.add_argument("--in", dest="infile", help="single file to repair")
    ap.add_argument("--out", dest="outfile", help="output file (default: overwrite input with .repaired suffix)")
    ap.add_argument("--root", help="directory root to glob files under")
    ap.add_argument("--glob", default="*_reasoner.jsonl*", help="glob pattern under --root")
    ap.add_argument("--inplace", action="store_true", help="overwrite files in place (keep .bak)")
    ap.add_argument("--strict_numeric_only", action="store_true", help="Do not infer/fill [[calc: ...]] when the surrounding line contains symbols (identifiers beyond sqrt, pi, e). Drop such calc markers instead.")
    args = ap.parse_args()

    global STRICT_NUMERIC_ONLY
    STRICT_NUMERIC_ONLY = args.strict_numeric_only

    files = []
    if args.infile:
        files = [Path(args.infile)]
    elif args.root:
        files = list(Path(args.root).rglob(args.glob))
        print(f"Found {len(files)} files matching '{args.glob}' under '{args.root}'")
    else:
        print("Provide --in FILE or --root DIR [--glob PATTERN]", file=sys.stderr)
        sys.exit(2)

    for fp in files:
        outp = Path(args.outfile) if args.outfile else (fp.with_suffix(fp.suffix + ".repaired"))
        if args.inplace:
            bak = fp.with_suffix(fp.suffix + ".bak")
            if not bak.exists():
                fp.replace(bak)
            src = bak
            outp = fp
        else:
            src = fp

        count = 0
        fixed = 0
        with src.open("r", encoding="utf-8") as fin, outp.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                count += 1
                txt = ex.get("reasoner_output", "")
                patched, notes = repair_calc_markers_in_text(txt)
                ok_after, msgs_after, patched2 = verify_calc_lines(patched)
                if patched2 != txt:
                    fixed += 1
                ex["reasoner_output"] = patched2
                # optionally capture notes in qc
                qc = ex.get("qc", {"passed": True, "reasons": []})
                qc_reasons = list(qc.get("reasons", []))
                qc_reasons.extend([f"[repair] {n}" for n in notes])
                qc["reasons"] = qc_reasons
                ex["qc"] = qc
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"[OK] {fp} -> {outp}  ({fixed}/{count} examples changed)")

if __name__ == "__main__":
    main()
