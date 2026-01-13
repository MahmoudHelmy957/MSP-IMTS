import os
import re
import ast
from typing import Dict, List, Optional
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RE_NAMESPACE = re.compile(r"args=Namespace\((.*)\)")
RE_COMMAND = re.compile(r"command:\s*(.*)")

def _parse_namespace_line(line: str) -> Optional[Dict]:
    m = RE_NAMESPACE.search(line)
    if not m:
        return None

    ns_str = m.group(1).strip()
    # Convert "a=1, b='x'" into a valid Python dict: {'a':1, 'b':'x'}
    # Note: values like device(...) remain as strings after literal_eval fails;
    # we'll handle by quoting unknown tokens safely.
    # First, turn keys into 'key':
    dict_like = re.sub(r"(\w+)=", r"'\1':", ns_str)

    # Wrap with braces
    dict_like = "{" + dict_like + "}"

    try:
        return ast.literal_eval(dict_like)
    except Exception:
        # fallback: quote anything that looks like bareword call e.g. device(...)
        # and any bareword identifiers (cuda, cpu, etc.)
        safe = dict_like

        # Quote patterns like device(type='cuda') or Namespace(...) if appear as values
        safe = re.sub(r":\s*([A-Za-z_]\w*\([^}]*?\))", r": '\1'", safe)

        # Quote barewords that are not True/False/None and not numbers
        def repl_bareword(m):
            word = m.group(1)
            if word in {"True", "False", "None"}:
                return f": {word}"
            return f": '{word}'"

        safe = re.sub(r":\s*([A-Za-z_]\w*)\s*(,|\})", lambda m: repl_bareword(m) + m.group(2), safe)

        return ast.literal_eval(safe)

def _parse_command_line(line: str) -> Optional[Dict[str, str]]:
    m = RE_COMMAND.search(line)
    if not m:
        return None
    cmd = m.group(1).strip()

    # naive parse: --key value (stores strings)
    parts = cmd.split()
    d = {}
    i = 0
    while i < len(parts):
        if parts[i].startswith("--"):
            key = parts[i][2:]
            val = True
            if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                val = parts[i + 1]
                i += 1
            d[key] = val
        i += 1
    return d

def extract_hparams(log_path: str) -> Dict:
    """
    Prefer args=Namespace(...) if present.
    Else fallback to command: ... parsing.
    """
    namespace_dict = None
    command_dict = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if namespace_dict is None:
                namespace_dict = _parse_namespace_line(line)
            if command_dict is None:
                command_dict = _parse_command_line(line)
            if namespace_dict is not None and command_dict is not None:
                break

    if namespace_dict is not None:
        # drop noisy runtime keys if you want (optional)
        drop_keys = {"PID", "device"}  # add more if needed
        for k in list(namespace_dict.keys()):
            if k in drop_keys:
                namespace_dict.pop(k, None)
        return namespace_dict

    if command_dict is not None:
        return command_dict

    raise ValueError(f"No hyperparameter block found in: {log_path}")

def compare_logs(log_paths: List[str], only_diff: bool = True) -> pd.DataFrame:
    records = {}
    for p in log_paths:
        name = os.path.basename(p)
        records[name] = extract_hparams(p)

    df = pd.DataFrame.from_dict(records, orient="columns")

    # normalize: make everything string for stable comparison (optional)
    df = df.applymap(lambda x: str(x))

    if only_diff:
        df = df.loc[df.nunique(axis=1) > 1]

    return df.sort_index()

if __name__ == "__main__":
    # Example usage:
    logs = [
    os.path.join(BASE_DIR, "analyzelogs/activity_MH_ACITIVTY_MS_job304607.run.log")]


    df = compare_logs(logs, only_diff=True)
    print("\nDIFFERING HYPERPARAMETERS:\n")
    print(df.to_string())
