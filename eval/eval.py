from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import pandas as pd

def validate_generated_script(
    script_path: str,
    expected_output_path: str,
    timeout: int = 60
) -> dict:

    script_path = Path(script_path)
    expected_output_path = Path(expected_output_path)

    result = {
        "script_exists": False,
        "syntax_valid": False,
        "execution_success": False,
        "xes_file_created": False,
        "overall_success": False,
        "error_type": None,
        "error_message": None,
        "stdout": None,
        "stderr": None,
    }

    if not script_path.exists():
        result["error_type"] = "FileNotFoundError"
        result["error_message"] = f"Script file does not exist: {script_path}"
        return result

    result["script_exists"] = True

    try:
        source_code = script_path.read_text(encoding="utf-8")
        compile(source_code, str(script_path), "exec")
        result["syntax_valid"] = True
    except SyntaxError as e:
        result["error_type"] = "SyntaxError"
        result["error_message"] = str(e)
        return result
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        return result

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(script_path.parent)
        )

        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr

        if completed.returncode == 0:
            result["execution_success"] = True
        else:
            result["error_type"] = "RuntimeError"
            result["error_message"] = completed.stderr
            return result

    except subprocess.TimeoutExpired as e:
        result["error_type"] = "TimeoutExpired"
        result["error_message"] = f"Script execution exceeded {timeout} seconds."
        result["stdout"] = e.stdout
        result["stderr"] = e.stderr
        return result

    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        return result

    if expected_output_path.exists():
        result["xes_file_created"] = True
    else:
        result["error_type"] = "OutputFileMissing"
        result["error_message"] = f"Expected XES file was not created: {expected_output_path}"
        return result
    
    result["overall_success"] = (
        result["script_exists"]
        and result["syntax_valid"]
        and result["execution_success"]
        and result["xes_file_created"]
    )

    return result


def validate_xes_file(xes_path: str) -> tuple[bool, list[str]]:
    issues = []
    try:
        tree = ET.parse(xes_path)
        root = tree.getroot()
    except ET.ParseError as e:
        return False, [f"XES parse error: {e}"]

    ns = {"xes": "http://www.xes-standard.org/"}
    tag = root.tag.lower()
    if "log" not in tag:
        issues.append(f"Root element is '{root.tag}', expected 'log'.")

    # Count traces and events
    traces = root.findall("trace") or root.findall("xes:trace", ns)
    if not traces:
        issues.append("No <trace> elements found in XES file.")
    else:
        event_count = 0
        for trace in traces:
            events = trace.findall("event") or trace.findall("xes:event", ns)
            event_count += len(events)
        if event_count == 0:
            issues.append("No <event> elements found across all traces.")

    return len(issues) == 0, issues


def load_xes_to_dataframe(xes_path: str) -> "pd.DataFrame":
    tree = ET.parse(xes_path)
    root = tree.getroot()

    def strip_ns(tag):
        return tag.split("}")[-1] if "}" in tag else tag

    rows = []
    for trace in root:
        if strip_ns(trace.tag) != "trace":
            continue
        case_id = None
        for attr in trace:
            if strip_ns(attr.tag) in ("string", "int", "float", "date", "boolean", "id"):
                if attr.attrib.get("key") == "concept:name":
                    case_id = attr.attrib.get("value")
            if strip_ns(attr.tag) == "event":
                break

        attrs_map = {}
        events_in_trace = []
        for child in trace:
            tag = strip_ns(child.tag)
            if tag in ("string", "int", "float", "date", "boolean", "id"):
                attrs_map[child.attrib.get("key")] = child.attrib.get("value")
            elif tag == "event":
                events_in_trace.append(child)

        case_id = attrs_map.get("concept:name", case_id)

        for event in events_in_trace:
            ev = {"case_id": case_id}
            for attr in event:
                tag = strip_ns(attr.tag)
                if tag in ("string", "int", "float", "date", "boolean", "id"):
                    ev[attr.attrib.get("key")] = attr.attrib.get("value")
            rows.append(ev)

    df = pd.DataFrame(rows)
    rename = {
        "concept:name": "activity",
        "time:timestamp": "timestamp",
        "org:resource": "resource",
    }
    df.rename(columns=rename, inplace=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        
    return df
