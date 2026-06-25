
from memory.main import Memory
from generation.prompt_construction import f_prompt
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from eval.eval import validate_xes_file, load_xes_to_dataframe
from eval.variants import create_variants_report, remove_traces_with_missing_values
from eval.time_e import evaluate_throughput_time 

async def generation_loop(
    mem: Memory,
    round_messages,
    current_session_id,
    general_llm,
    f_prompt_instance: f_prompt,
    process_model: str = "",
    config: str = "",
    special_requirements: str = "",
    output_base_dir: str = "",
):
    """
    Generation loop: prompt → LLM script → save → execute → validate → report.
    Runs until a valid XES is produced or user aborts.
    """
    output_xes = False

    while not output_xes:
        # ── 1. Construct prompt ──────────────────────────────────────────────
        print("Constructing generation prompt...")
        prompt = f_prompt_instance.construct(
            process_model=process_model,
            config=config,
            special_requirements=special_requirements,
            llm=general_llm
        )
        # ── 2. Call LLM to generate script ───────────────────────────────────
        print("Calling LLM to generate event log script...")
        messages = [{"role": "user", "content": prompt}]
        raw_script = general_llm.chat(messages)
        if raw_script.startswith("```"):
            raw_script = raw_script.split("\n", 1)[-1]
        if raw_script.endswith("```"):
            raw_script = raw_script.rsplit("```", 1)[0]
        raw_script = raw_script.strip()

        # ── 3. Save script to timestamped folder ─────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_base_dir) / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        script_path = run_dir / "generate_log.py"
        script_path.write_text(raw_script, encoding="utf-8")
        print(f"Script saved to: {script_path}")

        # ── 4. Syntax-check the script (compile without running) ─────────────
        print("Checking script syntax...")
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script_path)],
            capture_output=True, text=True
        )
        if compile_result.returncode != 0:
            error_msg = compile_result.stderr.strip()
            print(f"Syntax error in generated script:\n{error_msg}")
            round_messages.append({
                "role": "assistant",
                "content": f"Generated script has syntax errors:\n{error_msg}\nRetrying..."
            })
            mem.add_session_history(messages=round_messages, session_id=current_session_id)
            round_messages.clear()
            continue

        # ── 5. Execute the script ─────────────────────────────────────────────
        print("Executing generated script...")
        exec_result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(run_dir),
            timeout=120,
        )

        if exec_result.returncode != 0:
            error_msg = exec_result.stderr.strip()
            print(f"Script execution failed:\n{error_msg}")
            round_messages.append({
                "role": "assistant",
                "content": f"Script execution failed:\n{error_msg}\nRetrying..."
            })
            mem.add_session_history(messages=round_messages, session_id=current_session_id)
            round_messages.clear()
            continue

        # ── 6. Locate XES file ────────────────────────────────────────────────
        xes_path = run_dir / "event_log.xes"
        if not xes_path.exists():
            print("Script ran but no event_log.xes was produced. Retrying...")
            round_messages.append({
                "role": "assistant",
                "content": "Script ran successfully but produced no event_log.xes. Retrying..."
            })
            mem.add_session_history(messages=round_messages, session_id=current_session_id)
            round_messages.clear()
            continue

        # ── 7. Validate XES format ────────────────────────────────────────────
        print("Validating XES format...")
        is_valid, xes_issues = validate_xes_file(str(xes_path))
        if not is_valid:
            print(f"XES validation failed: {xes_issues}")
            round_messages.append({
                "role": "assistant",
                "content": f"XES file is malformed: {xes_issues}\nRetrying..."
            })
            mem.add_session_history(messages=round_messages, session_id=current_session_id)
            round_messages.clear()
            continue

        print("XES format is valid.")

        # ── 8. Load XES → DataFrame ───────────────────────────────────────────
        print("Loading XES into DataFrame...")
        log_df = load_xes_to_dataframe(str(xes_path))

        print("Cleaning event log...")
        log_df, clean_report = remove_traces_with_missing_values(
            log_df,
            required_columns=["case_id", "activity", "timestamp"],
            treat_empty_string_as_missing=True
        )

        print(
            f"Cleaning done: removed {clean_report['removed_trace_count']} traces "
            f"({clean_report['original_trace_count']} → {clean_report['remaining_trace_count']} traces, "
            f"{clean_report['original_event_count']} → {clean_report['remaining_event_count']} events)."
        )
        if log_df.empty:
            print("Event log is empty after cleaning. Retrying...")
            round_messages.append({
                "role": "assistant",
                "content": (
                    f"Event log is empty after removing traces with missing values. "
                    f"Clean report: {clean_report}\nRetrying..."
                )
            })
            mem.add_session_history(messages=round_messages, session_id=current_session_id)
            round_messages.clear()
            continue

        # ── 9. Generate reports ───────────────────────────────────────────────
        print("Generating throughput time report...")
        time_report = evaluate_throughput_time(log_df)

        print("Generating variant report...")
        variant_report = create_variants_report(log_df)

        # ── 10. Save reports as text ──────────────────────────────────────────
        report_path = run_dir / "report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Throughput Time Report ===\n")
            for k, v in time_report.items():
                if hasattr(v, "to_string"):
                    f.write(f"\n-- {k} --\n{v.to_string()}\n")
                elif isinstance(v, dict):
                    f.write(f"\n-- {k} --\n")
                    for kk, vv in v.items():
                        f.write(f"  {kk}: {vv}\n")
                else:
                    f.write(f"{k}: {v}\n")

            histogram_df = variant_report.get("histogram df")
            if histogram_df is not None and not histogram_df.empty:
                f.write("\n-- Histogram: Case Count Distribution per Variant --\n")
                f.write(f"  Total variants: {len(histogram_df)}\n")
                desc = histogram_df["count"].describe()
                f.write(f"  count   {desc['count']:>10.0f}\n")
                f.write(f"  mean    {desc['mean']:>10.2f}\n")
                f.write(f"  std     {desc['std']:>10.2f}\n")
                f.write(f"  min     {desc['min']:>10.0f}\n")
                f.write(f"  25%     {desc['25%']:>10.2f}\n")
                f.write(f"  50%     {desc['50%']:>10.2f}\n")
                f.write(f"  75%     {desc['75%']:>10.2f}\n")
                f.write(f"  max     {desc['max']:>10.0f}\n")
                f.write("\n  Top 20 variants (code | count | %):\n")
                top20 = histogram_df.head(20)
                for _, row in top20.iterrows():
                    label = str(row.get("mapped_trace_string", ""))[:30]
                    f.write(
                        f"    {label:30s}  count={int(row['count']):5d}  "
                        f"{row['percentage']:5.1f}%\n"
                    )

            # --- boxplot df ---
            boxplot_df = variant_report.get("boxplot df")
            if boxplot_df is not None and not boxplot_df.empty:
                f.write("\n-- Boxplot: Trace Length Distribution by Variant --\n")
                variant_stats = (
                    boxplot_df.groupby("variant_string")["trace_length"]
                    .describe()
                    .sort_values("50%", ascending=False)
                    .reset_index()
                )
                f.write(
                    f"  {'Variant':<45} {'count':>6} {'min':>5} "
                    f"{'25%':>6} {'median':>7} {'75%':>6} {'max':>5}\n"
                )
                f.write("  " + "-" * 85 + "\n")
                for _, row in variant_stats.iterrows():
                    label = str(row["variant_string"])[:42]
                    if len(str(row["variant_string"])) > 42:
                        label += "..."
                    f.write(
                        f"  {label:<45} {int(row['count']):>6} {int(row['min']):>5} "
                        f"{row['25%']:>6.1f} {row['50%']:>7.1f} {row['75%']:>6.1f} {int(row['max']):>5}\n"
                    )
        
        print(f"Reports saved to: {report_path}")
        prompt_path = run_dir / "prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        # ── 11. Summarise to memory & finish ──────────────────────────────────
        summary = (
            f"Event log generated successfully at {run_dir}.\n"
            f"Traces: {time_report['attributes'].get('num_cases_with_duration', 'N/A')}, "
            f"Variants: {variant_report.get('variants number', 'N/A')}."
        )
        round_messages.append({"role": "assistant", "content": summary})
        mem.add_session_history(messages=round_messages, session_id=current_session_id)
        round_messages.clear()
        print(summary)

        output_xes = True

    return {
        "run_dir": str(run_dir),
        "xes_path": str(xes_path),
        "time_report": time_report,
        "variant_report": variant_report,
    }