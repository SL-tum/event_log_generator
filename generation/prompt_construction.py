from tools import WatsonxLLM
class f_prompt:
    def __init__(
            self
    ) -> None:
        self.prompt: str = ""
        self.prompt_template = """
**Role:** You are an expert Process Mining Data Engineer.
**Task:** Generate a Python script that creates a synthetic event log in XES format based on the specific BPMN process model and the provided configuration.
"""
        self.requirement_template = """
### 3. Structural Requirements
The script must programmatically ensure the generated traces follow these **BPMN control-flow patterns**:
* **Parallelism (AND-split/join):** Include two tasks that execute concurrently with shuffled timestamps.
* **Optionality:** Include a conditional branch where certain steps occur only in specific cases.
* **Looping:** Include a "rework" loop where a task may repeat based on a probability (e.g., a quality check failure).
* **Student Discovery:** The patterns should be clear enough that a student can discover them using process discovery algorithms (like Alpha Miner or Inductive Miner).

### 4. Noise & Data Quality Requirements
To make the exercise realistic for students, the script must inject the following **data quality errors** into the log:
* **Controlled Typos: **DO NOT create dozens of variations. Select 1 specific activity from the BPMN model and generate 2 distinct, meaningful variations (e.g., replacing 'Approve' with 'Agree' or 'User Approval'). Ensure that these 2 variations collectively appear in exactly 5% of the total trace instances.
* **Consistency:** Except injected error, All traces MUST use the exact string names defined in the BPMN model as event name. The name of resources should be consistent with those mentioned in the process model.
* **Missing Data:** When a timestamp is "missing," represent it as None or NaT in the DataFrame before conversion, but ensure the Case ID remains intact. When a resources is "mising", represent it as an empty string (e.g., <string key="org:resource" value="" />)

"""

    def construct(
            self, 
            process_model: str = "", 
            config: str = "", 
            special_requirements: str = "", 
            llm: WatsonxLLM = None
            ) -> str:
        
        parts: list[str] = [self.prompt_template]
        parts.append("\n### 1. BPMN Process Model Reference\n")
        if process_model and process_model != "No Process Model given.":
            parts.append(
                "The script must generate traces that strictly adhere to the following BPMN logic:\n\n"
                f"```bpmn\n{process_model}\n```\n\n"
            )
        else:
            parts.append(
                "The script must generate traces that strictly adhere to one BPMN process model. The process model specification should follow the user requirements."
                "Specify the process model first, but do not return it.\n\n"
            )
        parts.append("### 2. Configuration Data\n")
        if config:
            parts.append(
                "Use the following JSON data as the parameters for the script:\n"
                f"```json\n{config}\n```\n\n"
            )
        else:
            parts.append("""
Use the following JSON data as the parameters for the script:
```json
{
    "user_preferences": {
        "language": "en",
        "tone": "technical",
        "user_is_student": true
    },
    "data": {
        "trace_num": 5000, 
        "notation": "BPMN",
        "trace_length": 5
    }
}
```
"""
            )
        if special_requirements:
            if llm is None or not self.requirement_template:
                raise ValueError(
                "Both llm and requirement_template are required when special_requirements is provided."
            )
            messages = [
            {
                "role": "system",
                "content":
                ("You are modifying the given requirements based on the user requirements."
                "Return requirements in the given format only."
                "Do not modify any information in the section 5.Technical Specifications."),
            },
            {
                "role": "user",
                "content": (
                    f"This is the requirement template:\n{self.requirement_template}\n\n"
                    f"This is the user requirements:\n{special_requirements}"
                ),
            },
            ]
            parts.append(str(llm.chat(messages)))
            parts.append("\n")
        else:
            parts.append(self.requirement_template)
        parts.append(
            """ 
### 5. Technical Specifications
* **Output Format:** The script should output a XES file named `event_log.xes`. 
* **XES Structure Required: ** The XES file should contains four parts: Extensions, Globals, Classifiers and Traces.
* **Traces Attributes Required:** `Case ID`, `Activity`, `Timestamp`, and `Resource`.
* **Libraries:** Use `pandas` and `random` for the generation logic.
* **Avoid Functions:**Avoide to use trace.extend. Avoid passing resource_key or resource_column to pm4py.format_dataframe. Instead, rename the DataFrame column to 'org:resource' manually before converting to an event log. Avoid using pm4py.write_xes, directly write data into xes file in xes format. Avoid using Pm4py library. Do not use `nonlocal current_time` unless `current_time` has already been initialized in the enclosing function scope. Do not call `.astimezone()` on pandas `Timestamp`.
* **Function usage:**When using `random.sample()`, always pass a sequence such as a `list` or `sorted()` result, not a `set` or dictionary view; for example, use `random.sample(sorted(my_set), k)` instead of `random.sample(my_set, k)`.
* **Normalize timestamps:**Always normalize timestamps before sorting or exporting: use `df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)`.
* **Avoid empty random choice:**Before using `random.choice()`, ensure the candidate list is non-empty; if it is empty, use a fallback list or default value.
* **Keep event tuple format consistent:**Use one consistent event tuple format throughout the script, e.g. `(event_id, activity, timestamp)`, and never mix 2-element and 3-element event tuples.
* **timestamps constraint:**All timestamps must be normalized to timezone-aware UTC using `df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)` before any sorting, comparison, grouping, or XES export, and the script must never mix timezone-naive and timezone-aware datetime values.
* **timestamps constraint:**Do not use `nonlocal` unless the referenced variable is defined in an enclosing function scope; for counters, define the counter in the same function, use a mutable object such as `event_counter = {"value": 0}`, or use `global` only when the variable is defined at module level.
***"""
        )
        parts.append(
        "\n\n**Instruction for the LLM:** Please generate the complete Python code now "
        "and return pure code only, without ```python at the beginning.\n")

        self.prompt = "".join(parts)
        return self.prompt