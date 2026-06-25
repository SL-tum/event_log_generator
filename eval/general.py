import pandas as pd
from typing import Dict, Any, Union, Optional
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
REQUIRED_COLUMNS = ["case_id", "activity", "timestamp", "resource"]
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

def load_and_check_event_log(
    log_input: str,
    required_columns: Optional[list] = None
) -> Dict[str, Any]:
    """
    Load an event log from a XES file or validate an existing pandas DataFrame.

    Parameters
    ----------
    log_input:
        A path to a .xes file.

    required_columns:
        Required normalized column names. If None, the default columns are:
        case_id, activity, timestamp, resource.

    Returns
    -------
    Dict[str, Any]
        A structured validation result containing:
        - readable: whether the input can be read
        - valid: whether the log satisfies the basic format requirements
        - log_df: normalized event log as pandas DataFrame if readable
        - issues: list of detected problems
    """

    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    issues = []
    log_df = None
    attributes = {
        "log_length": 0,
        "missing_values": {},
        "non_chronological_cases": 0,
        "activities": {},
        "resources": {}
    }
    # --------------------------------------------------
    # Step 1: Load event log
    # --------------------------------------------------
    if isinstance(log_input, str):
        try:
            event_log = pm4py.read_xes(log_input)
            log_object = pm4py.convert_to_event_log(event_log)
            log_df = pm4py.convert_to_dataframe(event_log)

            readable = True

        except Exception as e:
            return {
                "readable": False,
                "valid": False,
                "log_df": None,
                "attributes": None,
                "issues": [
                    f"Failed to read the XES file: {str(e)}"
                ]
            }

    else:
        return {
            "readable": False,
            "valid": False,
            "log_df": None,
            "attributes": None,
            "issues": [
                "Unsupported input type. Please provide a XES file path."
            ]
        }

    # --------------------------------------------------
    # Normalize common XES / PM4Py column names
    # --------------------------------------------------
    column_mapping = {
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp",
        "org:resource": "resource"
    }

    log_df = log_df.rename(columns=column_mapping)


    # --------------------------------------------------
    # Check required columns
    # --------------------------------------------------
    missing_columns = [
        col for col in required_columns
        if col not in log_df.columns
    ]

    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")

    # --------------------------------------------------
    # Check log length
    # --------------------------------------------------
    if log_df.empty:
        issues.append("The event log is empty.")
        attributes["log_length"] = 0
    else: 
        attributes["log_length"] = len(log_object)

    # --------------------------------------------------
    # Check missing values
    # --------------------------------------------------
    
    null_counts_dict = log_df.isna().sum().to_dict()
    
    attributes["missing_values"] = null_counts_dict
    #attributes["missing_values"]["resource"] = int(attributes["missing_values"]["resource"]) + int((log_df["resource"].astype(str).str.strip() == "").sum())


    # --------------------------------------------------
    # Check Activities
    # --------------------------------------------------
    
    activities_dict = log_df['activity'].value_counts().to_dict()
    
    attributes["activities"] = activities_dict

    # --------------------------------------------------
    # Check Resources
    # --------------------------------------------------
    
    resources_dict = log_df['resource'].value_counts().to_dict()
    
    attributes["resources"] = resources_dict

    # --------------------------------------------------
    # Check timestamp parseability
    # --------------------------------------------------
    if "timestamp" in log_df.columns:
        try:
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        except Exception as e:
            issues.append(f"Timestamp column cannot be parsed as datetime: {str(e)}")

    # --------------------------------------------------
    # Step 7: Check chronological order within each case
    # --------------------------------------------------
    num = 0
    if "case_id" in log_df.columns and "timestamp" in log_df.columns:
        try:
            for case_id, group in log_df.groupby("case_id"):
                timestamps = group["timestamp"].tolist()
                if timestamps != sorted(timestamps):
                    num = num + 1
                    issues.append(
                        f"Events in case '{case_id}' are not in chronological order."
                    )
            attributes["non_chronological_cases"] = num
        except Exception as e:
            issues.append(f"Failed to check chronological order: {str(e)}")

    return {
        "readable": readable,
        "valid": len(issues) == 0,
        "log_df": log_df if readable else None,
        "attributes": attributes,
        "issues": issues
    }


def parse_bpmn_xml(bpmn_path):
    """
    Parse a BPMN XML file and extract its main process structure.

    Returns
    -------
    dict
        Parsed BPMN structure containing:
        - participants
        - processes
        - lanes
        - nodes
        - tasks
        - gateways
        - events
        - sequence_flows
        - graph
        - node_to_lane
    """

    tree = ET.parse(bpmn_path)
    root = tree.getroot()

    ns = {
        "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL"
    }

    result = {
        "participants": {},
        "processes": {},
        "lanes": {},
        "nodes": {},
        "tasks": {},
        "gateways": {},
        "events": {},
        "sequence_flows": [],
        "graph": defaultdict(list),
        "reverse_graph": defaultdict(list),
        "node_to_lane": {}
    }

    # --------------------------------------------------
    # Participants
    # --------------------------------------------------
    for participant in root.findall(".//bpmn:participant", ns):
        pid = participant.attrib.get("id")
        result["participants"][pid] = {
            "id": pid,
            "name": participant.attrib.get("name"),
            "process_ref": participant.attrib.get("processRef")
        }

    # --------------------------------------------------
    # Processes
    # --------------------------------------------------
    for process in root.findall(".//bpmn:process", ns):
        process_id = process.attrib.get("id")
        result["processes"][process_id] = {
            "id": process_id,
            "name": process.attrib.get("name"),
            "is_executable": process.attrib.get("isExecutable"),
            "process_type": process.attrib.get("processType")
        }

        # --------------------------------------------------
        # Lanes and node-to-lane mapping
        # --------------------------------------------------
        for lane in process.findall(".//bpmn:lane", ns):
            lane_id = lane.attrib.get("id")
            lane_name = lane.attrib.get("name", "")

            flow_node_refs = [
                ref.text for ref in lane.findall("bpmn:flowNodeRef", ns)
                if ref.text is not None
            ]

            result["lanes"][lane_id] = {
                "id": lane_id,
                "name": lane_name,
                "process_id": process_id,
                "flow_node_refs": flow_node_refs
            }

            for node_id in flow_node_refs:
                result["node_to_lane"][node_id] = {
                    "lane_id": lane_id,
                    "lane_name": lane_name
                }

        # --------------------------------------------------
        # Flow nodes
        # --------------------------------------------------
        node_tags = {
            "task": "task",
            "userTask": "task",
            "serviceTask": "task",
            "manualTask": "task",
            "businessRuleTask": "task",
            "scriptTask": "task",
            "sendTask": "task",
            "receiveTask": "task",

            "exclusiveGateway": "gateway",
            "parallelGateway": "gateway",
            "inclusiveGateway": "gateway",
            "eventBasedGateway": "gateway",

            "startEvent": "event",
            "endEvent": "event",
            "intermediateCatchEvent": "event",
            "intermediateThrowEvent": "event"
        }

        for tag, category in node_tags.items():
            for elem in process.findall(f".//bpmn:{tag}", ns):
                node_id = elem.attrib.get("id")
                node_name = elem.attrib.get("name", "")
                incoming = [
                    x.text for x in elem.findall("bpmn:incoming", ns)
                    if x.text is not None
                ]
                outgoing = [
                    x.text for x in elem.findall("bpmn:outgoing", ns)
                    if x.text is not None
                ]

                node_info = {
                    "id": node_id,
                    "name": node_name,
                    "type": tag,
                    "category": category,
                    "process_id": process_id,
                    "incoming": incoming,
                    "outgoing": outgoing,
                    "lane": result["node_to_lane"].get(node_id)
                }

                result["nodes"][node_id] = node_info

                if category == "task":
                    result["tasks"][node_id] = node_info
                elif category == "gateway":
                    result["gateways"][node_id] = node_info
                elif category == "event":
                    result["events"][node_id] = node_info

        # --------------------------------------------------
        # Sequence flows
        # --------------------------------------------------
        for flow in process.findall(".//bpmn:sequenceFlow", ns):
            flow_id = flow.attrib.get("id")
            source = flow.attrib.get("sourceRef")
            target = flow.attrib.get("targetRef")

            flow_info = {
                "id": flow_id,
                "name": flow.attrib.get("name", ""),
                "source": source,
                "target": target,
                "process_id": process_id
            }

            result["sequence_flows"].append(flow_info)
            result["graph"][source].append(target)
            result["reverse_graph"][target].append(source)

    # Convert defaultdict to normal dict for cleaner output
    result["graph"] = dict(result["graph"])
    result["reverse_graph"] = dict(result["reverse_graph"])

    return result