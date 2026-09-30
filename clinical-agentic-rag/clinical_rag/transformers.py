"""clinical_rag/transformers.py"""

import json
from typing import Dict, Any, Union
from . import config


def _parse_item(item: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Ensures input items are parsed into dictionaries even if loaded as raw JSON strings."""
    if isinstance(item, str):
        try:
            item = json.loads(item)
            # Handle edge case where json.loads returns another string (double-encoded JSON)
            if isinstance(item, str):
                item = json.loads(item)
        except (json.JSONDecodeError, TypeError):
            return {}
    return item if isinstance(item, dict) else {}


def transform_clinical_trial(item: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    item = _parse_item(item)
    protocol = item.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    nct_id = ident.get("nctId", "unknown")
    title = ident.get("briefTitle", "No title available")

    conditions = protocol.get("conditionsModule", {}).get("conditions", [])
    interventions_list = protocol.get("armsInterventionsModule", {}).get("interventions", [])
    interventions = [i.get("name") for i in interventions_list if isinstance(i, dict) and i.get("name")]
    description = protocol.get("descriptionModule", {}).get("briefSummary", "")

    narrative_parts = [
        f"Title: {title}",
        f"Conditions: {', '.join(conditions)}" if conditions else "",
        f"Interventions: {', '.join(interventions)}" if interventions else "",
        f"Summary: {description}"
    ]

    return {
        "document_id": f"ct_{nct_id}",
        "source": config.SOURCE_CLINICAL,
        "title": title,
        "text": "\n".join([p for p in narrative_parts if p]),
        "metadata": {
            "nct_id": nct_id,
            "conditions": conditions,
            "interventions": interventions,
            "url": f"https://clinicaltrials.gov/study/{nct_id}"
        }
    }


def transform_fda_event(item: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    item = _parse_item(item)
    safety_id = item.get("safetyreportid", "unknown")
    patient = item.get("patient", {})

    drug_list = patient.get("drug", []) if isinstance(patient, dict) else []
    reaction_list = patient.get("reaction", []) if isinstance(patient, dict) else []

    drugs = [d.get("medicinalproduct") for d in drug_list if isinstance(d, dict) and d.get("medicinalproduct")]
    reactions = [r.get("reactionmeddrapt") for r in reaction_list if isinstance(r, dict) and r.get("reactionmeddrapt")]

    title = f"FDA Adverse Event Report {safety_id}"
    narrative_parts = [
        f"Report ID: {safety_id}",
        f"Suspect Drugs: {', '.join(drugs)}" if drugs else "",
        f"Reported Adverse Reactions: {', '.join(reactions)}" if reactions else "",
    ]

    return {
        "document_id": f"fda_{safety_id}",
        "source": config.SOURCE_FDA,
        "title": title,
        "text": "\n".join([p for p in narrative_parts if p]),
        "metadata": {
            "safety_report_id": safety_id,
            "drugs": drugs,
            "reactions": reactions,
            "url": f"https://api.fda.gov/drug/event.json?search=safetyreportid:{safety_id}"
        }
    }


def transform_pubmed_article(item: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    item = _parse_item(item)
    pmid = item.get("pmid", "unknown")
    title = item.get("title", "No Title")
    abstract = item.get("abstract", "")
    journal = item.get("journal", "")
    pub_date = item.get("pub_date", "")

    narrative_parts = [
        f"Title: {title}",
        f"Journal: {journal} ({pub_date})" if journal else "",
        f"Abstract: {abstract}"
    ]

    return {
        "document_id": f"pm_{pmid}",
        "source": config.SOURCE_PUBMED,
        "title": title,
        "text": "\n".join([p for p in narrative_parts if p]),
        "metadata": {
            "pmid": pmid,
            "journal": journal,
            "pub_date": pub_date,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }
    }