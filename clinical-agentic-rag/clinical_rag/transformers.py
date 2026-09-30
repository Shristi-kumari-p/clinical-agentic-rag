"""clinical_rag/transformers.py"""

import json
from typing import Dict, Any, Union
from . import config


def _parse_item(item: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except (json.JSONDecodeError, TypeError):
            return {}
    return item if isinstance(item, dict) else {}


def transform_clinical_trial(item: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    item = _parse_item(item)
    
    nct_id = item.get("NCT_ID") or item.get("nctId", "unknown")
    title = item.get("Title") or item.get("briefTitle", "No title available")
    
    raw_conditions = item.get("Conditions") or []
    raw_interventions = item.get("Interventions") or []
    summary = item.get("Summary") or ""

    # Ensure conditions is a list of strings
    if isinstance(raw_conditions, str):
        conditions = [raw_conditions]
    elif isinstance(raw_conditions, list):
        conditions = [
            c if isinstance(c, str) else str(c.get("name", c))
            for c in raw_conditions if c
        ]
    else:
        conditions = []

    # Ensure interventions is a list of strings (extract 'name' if dict)
    if isinstance(raw_interventions, str):
        interventions = [raw_interventions]
    elif isinstance(raw_interventions, list):
        interventions = [
            i if isinstance(i, str) else str(i.get("name", i))
            for i in raw_interventions if i
        ]
    else:
        interventions = []

    narrative_parts = [
        f"Title: {title}",
        f"Conditions: {', '.join(conditions)}" if conditions else "",
        f"Interventions: {', '.join(interventions)}" if interventions else "",
        f"Summary: {summary}"
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


def transform_fda_event(item: Union[Dict[str, Any], str], idx: int = 0) -> Dict[str, Any]:
    item = _parse_item(item)
    patient = item.get("patient", {})

    safety_id = item.get("safetyreportid") or item.get("safety_report_id") or f"report_{idx}"

    drug_list = patient.get("drug", []) if isinstance(patient, dict) else item.get("drug", [])
    reaction_list = patient.get("reaction", []) if isinstance(patient, dict) else item.get("reaction", [])

    drugs = [d.get("medicinalproduct") for d in drug_list if isinstance(d, dict) and d.get("medicinalproduct")] if isinstance(drug_list, list) else []
    reactions = [r.get("reactionmeddrapt") for r in reaction_list if isinstance(r, dict) and r.get("reactionmeddrapt")] if isinstance(reaction_list, list) else []

    title = f"FDA Adverse Event Report {safety_id}"
    narrative_parts = [
        f"Report ID: {safety_id}",
        f"Suspect Drugs: {', '.join(drugs)}" if drugs else "Suspect Drugs: Unspecified",
        f"Reported Adverse Reactions: {', '.join(reactions)}" if reactions else "Reactions: Unspecified",
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
    
    # Read flattened PubMed column names created by ingest.py
    pmid = item.get("PMID") or item.get("pmid", "unknown")
    title = item.get("Title") or item.get("title", "No Title")
    abstract = item.get("Abstract") or item.get("abstract", "")
    journal = item.get("Journal") or item.get("journal", "")
    pub_year = item.get("Publication_Year") or item.get("pub_date", "")

    narrative_parts = [
        f"Title: {title}",
        f"Journal: {journal} ({pub_year})" if journal else "",
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
            "pub_date": str(pub_year),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }
    }