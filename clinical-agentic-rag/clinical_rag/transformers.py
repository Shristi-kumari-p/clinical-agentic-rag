"""clinical_rag/transformers.py"""

from typing import Dict, Any
from . import config

def transform_clinical_trial(item: Dict[str, Any]) -> Dict[str, Any]:
    protocol = item.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    nct_id = ident.get("nctId", "unknown")
    title = ident.get("briefTitle", "No title available")

    conditions = protocol.get("conditionsModule", {}).get("conditions", [])
    interventions_list = protocol.get("armsInterventionsModule", {}).get("interventions", [])
    interventions = [i.get("name") for i in interventions_list if i.get("name")]
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

def transform_fda_event(item: Dict[str, Any]) -> Dict[str, Any]:
    safety_id = item.get("safetyreportid", "unknown")
    patient = item.get("patient", {})

    drugs = [d.get("medicinalproduct") for d in patient.get("drug", []) if d.get("medicinalproduct")]
    reactions = [r.get("reactionmeddrapt") for r in patient.get("reaction", []) if r.get("reactionmeddrapt")]

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

def transform_pubmed_article(item: Dict[str, Any]) -> Dict[str, Any]:
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