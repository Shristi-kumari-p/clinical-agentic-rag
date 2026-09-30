"""Step 1 - download raw data from ClinicalTrials.gov, openFDA and PubMed."""
import json
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from . import config

# Nested API field -> simple column name
CT_COLUMNS = {
    "protocolSection.identificationModule.nctId": "NCT_ID",
    "protocolSection.identificationModule.briefTitle": "Title",
    "protocolSection.identificationModule.officialTitle": "Official_Title",
    "protocolSection.statusModule.overallStatus": "Status",
    "protocolSection.designModule.studyType": "Study_Type",
    "protocolSection.designModule.phases": "Phase",
    "protocolSection.conditionsModule.conditions": "Conditions",
    "protocolSection.descriptionModule.briefSummary": "Summary",
    "protocolSection.descriptionModule.detailedDescription": "Detailed_Description",
    "protocolSection.eligibilityModule.eligibilityCriteria": "Eligibility",
    "protocolSection.designModule.enrollmentInfo.count": "Enrollment",
    "protocolSection.armsInterventionsModule.interventions": "Interventions",
    "protocolSection.outcomesModule.primaryOutcomes": "Primary_Outcomes",
    "protocolSection.outcomesModule.secondaryOutcomes": "Secondary_Outcomes",
}


def save_frame(df: pd.DataFrame, name: str) -> None:
    """JSONL keeps list/dict cells intact; CSV is just for eyeballing in VS Code."""
    df.to_json(config.RAW_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_csv(config.RAW_DIR / f"{name}.csv", index=False)


def load_frame(name: str) -> list[dict]:
    path_jsonl = config.RAW_DIR / f"{name}.jsonl"
    path_json = config.RAW_DIR / f"{name}.json"

    if path_jsonl.exists():
        df = pd.read_json(path_jsonl, orient="records", lines=True)
        return df.to_dict(orient="records")
    elif path_json.exists():
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("results", data) if isinstance(data, dict) else data
    else:
        raise FileNotFoundError(f"Data file for '{name}' not found in {config.RAW_DIR}.")


# ---------------------------------------------------------------- ClinicalTrials.gov
def fetch_clinical_trials(target: int = config.N_CLINICAL_TRIALS) -> pd.DataFrame:
    params = {"pageSize": min(config.CT_PAGE_SIZE, target)}
    studies: list[dict] = []

    while len(studies) < target:
        response = requests.get(config.CT_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        studies.extend(data.get("studies", []))
        print(f"  ClinicalTrials.gov downloaded: {len(studies)}")

        token = data.get("nextPageToken")
        if not token:
            print("  No more studies available.")
            break
        params["pageToken"] = token
        time.sleep(0.2)

    studies = studies[:target]
    df = pd.json_normalize(studies)
    df = df.reindex(columns=list(CT_COLUMNS)).rename(columns=CT_COLUMNS)
    save_frame(df, "clinical_trials")
    print(f"ClinicalTrials.gov: saved {len(df)} trials")
    return df


# ---------------------------------------------------------------- openFDA
def fetch_fda_events(target: int = config.N_FDA) -> pd.DataFrame:
    results: list[dict] = []
    skip = 0

    while len(results) < target:
        limit = min(config.FDA_PAGE_SIZE, target - len(results))
        response = requests.get(
            config.FDA_URL, params={"limit": limit, "skip": skip}, timeout=60
        )
        response.raise_for_status()
        batch = response.json().get("results", [])
        if not batch:
            break
        results.extend(batch)
        skip += len(batch)
        print(f"  FDA downloaded: {len(results)}")
        time.sleep(0.25)

    with open(config.RAW_DIR / "fda_drug_events.json", "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    df = pd.json_normalize(results)
    save_frame(df, "fda_drug_events")
    print(f"FDA: saved {len(df)} adverse-event reports")
    return df


# ---------------------------------------------------------------- PubMed
def _ncbi_params(extra: dict) -> dict:
    params = dict(extra)
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    if config.NCBI_EMAIL:
        params["email"] = config.NCBI_EMAIL
    return params


def search_pubmed(term: str, retmax: int) -> list[str]:
    response = requests.get(
        config.NCBI_ESEARCH,
        params=_ncbi_params(
            {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax}
        ),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["esearchresult"]["idlist"]


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//PubDate/Year")

        abstract = " ".join(
            "".join(part.itertext())
            for part in article.findall(".//Abstract/AbstractText")
        )

        authors = []
        for author in article.findall(".//Author"):
            name = f"{author.findtext('ForeName', '')} {author.findtext('LastName', '')}".strip()
            if name:
                authors.append(name)

        records.append(
            {
                "PMID": pmid_el.text if pmid_el is not None else "",
                "Title": "".join(title_el.itertext()) if title_el is not None else "",
                "Abstract": abstract,
                "Authors": ", ".join(authors),
                "Journal": journal_el.text if journal_el is not None else "",
                "Publication_Year": year_el.text if year_el is not None else "",
            }
        )
    return records


def fetch_pubmed(term: str = config.PUBMED_TERM, retmax: int = config.N_PUBMED) -> pd.DataFrame:
    pmids = search_pubmed(term, retmax)
    print(f"  PubMed IDs found: {len(pmids)}")

    records: list[dict] = []
    for start in range(0, len(pmids), 100):
        batch = pmids[start : start + 100]
        response = requests.get(
            config.NCBI_EFETCH,
            params=_ncbi_params({"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}),
            timeout=120,
        )
        response.raise_for_status()
        if start == 0:
            (config.RAW_DIR / "pubmed_articles.xml").write_text(response.text, encoding="utf-8")
        records.extend(parse_pubmed_xml(response.text))
        time.sleep(0.11 if config.NCBI_API_KEY else 0.4)

    df = pd.DataFrame(records)
    save_frame(df, "pubmed_articles")
    print(f"PubMed: saved {len(df)} articles")
    return df
