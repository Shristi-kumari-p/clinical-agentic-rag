import pandas as pd

from clinical_rag.documents import build_clinical_documents, build_fda_documents, clean_documents
from clinical_rag.ingest import parse_pubmed_xml


def test_clinical_documents_handle_lists_and_nan():
    df = pd.DataFrame([{
        "NCT_ID": "NCT1", "Title": "T", "Official_Title": None, "Status": "Completed",
        "Study_Type": "Interventional", "Phase": ["PHASE3"], "Conditions": ["Hepatitis B"],
        "Summary": "A summary long enough to survive cleaning." * 3,
        "Detailed_Description": float("nan"), "Eligibility": "x", "Enrollment": 10,
        "Interventions": [{"name": "Drug"}], "Primary_Outcomes": None, "Secondary_Outcomes": None,
    }])
    docs = build_clinical_documents(df)
    assert docs[0]["document_id"] == "NCT1"
    assert "Hepatitis B" in docs[0]["text"]
    assert "nan" not in docs[0]["text"].lower().replace("n/a", "")


def test_fda_documents_skip_empty_values():
    df = pd.DataFrame([{"safetyreportid": "123", "serious": "1", "empty": None}])
    doc = build_fda_documents(df)[0]
    assert doc["document_id"] == "123"
    assert "empty" not in doc["text"]


def test_clean_removes_short_and_duplicates():
    long_text = "x" * 80
    docs = [
        {"text": "short", "source": "A", "document_id": "1", "metadata": {}},
        {"text": long_text, "source": "A", "document_id": "2", "metadata": {}},
        {"text": long_text, "source": "A", "document_id": "2", "metadata": {}},
    ]
    assert len(clean_documents(docs)) == 1


def test_parse_pubmed_xml():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article>
    <Journal><Title>J Test</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>Title</ArticleTitle><Abstract><AbstractText>Abs.</AbstractText></Abstract>
    <AuthorList><Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author></AuthorList>
    </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    rec = parse_pubmed_xml(xml)[0]
    assert rec["PMID"] == "42" and rec["Authors"] == "Jane Doe" and rec["Publication_Year"] == "2024"
