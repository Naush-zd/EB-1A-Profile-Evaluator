"""Optional public-opportunity discovery for weak criteria.

This module intentionally sends no resume text or personal identifiers to the
search provider. Results are leads, never evidence of an EB-1A criterion.
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from .criteria import CriterionId
from .schemas import Assessment


class OpportunityLead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: CriterionId
    title: str
    url: str
    description: str = ""
    source: str = ""
    date_hint: str | None = None
    fit_reason: str
    status: str = "lead_only"


class OpportunitySearchProvider:
    provider_name = "abstract"

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        raise NotImplementedError


class TavilySearchProvider(OpportunitySearchProvider):
    provider_name = "tavily"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError("opportunity search requires TAVILY_API_KEY; no leads were generated")

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        body = json.dumps({"api_key": self.api_key, "query": query, "max_results": max_results, "search_depth": "basic", "include_answer": False}).encode()
        request = urllib.request.Request("https://api.tavily.com/search", data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"opportunity search failed: {exc}") from exc
        return payload.get("results", [])


def suggest_opportunities(assessment: Assessment, field: str, provider: OpportunitySearchProvider, max_criteria: int = 2, max_results: int = 3) -> dict:
    ranked = sorted(
        assessment.criteria,
        key=lambda item: {"insufficient_info": 0, "missing": 1, "partial": 2, "strong": 3}[item.rating],
    )[:max_criteria]
    leads = []
    queries = []
    for item in ranked:
        criterion = item.criterion_id.value.replace("_", " ")
        query = f"{field} {criterion} opportunity call for applications professional"
        queries.append({"criterion_id": item.criterion_id.value, "query": query})
        for result in provider.search(query, max_results=max_results):
            url = str(result.get("url", "")).strip()
            title = str(result.get("title", "")).strip()
            if not url or not title:
                continue
            leads.append(OpportunityLead(
                criterion_id=item.criterion_id,
                title=title,
                url=url,
                description=str(result.get("content", result.get("snippet", "")))[:1000],
                source=url.split("/")[2] if "://" in url else "",
                date_hint=str(result.get("published_date", "")) or None,
                fit_reason=f"Public lead related to the profile gap for {criterion} in the declared field.",
            ))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "field": field,
        "provider": provider.provider_name,
        "queries": queries,
        "leads": [lead.model_dump(mode="json") for lead in leads],
        "disclaimer": "These are public opportunity leads, not evidence of eligibility, accomplishments, or USCIS approval.",
    }
