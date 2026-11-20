from enum import StrEnum


class CriterionId(StrEnum):
    AWARDS = "awards"
    MEMBERSHIP = "membership"
    PUBLISHED_MATERIAL = "published_material"
    JUDGING = "judging"
    ORIGINAL_CONTRIBUTIONS = "original_contributions"
    SCHOLARLY_ARTICLES = "scholarly_articles"
    EXHIBITIONS = "exhibitions"
    LEADING_CRITICAL_ROLE = "leading_critical_role"
    HIGH_SALARY = "high_salary"
    COMMERCIAL_SUCCESS = "commercial_success"


CRITERIA = {
    CriterionId.AWARDS: "Less commonly recognized prizes or awards for excellence in the field.",
    CriterionId.MEMBERSHIP: "Membership in associations requiring outstanding achievements.",
    CriterionId.PUBLISHED_MATERIAL: "Published material about the candidate in professional or major media.",
    CriterionId.JUDGING: "Participation as a judge of the work of others in the field.",
    CriterionId.ORIGINAL_CONTRIBUTIONS: "Original scientific, scholarly, artistic, athletic, or business-related contributions of major significance.",
    CriterionId.SCHOLARLY_ARTICLES: "Authorship of scholarly articles in professional or major trade publications or other major media.",
    CriterionId.EXHIBITIONS: "Display of the candidate's work at artistic exhibitions or showcases.",
    CriterionId.LEADING_CRITICAL_ROLE: "Leading or critical role for organizations or establishments with a distinguished reputation.",
    CriterionId.HIGH_SALARY: "High salary or other significantly high remuneration relative to others in the field.",
    CriterionId.COMMERCIAL_SUCCESS: "Commercial success in the performing arts.",
}

CRITERION_QUERY_TERMS = {
    CriterionId.AWARDS: "award prize recognized excellence field",
    CriterionId.MEMBERSHIP: "membership association outstanding achievement",
    CriterionId.PUBLISHED_MATERIAL: "published material about person professional major media",
    CriterionId.JUDGING: "judge judging work of others field",
    CriterionId.ORIGINAL_CONTRIBUTIONS: "original contributions major significance",
    CriterionId.SCHOLARLY_ARTICLES: "authorship scholarly articles professional trade publication media",
    CriterionId.EXHIBITIONS: "display work artistic exhibition showcase",
    CriterionId.LEADING_CRITICAL_ROLE: "leading critical role distinguished organization reputation",
    CriterionId.HIGH_SALARY: "high salary significantly high remuneration others field",
    CriterionId.COMMERCIAL_SUCCESS: "commercial success performing arts box office sales",
}
