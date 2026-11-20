from dataclasses import dataclass


@dataclass
class ErrorReport:
    code: str
    message: str
    details: dict[str, str]

    def as_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "details": self.details}


class Eb1aError(Exception):
    code = "eb1a_error"

    def report(self) -> ErrorReport:
        return ErrorReport(self.code, str(self), {})
