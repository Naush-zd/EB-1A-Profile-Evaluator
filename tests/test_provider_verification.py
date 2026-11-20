from eb1a.providers import FailoverLLMProvider


class VerifiableProvider:
    provider_name = "test-provider"

    def verify_credentials(self):
        return {"provider": self.provider_name, "ok": True}


def test_failover_verification_reports_each_provider():
    result = FailoverLLMProvider([VerifiableProvider()]).verify_credentials()

    assert result["provider"] == "failover"
    assert result["ok"] is True
    assert result["providers"] == [{"provider": "test-provider", "ok": True}]
