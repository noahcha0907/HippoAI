"""Provider boundary tests: no real API calls or credentials."""

import json
import os
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from kirby.astra import ASTRA_MODEL, AstraClient, AstraConfig, ProviderFailure, provider_error_details, validate_proposal
from kirby.cli import main
from kirby.records import canonical_json


def packet():
    return {"base_checkpoint_id": "a" * 64, "evidence_sha256": "b" * 64,
            "evidence_ids": ["train-1", "layer:0"],
            "activation_summary": [{"layer": 0, "columns": [{"column": 7}, {"column": 9}]}]}


def proposal(evidence=None):
    evidence = evidence or packet()
    return {"decision": "propose", "base_checkpoint_id": evidence["base_checkpoint_id"],
            "evidence_sha256": evidence["evidence_sha256"], "hypothesis": "A bounded experimental reduction may change the measured errors.",
            "evidence_ids": [evidence["evidence_ids"][0]],
            "operation": {"operation": "scale_columns", "layer": 0, "columns": [7], "scale": 0.97}}


def response(text=None, status="completed"):
    return {"id": "resp_synthetic", "object": "response", "created_at": 0, "model": ASTRA_MODEL,
            "status": status, "output": [{"type": "message", "id": "msg_synthetic", "role": "assistant",
                                          "status": "completed", "content": [{"type": "output_text", "annotations": [],
                                          "text": text if text is not None else json.dumps(proposal())}]}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 5}}}


class AstraTests(unittest.TestCase):
    def test_valid_proposal_and_explicit_abstention(self):
        data, operation = validate_proposal(json.dumps(proposal()), packet(), "request1")
        self.assertEqual(operation.columns, (7,))
        self.assertEqual(operation.scale, 0.97)
        value = proposal()
        value.update(decision="abstain", operation=None)
        self.assertIsNone(validate_proposal(json.dumps(value), packet(), "request2")[1])

    def test_untrusted_proposals_cannot_change_code_evidence_or_bounds(self):
        cases = []
        for key, value in (("code", "delete everything"), ("base_checkpoint_id", "c" * 64),
                           ("evidence_sha256", "d" * 64), ("evidence_ids", ["final-question"]),
                           ("decision", "execute"), ("hypothesis", "")):
            item = proposal()
            item[key] = value
            cases.append(item)
        for key, value in (("operation", "replace_tensor"), ("layer", True), ("layer", 99),
                           ("columns", [8]), ("columns", [7, 7]), ("columns", [True]),
                           ("scale", 1.5), ("scale", float("nan")), ("scale", "0.97")):
            item = proposal()
            item["operation"][key] = value
            cases.append(item)
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_proposal(json.dumps(item), packet(), "request1")
        duplicate = json.dumps(proposal()).replace('"decision": "propose"', '"decision": "propose", "decision": "abstain"')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_proposal(duplicate, packet(), "request1")

    def test_sensitivity_packet_requires_citing_a_measured_probe(self):
        evidence = packet()
        evidence.update(schema_version=2, training_sensitivity=[{"evidence_id": "probe-1"}])
        evidence["evidence_ids"].append("probe-1")
        value = proposal(evidence)
        with self.assertRaisesRegex(ValueError, "cite measured"):
            validate_proposal(json.dumps(value), evidence, "request1")
        value["evidence_ids"].append("probe-1")
        self.assertIsNotNone(validate_proposal(json.dumps(value), evidence, "request1")[1])

    def test_request_is_bounded_structured_and_single_use(self):
        client = Mock()
        client.responses.create.return_value = response()
        editor = AstraClient(client=client)
        body = editor.request_body(packet())
        self.assertEqual(body["model"], ASTRA_MODEL)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertFalse(body["store"])
        self.assertNotIn("tools", body)
        audit = editor.request(body, "request1")
        self.assertEqual(audit["usage"]["total_tokens"], 120)
        self.assertEqual(audit["usage"]["reasoning_tokens"], 5)
        with self.assertRaisesRegex(ProviderFailure, "One API request"):
            editor.request(body, "request1")
        self.assertEqual(client.responses.create.call_count, 1)
        with self.assertRaisesRegex(ValueError, "byte ceiling"):
            editor.request_body({"untrusted_text": "x" * 128_000})

    def test_refusal_incomplete_and_tool_calls_never_become_proposals(self):
        refused = response()
        refused["output"][0]["content"] = [{"type": "refusal", "refusal": "Declined"}]
        tool_call = response()
        tool_call["output"].append({"type": "function_call", "name": "execute"})
        for item in (refused, response(status="incomplete"), tool_call):
            with self.subTest(item=item):
                client = Mock()
                client.responses.create.return_value = item
                editor = AstraClient(client=client)
                with self.assertRaises(ProviderFailure) as caught:
                    editor.request(editor.request_body(packet()), "request1")
                self.assertEqual(caught.exception.audit["usage"]["total_tokens"], 120)

    def test_provider_errors_do_not_leak_secrets_or_retry(self):
        client = Mock()
        client.responses.create.side_effect = RuntimeError("Bearer sk-synthetic-secret")
        editor = AstraClient(client=client)
        with self.assertRaises(ProviderFailure) as caught:
            editor.request(editor.request_body(packet()), "request1")
        self.assertNotIn("sk-synthetic-secret", str(caught.exception) + canonical_json(caught.exception.audit))
        self.assertFalse(caught.exception.audit["usage_known"])
        self.assertEqual(client.responses.create.call_count, 1)

    def test_missing_key_does_not_make_a_request(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), self.assertRaises(ProviderFailure) as caught:
            AstraClient().connect()
        self.assertEqual(caught.exception.audit["api_calls"], 0)

    def test_error_diagnostics_distinguish_quota_and_request_schema(self):
        for status, code, param, hint in (
            (429, "insufficient_quota", None, "credits"),
            (429, "rate_limit_exceeded", None, "Rate or quota"),
            (400, "invalid_json_schema", "text.format.schema", "request format"),
            (401, "invalid_api_key", None, "active API key"),
            (404, "model_not_found", "model", "model access"),
        ):
            with self.subTest(code=code):
                error = RuntimeError("Do not persist sk-synthetic-secret")
                error.status_code = status
                error.body = {"error": {"code": code, "param": param, "message": "sk-synthetic-secret"}}
                error.request_id = "req_synthetic"
                details = provider_error_details(error)
                self.assertEqual(details["provider_code"], code)
                self.assertEqual(details["provider_parameter"], param)
                self.assertEqual(details["request_id"], "req_synthetic")
                self.assertIn(hint, details["guidance"])
                self.assertNotIn("sk-synthetic-secret", canonical_json(details))

    def test_arbitrary_error_fields_are_not_persisted(self):
        error = RuntimeError("sk-synthetic-secret")
        error.body = {"code": "sk-synthetic-secret", "param": "sk-synthetic-secret"}
        error.request_id = "sk-synthetic-secret"
        details = provider_error_details(error)
        self.assertIsNone(details["provider_code"])
        self.assertIsNone(details["provider_parameter"])
        self.assertIsNone(details["request_id"])

    def test_cli_routes_to_astra_with_frozen_limits(self):
        import contextlib
        import io
        with patch("kirby.cli.run_astra_attempt") as run, patch("kirby.cli.run_baseline") as baseline:
            run.return_value = {"directory": "synthetic", "report": {"state": "completed", "outcome": "Astra abstained"}}
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["astra-attempt", "--manifest", "development.json", "--rollback-report", "report.json",
                             "--revision", "a" * 40, "--editor-max-output-tokens", "2048", "--diagnostic-count", "4"])
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.args[1], Path("report.json"))
            self.assertEqual(run.call_args.args[4].config.max_output_tokens, 2048)
            self.assertEqual(run.call_args.args[6].diagnostic_count, 4)
            baseline.assert_not_called()


try:
    import httpx
    from openai import OpenAI
except ImportError:
    OpenAI = None


@unittest.skipIf(OpenAI is None, "Install astra extra for real SDK/HTTP-boundary tests")
class AstraSDKTests(unittest.TestCase):
    def test_real_sdk_serialization_with_mock_http(self):
        calls = []
        def handle(request):
            calls.append(request)
            self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
            body = json.loads(request.content)
            self.assertEqual(body["text"]["format"]["name"], "kirby_column_proposal_v1")
            self.assertEqual(request.headers["X-Client-Request-Id"], "test-http")
            self.assertNotIn("sk-synthetic-secret", request.content.decode())
            return httpx.Response(200, json=response(), headers={"x-request-id": "req_synthetic"})
        with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
            with OpenAI(api_key="sk-synthetic-secret", base_url="https://api.openai.com/v1", max_retries=0,
                        http_client=transport) as client:
                editor = AstraClient(client=client)
                audit = editor.request(editor.request_body(packet()), "test-http")
        self.assertEqual(len(calls), 1)
        self.assertEqual(audit["request_id"], "req_synthetic")
        self.assertEqual(audit["usage"]["total_tokens"], 120)
        validate_proposal(audit["output_text"], packet(), "test-http")

    def test_real_sdk_server_failure_is_not_retried_or_logged_raw(self):
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(503, json={"error": {"message": "sk-synthetic-secret", "type": "server_error"}})
        with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
            with OpenAI(api_key="sk-synthetic-secret", max_retries=0, http_client=transport) as client:
                editor = AstraClient(client=client)
                with self.assertRaises(ProviderFailure) as caught:
                    editor.request(editor.request_body(packet()), "failure-test")
        self.assertEqual(len(calls), 1)
        self.assertEqual(caught.exception.audit["http_status"], 503)
        self.assertNotIn("sk-synthetic-secret", canonical_json(caught.exception.audit))


if __name__ == "__main__":
    unittest.main()
