from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import Decimal
import json
from pathlib import Path
import unittest

from binance_spot_strategy.identity.manifests_v1 import (
    CandidateManifestV1,
    ContractDigest,
    HistoricalRunManifestV1,
    ModuleDigest,
    PaperRunManifestV1,
    RunCommonV1,
    candidate_manifest_from_payload,
    candidate_strategy_fingerprint,
    run_fingerprint,
    run_manifest_from_payload,
)
from binance_spot_strategy.protocols import Q18


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "identity_manifests_v1.json"
EXPECTED_CANDIDATE = "b5d1981da1427653c76a6cfc3504f15a5bf88b813b34029eb56ab1ec5b2d59ad"
EXPECTED_HISTORICAL = "72bd55a8c39767c3444455aa892e41ad80183f7bc3b3d37ce564ef0f95676fdc"
EXPECTED_PAPER = "5fbb792d5df25200720ba766349afa7a343855c750e20eb164eb90a6f6ef3051"


class StringSubclass(str):
    pass


class ManifestSubclass(CandidateManifestV1):
    pass


class FingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payloads = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        self.candidate_payload = deepcopy(self.payloads["candidate"])
        self.historical_payload = deepcopy(self.payloads["historical_run"])
        self.paper_payload = deepcopy(self.payloads["paper_run"])

    def test_synthetic_golden_fingerprints_are_exact_and_repeatable(self) -> None:
        candidate = candidate_manifest_from_payload(self.candidate_payload)
        historical = run_manifest_from_payload(self.historical_payload)
        paper = run_manifest_from_payload(self.paper_payload)

        self.assertIs(type(candidate), CandidateManifestV1)
        self.assertIs(type(historical), HistoricalRunManifestV1)
        self.assertIs(type(paper), PaperRunManifestV1)
        self.assertEqual(candidate_strategy_fingerprint(candidate), EXPECTED_CANDIDATE)
        self.assertEqual(run_fingerprint(historical), EXPECTED_HISTORICAL)
        self.assertEqual(run_fingerprint(paper), EXPECTED_PAPER)
        self.assertEqual(
            candidate_strategy_fingerprint(candidate),
            candidate_strategy_fingerprint(candidate),
        )
        self.assertEqual(run_fingerprint(historical), run_fingerprint(historical))
        self.assertEqual(historical.candidate_strategy_fingerprint, EXPECTED_CANDIDATE)
        self.assertEqual(paper.candidate_strategy_fingerprint, EXPECTED_CANDIDATE)
        self.assertNotEqual(run_fingerprint(historical), run_fingerprint(paper))

    def test_candidate_semantic_mutations_change_candidate_and_rebuilt_run(self) -> None:
        original_candidate = candidate_manifest_from_payload(self.candidate_payload)
        original_candidate_hash = candidate_strategy_fingerprint(original_candidate)
        original_run_hash = run_fingerprint(
            run_manifest_from_payload(self.historical_payload)
        )
        mutations = (
            ("shared_module", lambda p: p["shared_modules"][0].__setitem__("sha256", "0" * 64)),
            ("canonical", lambda p: p["protocols"].__setitem__("canonical_protocol_sha256", "0" * 64)),
            ("numeric", lambda p: p["protocols"].__setitem__("numeric_protocol_sha256", "0" * 64)),
            ("golden", lambda p: p["protocols"].__setitem__("golden_vectors_sha256", "0" * 64)),
            ("lock", lambda p: p["protocols"].__setitem__("semantic_dependency_lock_sha256", "0" * 64)),
        )
        for name, mutate in mutations:
            candidate_payload = deepcopy(self.candidate_payload)
            mutate(candidate_payload)
            changed_candidate = candidate_manifest_from_payload(candidate_payload)
            changed_candidate_hash = candidate_strategy_fingerprint(changed_candidate)
            run_payload = deepcopy(self.historical_payload)
            run_payload["candidate_strategy_fingerprint"] = changed_candidate_hash
            with self.subTest(name=name):
                self.assertNotEqual(changed_candidate_hash, original_candidate_hash)
                self.assertNotEqual(
                    run_fingerprint(run_manifest_from_payload(run_payload)),
                    original_run_hash,
                )

    def test_run_only_mutations_do_not_change_candidate(self) -> None:
        candidate = candidate_manifest_from_payload(self.candidate_payload)
        candidate_hash = candidate_strategy_fingerprint(candidate)
        original_run_hash = run_fingerprint(
            run_manifest_from_payload(self.historical_payload)
        )
        mutations = (
            ("broker", lambda p: p["broker_adapter"].__setitem__("sha256", "0" * 64)),
            ("stage", lambda p: p.__setitem__("stage_manifest_sha256", "0" * 64)),
            ("execution", lambda p: p.__setitem__("execution_manifest_sha256", "0" * 64)),
            ("environment", lambda p: p.__setitem__("environment_contract_sha256", "0" * 64)),
            ("raw", lambda p: p["input"].__setitem__("immutable_raw_data_manifest_sha256", "0" * 64)),
        )
        for name, mutate in mutations:
            payload = deepcopy(self.historical_payload)
            mutate(payload)
            with self.subTest(name=name):
                self.assertEqual(candidate_strategy_fingerprint(candidate), candidate_hash)
                self.assertNotEqual(
                    run_fingerprint(run_manifest_from_payload(payload)),
                    original_run_hash,
                )

    def test_unknown_or_missing_keys_are_rejected_at_every_level(self) -> None:
        candidate_targets = (
            (),
            ("baseline",),
            ("conventions",),
            ("contracts", 0),
            ("baseline_strategy_module",),
            ("shared_modules", 0),
            ("protocols",),
        )
        for path in candidate_targets:
            for mode in ("unknown", "missing"):
                payload = deepcopy(self.candidate_payload)
                target = payload
                for component in path:
                    target = target[component]
                if mode == "unknown":
                    target["renderer_metadata"] = "forbidden"
                else:
                    target.pop(next(iter(target)))
                with self.subTest(kind="candidate", path=path, mode=mode):
                    with self.assertRaises((TypeError, ValueError)):
                        candidate_manifest_from_payload(payload)

        run_targets = ((), ("broker_adapter",), ("input",))
        for source in (self.historical_payload, self.paper_payload):
            for path in run_targets:
                for mode in ("unknown", "missing"):
                    payload = deepcopy(source)
                    target = payload
                    for component in path:
                        target = target[component]
                    if mode == "unknown":
                        target["renderer_metadata"] = "forbidden"
                    else:
                        target.pop(next(iter(target)))
                    with self.subTest(schema=source["schema_version"], path=path, mode=mode):
                        with self.assertRaises((TypeError, ValueError)):
                            run_manifest_from_payload(payload)

    def test_metadata_and_run_results_are_never_accepted(self) -> None:
        forbidden = ("run_id", "database_id", "timestamp", "outcome", "renderer_metadata")
        for field in forbidden:
            for source, parser in (
                (self.candidate_payload, candidate_manifest_from_payload),
                (self.historical_payload, run_manifest_from_payload),
                (self.paper_payload, run_manifest_from_payload),
            ):
                payload = deepcopy(source)
                payload[field] = "forbidden"
                with self.subTest(field=field, schema=source["schema_version"]):
                    with self.assertRaises((TypeError, ValueError)):
                        parser(payload)

    def test_order_is_validated_and_never_repaired(self) -> None:
        for field in ("contracts", "shared_modules"):
            payload = deepcopy(self.candidate_payload)
            payload[field].reverse()
            original_order = deepcopy(payload[field])
            with self.subTest(field=field):
                with self.assertRaises((TypeError, ValueError)):
                    candidate_manifest_from_payload(payload)
                self.assertEqual(payload[field], original_order)

        payload = deepcopy(self.candidate_payload)
        extra = deepcopy(payload["shared_modules"][0])
        extra["logical_name"] = "binance_spot_strategy.synthetic.alpha"
        payload["shared_modules"].insert(0, extra)
        candidate_manifest_from_payload(payload)
        payload["shared_modules"][0], payload["shared_modules"][1] = (
            payload["shared_modules"][1],
            payload["shared_modules"][0],
        )
        with self.assertRaises((TypeError, ValueError)):
            candidate_manifest_from_payload(payload)

    def test_shared_roles_may_repeat_but_each_role_is_required_and_names_are_unique(self) -> None:
        payload = deepcopy(self.candidate_payload)
        extra = deepcopy(payload["shared_modules"][0])
        extra["logical_name"] = "binance_spot_strategy.synthetic.alpha"
        payload["shared_modules"].insert(0, extra)
        candidate = candidate_manifest_from_payload(payload)
        self.assertEqual(len(candidate.shared_modules), 8)

        invalid_payloads = []
        missing_role = deepcopy(payload)
        missing_role["shared_modules"] = [
            item for item in missing_role["shared_modules"] if item["role"] != "metric"
        ]
        invalid_payloads.append(missing_role)
        duplicate = deepcopy(payload)
        duplicate["shared_modules"][1]["logical_name"] = duplicate["shared_modules"][0]["logical_name"]
        invalid_payloads.append(duplicate)
        non_nfc = deepcopy(payload)
        non_nfc["shared_modules"][0]["logical_name"] = "synthetic.e\u0301"
        invalid_payloads.append(non_nfc)
        for invalid in invalid_payloads:
            with self.assertRaises((TypeError, ValueError)):
                candidate_manifest_from_payload(invalid)

    def test_schema_and_input_discriminator_are_bound_as_exact_pairs(self) -> None:
        mutations = []
        for schema, kind in (
            ("historical_run_manifest_v1", "paper"),
            ("paper_run_manifest_v1", "historical"),
            ("unknown", "historical"),
            ("historical_run_manifest_v1", "unknown"),
        ):
            payload = deepcopy(self.historical_payload)
            payload["schema_version"] = schema
            payload["input"]["kind"] = kind
            mutations.append(payload)
        for payload in mutations:
            with self.assertRaises((TypeError, ValueError)):
                run_manifest_from_payload(payload)

    def test_parsers_reject_wrong_container_and_scalar_types(self) -> None:
        mutations = []
        payload = deepcopy(self.candidate_payload)
        payload["contracts"] = tuple(payload["contracts"])
        mutations.append((candidate_manifest_from_payload, payload))
        payload = deepcopy(self.candidate_payload)
        payload["baseline"]["id"] = StringSubclass("baseline_a")
        mutations.append((candidate_manifest_from_payload, payload))
        payload = deepcopy(self.candidate_payload)
        payload["protocols"]["canonical_protocol_sha256"] = "F" * 64
        mutations.append((candidate_manifest_from_payload, payload))
        payload = deepcopy(self.historical_payload)
        payload["source_commit"] = "A" * 40
        mutations.append((run_manifest_from_payload, payload))
        payload = deepcopy(self.historical_payload)
        payload["broker_adapter"] = []
        mutations.append((run_manifest_from_payload, payload))
        for parser, payload in mutations:
            with self.assertRaises((TypeError, ValueError)):
                parser(payload)

    def test_frozen_direct_construction_and_fingerprint_type_boundaries(self) -> None:
        candidate = candidate_manifest_from_payload(self.candidate_payload)
        historical = run_manifest_from_payload(self.historical_payload)
        with self.assertRaises(FrozenInstanceError):
            candidate.baseline_id = "baseline_b"
        self.assertIsInstance(candidate.contracts, tuple)
        self.assertIsInstance(candidate.shared_modules, tuple)
        self.assertTrue(all(type(item) is ContractDigest for item in candidate.contracts))
        self.assertTrue(all(type(item) is ModuleDigest for item in candidate.shared_modules))

        direct_fields = dict(
            schema_version=candidate.schema_version,
            numeric_protocol_version=candidate.numeric_protocol_version,
            design_revision_sha256=candidate.design_revision_sha256,
            baseline_id=candidate.baseline_id,
            baseline_semantic_version=candidate.baseline_semantic_version,
            universe=candidate.universe,
            bar_interval=candidate.bar_interval,
            decision_timing=candidate.decision_timing,
            formal_starting_balance=candidate.formal_starting_balance,
            contracts=candidate.contracts,
            baseline_strategy_module=candidate.baseline_strategy_module,
            shared_modules=candidate.shared_modules,
            canonical_json_version=candidate.canonical_json_version,
            canonical_protocol_sha256=candidate.canonical_protocol_sha256,
            numeric_protocol_sha256=candidate.numeric_protocol_sha256,
            golden_vectors_sha256=candidate.golden_vectors_sha256,
            semantic_dependency_lock_sha256=candidate.semantic_dependency_lock_sha256,
        )
        bad_fields = deepcopy(direct_fields)
        bad_fields["formal_starting_balance"] = Q18(Decimal("501.000000000000000000"))
        with self.assertRaises((TypeError, ValueError)):
            CandidateManifestV1(**bad_fields)
        bad_fields = deepcopy(direct_fields)
        bad_fields["contracts"] = list(candidate.contracts)
        with self.assertRaises((TypeError, ValueError)):
            CandidateManifestV1(**bad_fields)

        subclass = ManifestSubclass(**direct_fields)
        with self.assertRaises(TypeError):
            candidate_strategy_fingerprint(subclass)
        with self.assertRaises(TypeError):
            candidate_strategy_fingerprint(object())
        with self.assertRaises(TypeError):
            run_fingerprint(candidate)
        self.assertIsInstance(historical, RunCommonV1)

    def test_to_payload_is_exact_and_does_not_leak_metadata(self) -> None:
        candidate = candidate_manifest_from_payload(self.candidate_payload)
        historical = run_manifest_from_payload(self.historical_payload)
        paper = run_manifest_from_payload(self.paper_payload)
        self.assertEqual(candidate.to_payload(), self.candidate_payload)
        self.assertEqual(historical.to_payload(), self.historical_payload)
        self.assertEqual(paper.to_payload(), self.paper_payload)
        for manifest in (candidate, historical, paper):
            text = json.dumps(manifest.to_payload(), sort_keys=True)
            for forbidden in (
                "run_id",
                "database_id",
                "activation",
                "audit",
                "result",
                "timestamp",
                "renderer",
            ):
                self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
