from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ITICK_DIR = ROOT / "local_data" / "itick"
if str(ITICK_DIR) not in sys.path:
    sys.path.insert(0, str(ITICK_DIR))


def test_load_env_file_reads_local_token_without_quotes(tmp_path):
    from itick_tools import load_env_file

    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "# local secret",
                "ITICK_TOKEN = \"abc.def.123\"",
                "OTHER=value",
            ]
        ),
        encoding="utf-8",
    )

    values = load_env_file(env_file)

    assert values["ITICK_TOKEN"] == "abc.def.123"
    assert values["OTHER"] == "value"


def test_build_kline_request_keeps_token_out_of_url():
    from itick_tools import build_kline_request

    url, headers = build_kline_request(
        token="secret-token",
        region="CN",
        code="510300",
        k_type=2,
        limit=10,
    )

    assert "secret-token" not in url
    assert headers["token"] == "secret-token"
    assert "region=CN" in url
    assert "code=510300" in url
    assert "kType=2" in url
    assert "limit=10" in url


if __name__ == "__main__":
    for test in [
        test_load_env_file_reads_local_token_without_quotes,
        test_build_kline_request_keeps_token_out_of_url,
    ]:
        if test.__name__.startswith("test_load"):
            import tempfile

            with tempfile.TemporaryDirectory() as d:
                test(Path(d))
        else:
            test()
