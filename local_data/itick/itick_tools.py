from pathlib import Path
from urllib.parse import urlencode


API_BASE = "https://api.itick.org"


def load_env_file(path):
    values = {}
    path = Path(path)
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_itick_token(env_path=None):
    if env_path is None:
        env_path = Path(__file__).with_name(".env.local")
    values = load_env_file(env_path)
    token = values.get("ITICK_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "ITICK_TOKEN is missing. Put it in local_data/itick/.env.local."
        )
    return token


def build_kline_request(
    token,
    region,
    code,
    k_type,
    limit,
    *,
    endpoint="/fund/kline",
    et=None,
):
    params = {
        "region": region,
        "code": code,
        "kType": str(k_type),
        "limit": str(limit),
    }
    if et is not None:
        params["et"] = str(et)

    url = f"{API_BASE}{endpoint}?{urlencode(params)}"
    headers = {
        "accept": "application/json",
        "token": token,
    }
    return url, headers
