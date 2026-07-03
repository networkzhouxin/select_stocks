import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from itick_tools import build_kline_request, load_itick_token


DEFAULT_CODES = ["510300", "159915", "513100"]


def fetch_json(url, headers, timeout=20):
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return json.loads(resp.read().decode(charset))


def summarize_payload(payload):
    if isinstance(payload, dict):
        data = payload.get("data")
        code = payload.get("code")
        msg = payload.get("msg") or payload.get("message")
        rows = len(data) if isinstance(data, list) else 0
        first = data[0] if rows else None
        last = data[-1] if rows else None
        return {"code": code, "msg": msg, "rows": rows, "first": first, "last": last}
    return {"code": None, "msg": "non-dict response", "rows": 0, "first": None, "last": None}


def probe_one(token, endpoint, region, code, k_type, limit, et=None):
    url, headers = build_kline_request(
        token=token,
        region=region,
        code=code,
        k_type=k_type,
        limit=limit,
        endpoint=endpoint,
        et=et,
    )
    try:
        payload = fetch_json(url, headers)
        summary = summarize_payload(payload)
        ok = summary["rows"] > 0
        return ok, summary, None
    except HTTPError as exc:
        return False, None, f"HTTP {exc.code}"
    except URLError as exc:
        return False, None, f"URL error: {exc.reason}"
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Probe iTick fund/ETF kline coverage without printing the token."
    )
    parser.add_argument("--env", default=str(Path(__file__).with_name(".env.local")))
    parser.add_argument("--endpoint", default="/fund/kline")
    parser.add_argument("--regions", default="CN,HK,US")
    parser.add_argument("--codes", default=",".join(DEFAULT_CODES))
    parser.add_argument("--k-type", type=int, default=2, help="iTick kType, e.g. 1/2/3/4/5")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--et", default=None, help="Optional end timestamp for historical paging.")
    args = parser.parse_args(argv)

    token = load_itick_token(args.env)
    regions = [x.strip() for x in args.regions.split(",") if x.strip()]
    codes = [x.strip() for x in args.codes.split(",") if x.strip()]

    any_ok = False
    print("Token loaded: yes")
    print(f"Endpoint: {args.endpoint}")
    print(f"kType={args.k_type} limit={args.limit} et={args.et or ''}")
    for region in regions:
        for code in codes:
            ok, summary, error = probe_one(
                token, args.endpoint, region, code, args.k_type, args.limit, args.et
            )
            any_ok = any_ok or ok
            label = f"{region}:{code}"
            if error:
                print(f"{label} -> ERROR {error}")
            else:
                print(
                    f"{label} -> rows={summary['rows']} "
                    f"code={summary['code']} msg={summary['msg']}"
                )
                if summary["first"] is not None:
                    print(f"  first={summary['first']}")
                    print(f"  last={summary['last']}")

    return 0 if any_ok else 2


if __name__ == "__main__":
    sys.exit(main())
