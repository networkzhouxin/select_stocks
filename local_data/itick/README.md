# iTick Local Data Probe

This folder is for local iTick data experiments only. It is separate from the strategy code so API tokens and downloaded market data do not mix with production strategy files.

## Setup

Create `local_data/itick/.env.local`:

```text
ITICK_TOKEN=your_token_here
```

`.env.local` is ignored by git. Do not paste the token into source files or logs.

## Probe ETF K-Line Coverage

Run:

```powershell
python local_data\itick\download_itick_sample.py --regions CN,HK,US --codes 510300,159915,513100 --k-type 2 --limit 10
```

The script prints only whether the token was loaded, not the token value. `kType` follows iTick's fund K-line API. We still need live probing to confirm whether Chinese exchange-traded ETFs such as `510300` are available under the fund endpoint or another endpoint.
