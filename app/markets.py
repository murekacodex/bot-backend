from app.models import Market


MARKETS: dict[str, Market] = {
    "EURUSD": Market(code="EURUSD", symbol="EURUSD=X", name="Euro / US Dollar", category="forex"),
    "GBPUSD": Market(code="GBPUSD", symbol="GBPUSD=X", name="British Pound / US Dollar", category="forex"),
    "USDJPY": Market(code="USDJPY", symbol="JPY=X", name="US Dollar / Japanese Yen", category="forex"),
    "USDCHF": Market(code="USDCHF", symbol="CHF=X", name="US Dollar / Swiss Franc", category="forex"),
    "AUDUSD": Market(code="AUDUSD", symbol="AUDUSD=X", name="Australian Dollar / US Dollar", category="forex"),
    "USDCAD": Market(code="USDCAD", symbol="CAD=X", name="US Dollar / Canadian Dollar", category="forex"),
    "NZDUSD": Market(code="NZDUSD", symbol="NZDUSD=X", name="New Zealand Dollar / US Dollar", category="forex"),
    "EURGBP": Market(code="EURGBP", symbol="EURGBP=X", name="Euro / British Pound", category="forex"),
    "XAUUSD": Market(code="XAUUSD", symbol="GC=F", name="Gold Futures / US Dollar", category="metal"),
}


def get_market(code: str) -> Market:
    normalized = code.upper().replace("/", "")
    if normalized not in MARKETS:
        supported = ", ".join(MARKETS)
        raise KeyError(f"Unsupported market '{code}'. Supported markets: {supported}")
    return MARKETS[normalized]
