from app.models import Market


MARKETS: dict[str, Market] = {
    "EURUSD": Market(
        code="EURUSD",
        symbol="EURUSD=X",
        name="Euro / US Dollar",
        category="forex",
        session="forex",
        preferred_sessions=["london", "new_york"],
    ),
    "GBPUSD": Market(
        code="GBPUSD",
        symbol="GBPUSD=X",
        name="British Pound / US Dollar",
        category="forex",
        session="forex",
        preferred_sessions=["london", "new_york"],
    ),
    "USDJPY": Market(
        code="USDJPY",
        symbol="JPY=X",
        name="US Dollar / Japanese Yen",
        category="forex",
        session="forex",
        preferred_sessions=["asia", "london"],
    ),
    "USDCHF": Market(
        code="USDCHF",
        symbol="CHF=X",
        name="US Dollar / Swiss Franc",
        category="forex",
        session="forex",
        preferred_sessions=["london", "new_york"],
    ),
    "AUDUSD": Market(
        code="AUDUSD",
        symbol="AUDUSD=X",
        name="Australian Dollar / US Dollar",
        category="forex",
        session="forex",
        preferred_sessions=["asia", "london"],
    ),
    "USDCAD": Market(
        code="USDCAD",
        symbol="CAD=X",
        name="US Dollar / Canadian Dollar",
        category="forex",
        session="forex",
        preferred_sessions=["new_york", "london"],
    ),
    "NZDUSD": Market(
        code="NZDUSD",
        symbol="NZDUSD=X",
        name="New Zealand Dollar / US Dollar",
        category="forex",
        session="forex",
        preferred_sessions=["asia", "london"],
    ),
    "EURGBP": Market(
        code="EURGBP",
        symbol="EURGBP=X",
        name="Euro / British Pound",
        category="forex",
        session="forex",
        preferred_sessions=["london"],
    ),
    "XAUUSD": Market(
        code="XAUUSD",
        symbol="GC=F",
        name="Gold Futures / US Dollar",
        category="metal",
        session="cme_globex",
        preferred_sessions=["london", "new_york"],
    ),
}


def get_market(code: str) -> Market:
    normalized = code.upper().replace("/", "")
    if normalized not in MARKETS:
        supported = ", ".join(MARKETS)
        raise KeyError(f"Unsupported market '{code}'. Supported markets: {supported}")
    return MARKETS[normalized]
