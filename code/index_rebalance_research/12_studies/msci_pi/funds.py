"""Which s12 funds count as visible MSCI trackers (rules used by s03_pi.py).

A fund counts for a segment (Standard or Small Cap) of a market (Canada, or one DM-Europe country) in a quarter if
  1. its NAME says it tracks an MSCI index (MSCI | EAFE | KOKUSAI | ACWI, all MSCI index names), or it is on the
     WHITELIST below (index funds whose benchmark is an MSCI index but whose name does not say so; benchmark from the
     fund's public documents as we know them, tag I), and the name has none of the EXCLUDE words (screened, ESG,
     factor, style, sector, leveraged, synthetic-looking); and it is not on the BLACKLIST;
  2. its holdings pass the TRACKER TEST against the segment's ruler fund (s12, same quarter): it holds >= COV_MIN of
     the ruler's weight in that market/country, and the inter-quartile range of log(fund shares / ruler shares)
     across those names is <= IQR_MAX (cap-weighted, proportional holding).
Look-alikes that pass the holdings test but track FTSE / S&P / STOXX / Morningstar / Solactive / CRSP indices
(Vanguard US and UK funds, Schwab International Equity ETF, JPMorgan BetaBuilders, iShares S&P/TSX, BlackRock UK
ACS 50:50 trackers, UK insured pension funds) fail rule 1. Vanguard Australia's International Shares Index funds DO
track MSCI World ex-Australia and are whitelisted. Vanguard VEA / VXUS / VGK (FTSE) are excluded by name.
"""
import re

TOKEN = re.compile(r"MSCI|EAFE|KOKUSAI|ACWI")
EXCLUDE = re.compile(r"ESG|SRI\b|SCREEN|CLIMATE|PARIS|CARBON|FOSSIL|MIN VOL|MINIMUM|QUALITY|MOMENTUM|FACTOR|DIVIDEND|"
                     r"SELECT|SELECTION|LEADERS|UNIVERSAL|ISLAMIC|EQUAL|VALUE|GROWTH|ETHIC|SUSTAIN|CONTROVERS|"
                     r"STRATEGICF|INCOME|COVERED|BUFFER|ULTRA|SHORT|LEVERAG|2X|3X|ENHANCED|ADAPTIVE|CAPPING|"
                     r"FTSE|S&P|STOXX|SOLACTIVE|MORNINGSTAR|BETABUILDERS|RUSSELL|CRSP|RAFI|DOW JONES|FINANCIAL|"
                     r"HEALTH|ENERGY|CONSUMER|CNSM|CONS |INDUSTR|INFORMATION|INFO|UTILIT|MATERIAL|REAL EST|COMM\b|"
                     r"TECH|BANK|MIDCAP|MID CAP| LC |LARGE C|PRIVATE")
WHITELIST = {   # fundno: (name, benchmark)  -- tag I
    13761: ("Fidelity International Index (FSPSX)", "MSCI EAFE"),
    104402: ("Fidelity SAI International Index", "MSCI EAFE"),
    9753: ("Fidelity Global ex U.S. Index (FSGGX)", "MSCI ACWI ex USA"),
    166392: ("Fidelity Global ex U.S. Index", "MSCI ACWI ex USA"),
    130771: ("Fidelity Total International Index (FTIHX)", "MSCI ACWI ex USA IMI"),
    105988: ("Fidelity Flex International Index", "MSCI ACWI ex USA IMI"),
    165817: ("Fidelity Index World Fund (UK)", "MSCI World"),
    75718: ("TIAA-CREF International Equity Index", "MSCI EAFE"),
    13597: ("BlackRock International Index", "MSCI EAFE"),
    54900: ("Schwab International Index (SWISX)", "MSCI EAFE"),
    108723: ("Northern International Equity Index", "MSCI World ex USA"),
    196659: ("State Street Global All Cap Equity ex-US Index", "MSCI ACWI ex USA IMI"),
    139892: ("State Street Global All Cap Equity ex-US Index", "MSCI ACWI ex USA IMI"),
    113988: ("State Street International Developed Equity Index", "MSCI EAFE"),
    102723: ("State Street Hedged International Developed Equity Index", "MSCI EAFE hedged"),
    168086: ("John Hancock International Equity Index", "MSCI ACWI ex USA"),
    129251: ("LVIP SSgA International Index", "MSCI EAFE"),
    184428: ("JNL International Index", "MSCI EAFE"),
    41732: ("JNL/Mellon International Index", "MSCI EAFE"),
    82943: ("Voya International Index", "MSCI EAFE"),
    69978: ("VALIC Company I International Equities Index", "MSCI EAFE"),
    79928: ("Nationwide NVIT International Index", "MSCI EAFE"),
    64727: ("Nationwide International Index", "MSCI EAFE"),
    115672: ("AZL International Index", "MSCI EAFE"),
    108859: ("Principal International Equity Index", "MSCI EAFE"),
    110614: ("Great-West International Index", "MSCI EAFE"),
    30075: ("TD Emerald International Equity Index", "MSCI EAFE"),
    30496: ("CIBC International Index", "MSCI EAFE"),
    163527: ("iShares Developed World Index Fund (IE)", "MSCI World"),
    172863: ("Vanguard International Shares Index (AU)", "MSCI World ex Australia"),
    106638: ("Vanguard International Shares Index (AU)", "MSCI World ex Australia"),
    114431: ("Vanguard International Small Companies Index (AU)", "MSCI World ex Australia Small Cap"),
    91436: ("Vanguard International Small Companies Index (AU)", "MSCI World ex Australia Small Cap"),
    144364: ("Nissay Foreign Equity Index", "MSCI Kokusai"),
    8304: ("CSIF (CH) Equity Canada Index", "MSCI Canada"),
    103169: ("CSIF (CH) Equity Europe ex CH Index", "MSCI Europe ex Switzerland"),
    138079: ("CSIF (CH) Equity World ex CH Small Cap Index", "MSCI World ex CH Small Cap"),
    110465: ("CSIF (CH) Equity World ex CH Small Cap Index", "MSCI World ex CH Small Cap"),
    136221: ("KLP AksjeGlobal Indeks", "MSCI World"),
    171051: ("KLP AksjeGlobal Small Cap Indeks", "MSCI World Small Cap"),
    82917: ("KLP AksjeEuropa Indeks", "MSCI Europe"),
    158939: ("DNB Global Indeks", "MSCI World"),
}
BLACKLIST = {148406: "Fidelity ZERO International (Fidelity index)", 120491: "TD International Equity Index ETF (Solactive)",
             57776: "T. Rowe Price International Equity Index (FTSE)"}
COV_MIN = {"STD": 0.85, "SC": 0.70}
IQR_MAX = {"STD": 0.30, "SC": 0.40}


def name_ok(fundno: int, name: str) -> bool:
    if fundno in BLACKLIST:
        return False
    if fundno in WHITELIST:
        return True
    n = str(name).upper()
    return bool(TOKEN.search(n)) and not EXCLUDE.search(n)
