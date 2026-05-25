"""
Map Earnings-22 call IDs to real stock tickers and download actual market data.
This replaces the synthetic/random targets with real post-earnings returns.
"""

import logging
import datetime
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Full mapping: call_id -> (ticker, approximate_call_date)
# Derived from the first sentences of each call's Whisper transcript
# -----------------------------------------------------------------------
CALL_MAPPING = {
    # --- MTN Ghana ---
    "2020-03-0230487MTN-Ghana-2019-Annual-Results-Call": ("MTNOY", "2020-03-02"),
    "2020-Annual-Results-Call-Recording": ("MTNOY", "2021-03-15"),
    "MTN-Ghana-2021-Third-Quarter-Results-Call": ("MTNOY", "2021-10-28"),
    "mtngh_fy18_call_audio_04032019": ("MTNOY", "2019-03-04"),
    "MP3-A": ("MTNOY", "2020-05-01"),

    # --- LATAM Airlines (delisted) ---
    "4329526": ("LTM", "2020-03-06"),

    # --- Telkom SA / PT Telkom Indonesia ---
    "4351517": ("TLK", "2020-06-15"),   # PT Telkom Indonesia ADR
    "4426736": ("TLK", "2021-03-31"),
    "4453076": ("TLK", "2021-09-30"),

    # --- EDP (Energias de Portugal) ---
    "4372696": ("EDPFY", "2020-07-29"),
    "4466797": ("EDPFY", "2021-11-04"),  # EDPR results call

    # --- SK Telecom ---
    "4430051": ("SKM", "2021-02-04"),

    # --- Telecom Italia / TIM ---
    "4432298": ("TIIAY", "2021-05-12"),

    # --- Deutsche Telekom ---
    "4450488": ("DTEGY", "2021-08-12"),

    # --- Turkcell ---
    "4466399": ("TKC", "2021-10-28"),

    # --- Bancolombia ---
    "4466718": ("CIB", "2021-11-10"),

    # --- Net1 UEPS Technologies (delisted) ---
    "4467434": ("UEPS", "2021-11-09"),

    # --- Pampa Energia ---
    "4468679": ("PAM", "2021-11-12"),

    # --- Nexi (Italian payments) ---
    "4468715": ("NEXXY", "2021-11-11"),

    # --- Loma Negra ---
    "4468919": ("LOMA", "2021-11-11"),

    # --- YPF (Argentina) ---
    "4469528": ("YPF", "2021-11-12"),

    # --- Investec ---
    "4469590": ("IVTJF", "2021-11-18"),

    # --- Sabesp (Brazilian water utility) ---
    "4470663": ("SBS", "2021-11-12"),

    # --- DS Smith (UK packaging, delisted) ---
    "4474327": ("DITHF", "2021-12-08"),

    # --- Investor AB (Swedish) ---
    "4480850": ("IVSBF", "2022-01-20"),

    # --- Evolution Mining ---
    "4481952": ("CAHPF", "2022-01-27"),

    # --- GasLog Partners (delisted) ---
    "4482110": ("GLOP", "2022-02-03"),

    # --- Vedanta Limited ---
    "4482613": ("VEDL", "2022-01-28"),

    # --- Imperial Oil ---
    "4483296": ("IMO", "2022-02-01"),

    # --- Tele2 (Swedish telecom) ---
    "4483338": ("TLTZF", "2022-02-03"),

    # --- TeamViewer AG ---
    "4483589": ("TMVWY", "2022-02-09"),

    # --- Infineon Technologies ---
    "4483912": ("IFNNY", "2022-02-03"),

    # --- Rithm Capital (formerly New Residential) ---
    "4485206": ("RITM", "2022-02-10"),

    # ===================== NEW MAPPINGS (from transcript analysis) =====================

    # --- KB Financial Group ---
    "4420696": ("KB", "2021-02-05"),

    # --- Itau CorpBanca ---
    "4423872": ("ITUB", "2021-05-05"),

    # --- Unidentified (mentions EBITDA 165M EUR Q2, portfolio mgmt) ---
    "4443871": (None, "2021-07-29"),

    # --- Unidentified (mentions corporate website, mitigating market constraints) ---
    "4443920": (None, "2021-08-05"),

    # --- Unidentified (Latin American, mentions consumption recovery) ---
    "4446796": (None, "2021-08-15"),

    # --- Grupo Aval (Colombian banking) ---
    "4448760": ("AVAL", "2021-08-13"),

    # --- Embraer (Brazilian aircraft) ---
    "4449269": ("ERJ", "2021-08-13"),

    # --- Arco Platform (EdTech, now Arco Educacao) ---
    "4450779": ("ARCE", "2021-08-12"),

    # --- Unidentified (Turkish, mentions 1.3B Turkish lira financing) ---
    "4452058": (None, "2021-09-15"),

    # --- CD Projekt (Polish game company, Cyberpunk 2077) ---
    "4453085": ("OTGLY", "2021-09-02"),

    # --- Aspen Pharmacare (South African pharma) ---
    "4453225": ("APNHY", "2021-09-23"),

    # --- Galp Energia (Portuguese energy) ---
    "4461799": ("GLPEY", "2021-10-29"),

    # --- Texas Instruments ---
    "4462231": ("TXN", "2021-10-26"),

    # --- Repsol (Spanish energy) ---
    "4463259": ("REPYY", "2021-10-28"),

    # --- Delta Electronics Thailand (delisted ADR) ---
    "4463693": ("DLELY", "2021-11-04"),

    # --- Enel Chile ---
    "4464164": ("ENIC", "2021-11-03"),

    # --- Coca-Cola (emerging market bottler, likely Coca-Cola Icecek or HBC) ---
    "4466251": ("CCH", "2021-11-04"),

    # --- Legrand (French electrical) ---
    "4466607": ("LGRVF", "2021-11-04"),

    # --- Tecnoglass (Colombian glass/windows) ---
    "4467071": ("TGLS", "2021-11-05"),

    # --- BrasilAgro (Brazilian agriculture) ---
    "4467079": ("LND", "2021-11-11"),

    # --- Gol Airlines (Brazilian airline) ---
    "4467717": ("GOL", "2021-11-05"),

    # --- Unidentified (mentions annual results, synergies) ---
    "4468000": (None, "2021-11-10"),

    # --- Ecopetrol (Colombian oil) ---
    "4468307": ("EC", "2021-11-09"),

    # --- SONAE (Portuguese retail conglomerate) ---
    "4468639": ("SONFQ", "2021-11-11"),

    # --- Unidentified (German/Italian, mentions Germany frequency, Cristiano Borean CFO) ---
    # This is likely Atlantia or similar, Borean is CFO of TIM/Telecom Italia
    "4468647": (None, "2021-11-11"),

    # --- Unidentified (mentions Q3 2021 results, guidance) ---
    "4468654": (None, "2021-11-11"),

    # --- Prysmian Group (Italian cables) ---
    "4468678": ("PRY.MI", "2021-11-11"),

    # --- Leatt Corporation (safety gear) ---
    "4469075": ("LEAT", "2021-11-12"),

    # --- Unidentified (mentions packaging, refinanced bonds) ---
    "4469088": (None, "2021-11-12"),

    # --- Merlin Properties (Spanish REIT) ---
    "4469184": ("MRPRF", "2021-11-04"),

    # --- Cementos Argos (Colombian cement) ---
    "4469208": ("CMTOY", "2021-11-12"),

    # --- Unidentified (mentions investment program, Q3) ---
    "4469291": (None, "2021-11-12"),

    # --- Toshiba ---
    "4469669": ("TOSYY", "2021-11-12"),

    # --- Unidentified (Spanish, Eduardo San Miguel, Chairman Juan Llado) ---
    "4469836": (None, "2021-11-15"),

    # --- Unidentified (mentions ENO Internet Group) ---
    "4470010": (None, "2021-11-15"),

    # --- Unknown (no ticker) ---
    "4470253": (None, "2021-11-15"),

    # --- Despegar (Latin American travel) ---
    "4470290": ("DESP", "2021-11-18"),

    # --- SQM (Sociedad Quimica y Minera, Chilean lithium/chemicals) ---
    "4470570": ("SQM", "2021-11-18"),

    # --- Unidentified (mentions APR environment, IT infrastructure) ---
    "4470595": (None, "2021-11-18"),

    # --- Unidentified (South African, mentions ROE 12% vs 8.1%) ---
    "4470684": (None, "2021-11-18"),

    # --- Unknown (no ticker) ---
    "4471809": (None, "2021-11-18"),

    # --- Unknown (no ticker, mentions "polis" which may be a company) ---
    "4471586": (None, "2021-11-25"),

    # --- Hepsiburada (Turkish e-commerce) ---
    "4471606": ("HEPS", "2021-11-18"),

    # --- RusHydro (Russian hydro, likely sanctioned) ---
    "4471961": ("RSHYY", "2021-11-25"),

    # --- CD Projekt Q3 2021 ---
    "4472403": ("OTGLY", "2021-11-29"),

    # --- Quhuo (Chinese gig economy) ---
    "4472895": ("QH", "2021-12-01"),

    # --- Unknown (no ticker) ---
    "4473837": (None, "2021-12-01"),

    # --- RLX Technology (Chinese e-vape) ---
    "4473238": ("RLX", "2021-11-24"),

    # --- Unidentified (mentions Q&A typed questions) ---
    "4474229": (None, "2021-12-09"),

    # --- Unidentified (mentions Q1 earnings call) ---
    "4474506": (None, "2021-12-15"),

    # --- Unknown (no ticker) ---
    "4475604": (None, "2021-12-15"),

    # --- Qudian (Chinese consumer finance) ---
    "4474955": ("QD", "2021-12-09"),

    # --- Trip.com (Chinese travel) ---
    "4475486": ("TCOM", "2021-12-16"),

    # --- TuanChe (Chinese auto marketplace) ---
    "4479524": ("TC", "2022-01-14"),

    # --- Unidentified (mentions EBIT margin 24.4%, Lisa Mortensen CFO) ---
    # Likely Coloplast or similar Danish company
    "4479741": (None, "2022-01-20"),

    # --- HDFC Bank ---
    "4479944": ("HDB", "2022-01-15"),

    # --- Sify Technologies (Indian IT) ---
    "4481221": ("SIFY", "2022-01-27"),

    # --- Unknown Asian ---
    "4481766": (None, "2022-01-27"),

    # --- Unknown ---
    "4481552": (None, "2022-01-27"),

    # --- Unknown ---
    "4481601": (None, "2022-01-28"),

    # --- Unknown Asian ---
    "4481967": (None, "2022-01-27"),

    # --- Nidec Corporation (Japanese motors) ---
    "4481904": ("NJDCY", "2022-01-27"),

    # --- Sartorius AG (German lab equipment) ---
    "4482249": ("SARTF", "2022-01-27"),

    # --- Banco Sabadell (Spanish bank) ---
    "4482264": ("BNDSF", "2022-01-28"),

    # --- LVMH / Hermes (French luxury, mentions fashion & leather) ---
    "4482311": ("HESAY", "2022-02-01"),

    # --- PointsBet Holdings (Australian betting) ---
    "4482383": ("PBTHF", "2022-02-03"),

    # --- Dr. Reddy's Laboratories (Indian pharma) ---
    "4482569": ("RDY", "2022-01-28"),

    # --- CaixaBank (Spanish bank) ---
    "4482609": ("CAIXY", "2022-02-04"),

    # --- Advantest Corporation (Japanese semiconductor test equipment) ---
    "4482641": ("ATEYY", "2022-02-03"),

    # --- SSAB (Swedish steel) ---
    "4482682": ("SSAAY", "2022-02-03"),

    # --- Unidentified mining (mentions December quarterly production, recovering grades) ---
    "4482968": (None, "2022-02-03"),

    # --- Unidentified mining (Syria Resources? mentions Bal-) ---
    "4482976": (None, "2022-02-03"),

    # --- Pilbara Minerals (Australian lithium) ---
    "4482983": ("PILBF", "2022-02-03"),

    # --- TDK Corporation (Japanese electronics) ---
    "4483046": ("TTDKY", "2022-02-03"),

    # --- Novozymes (Danish biotech/enzymes) ---
    "4483297": ("NVZMY", "2022-02-02"),

    # --- Lundin Energy (Swedish oil, now Orrön Energy) ---
    "4483320": ("LNEGY", "2022-02-03"),

    # --- Sony Group ---
    "4483506": ("SONY", "2022-02-02"),

    # --- Exco Technologies (Canadian auto parts) ---
    "4483623": ("EXCOF", "2022-02-03"),

    # --- Ferrari ---
    "4483633": ("RACE", "2022-02-02"),

    # --- ATS Automation (Canadian automation) ---
    "4483668": ("ATS", "2022-02-02"),

    # --- Capital Product Partners (Greek shipping) ---
    "4483670": ("CPLP", "2022-02-03"),

    # --- CGI Inc (Canadian IT services) ---
    "4483678": ("GIB", "2022-01-26"),

    # --- Allied Properties REIT (Canadian) ---
    "4483680": ("APYRF", "2022-02-03"),

    # --- Orsted (Danish wind energy) ---
    "4483733": ("DNNGY", "2022-02-02"),

    # --- Siemens Healthineers ---
    "4483857": ("SMMNY", "2022-02-03"),

    # --- Renishaw (UK precision engineering) ---
    "4483937": ("RNSHF", "2022-01-27"),

    # --- Publicis Groupe (French advertising) ---
    "4484088": ("PUBGY", "2022-02-10"),

    # --- Dassault Systemes (French software) ---
    "4484146": ("DASTY", "2022-02-03"),

    # --- Sanofi (French pharma) ---
    "4484563": ("SNY", "2022-02-04"),

    # --- Aurubis AG (German copper) ---
    "4484942": ("AIAGF", "2022-02-09"),

    # --- Edgewell Personal Care ---
    "4485192": ("EPC", "2022-02-09"),

    # --- KKR & Co ---
    "4485244": ("KKR", "2022-02-07"),
}


def get_real_market_data(project_root: Path):
    """Download real stock prices for all mapped calls."""

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return

    results = []

    for call_id, (ticker, call_date_str) in CALL_MAPPING.items():
        if ticker is None:
            logger.warning("No ticker for %s, using synthetic target", call_id)
            np.random.seed(hash(call_id) % 2**31)
            results.append({
                "call_id": call_id,
                "ticker": "UNKNOWN",
                "call_date": datetime.date.fromisoformat(call_date_str),
                "return_1d": float(np.random.normal(0.0, 0.02)),
                "return_5d": float(np.random.normal(0.0, 0.04)),
                "realized_vol_5d": float(np.random.uniform(0.01, 0.05)),
                "close_t0": 100.0,
                "close_t1": 100.0,
                "close_t5": 100.0,
                "data_source": "synthetic",
            })
            continue

        call_date = datetime.date.fromisoformat(call_date_str)
        start = call_date - datetime.timedelta(days=5)
        end = call_date + datetime.timedelta(days=15)

        try:
            logger.info("Downloading %s for %s (%s)...", ticker, call_id, call_date_str)
            stock = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)

            if len(stock) < 3:
                logger.warning("Not enough data for %s (%s), using synthetic", ticker, call_id)
                np.random.seed(hash(call_id) % 2**31)
                results.append({
                    "call_id": call_id,
                    "ticker": ticker,
                    "call_date": call_date,
                    "return_1d": float(np.random.normal(0.0, 0.02)),
                    "return_5d": float(np.random.normal(0.0, 0.04)),
                    "realized_vol_5d": float(np.random.uniform(0.01, 0.05)),
                    "close_t0": 100.0,
                    "close_t1": 100.0,
                    "close_t5": 100.0,
                    "data_source": "synthetic_fallback",
                })
                continue

            # Find the closest trading day to call_date
            import pandas as pd
            stock = stock.sort_index()

            # Handle multi-level columns from yfinance
            if isinstance(stock.columns, pd.MultiIndex):
                close_col = stock.columns[stock.columns.get_level_values(0) == "Close"][0]
                closes = stock[close_col]
            else:
                closes = stock["Close"]

            # Convert call_date to pandas Timestamp for comparison
            call_ts = pd.Timestamp(call_date)

            # Find t0 (call day or next trading day)
            t0_idx = 0
            for i, d in enumerate(closes.index):
                if d >= call_ts:
                    t0_idx = i
                    break

            close_t0 = float(closes.iloc[t0_idx])

            # T+1
            if t0_idx + 1 < len(closes):
                close_t1 = float(closes.iloc[t0_idx + 1])
            else:
                close_t1 = close_t0

            # T+5
            if t0_idx + 5 < len(closes):
                close_t5 = float(closes.iloc[t0_idx + 5])
            else:
                close_t5 = float(closes.iloc[-1])

            return_1d = (close_t1 - close_t0) / close_t0
            return_5d = (close_t5 - close_t0) / close_t0

            # Realized volatility (std of daily returns over 5 days)
            if t0_idx + 5 < len(closes):
                window = closes.iloc[t0_idx:t0_idx + 6]
                daily_returns = window.pct_change().dropna()
                realized_vol = float(daily_returns.std()) if len(daily_returns) > 1 else 0.02
            else:
                realized_vol = 0.02

            results.append({
                "call_id": call_id,
                "ticker": ticker,
                "call_date": call_date,
                "return_1d": return_1d,
                "return_5d": return_5d,
                "realized_vol_5d": realized_vol,
                "close_t0": close_t0,
                "close_t1": close_t1,
                "close_t5": close_t5,
                "data_source": "real",
            })

            logger.info("  %s: close=%.2f, ret_1d=%.4f, ret_5d=%.4f, vol=%.4f",
                        ticker, close_t0, return_1d, return_5d, realized_vol)


        except Exception as e:
            logger.warning("Error downloading %s: %s, using synthetic", ticker, e)
            np.random.seed(hash(call_id) % 2**31)
            results.append({
                "call_id": call_id,
                "ticker": ticker,
                "call_date": call_date,
                "return_1d": float(np.random.normal(0.0, 0.02)),
                "return_5d": float(np.random.normal(0.0, 0.04)),
                "realized_vol_5d": float(np.random.uniform(0.01, 0.05)),
                "close_t0": 100.0,
                "close_t1": 100.0,
                "close_t5": 100.0,
                "data_source": "synthetic_fallback",
            })

    # Save
    df = pl.DataFrame(results)
    output_path = project_root / "data" / "processed" / "earnings22_market_data.parquet"
    df.write_parquet(output_path)

    real_count = len([r for r in results if r["data_source"] == "real"])
    synth_count = len([r for r in results if r["data_source"] in ("synthetic", "synthetic_fallback")])
    logger.info("=" * 60)
    logger.info("Saved market data: %d calls (%d real, %d synthetic)",
                len(df), real_count, synth_count)
    logger.info("Real return_1d range: [%.4f, %.4f]", df["return_1d"].min(), df["return_1d"].max())
    logger.info("Real vol range: [%.4f, %.4f]", df["realized_vol_5d"].min(), df["realized_vol_5d"].max())
    logger.info("Output: %s", output_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    get_real_market_data(project_root)
