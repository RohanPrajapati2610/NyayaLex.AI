"""
india_constitution.py — Constitution of India ingestion for NyayaLex.AI

What it does:
    Downloads the full text of the Constitution of India article by article
    using the Indian Kanoon search API, chunks each article's text, and
    writes the results to data/processed/india_constitution.jsonl.

Data source:
    Indian Kanoon (https://indiankanoon.org/) via its search endpoint.
    Each article is found by searching:
        https://indiankanoon.org/search/?formInput=constitution+of+india+article+{n}&pagenum=0
    Then the top document is fetched from:
        https://indiankanoon.org/doc/{doc_id}/

    The Preamble and 12 Schedules are fetched using named search queries.

How to run:
    python -m src.ingestion.india_constitution

Output files:
    data/raw/india/constitution.txt          — concatenated plain-text of all articles
    data/processed/india_constitution.jsonl  — one JSON object per chunk
"""

import json
import os
import re
import time
from pathlib import Path

import jsonlines
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

from src.ingestion.chunker import chunk_text

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw/india")
PROCESSED_DIR = Path("data/processed")
RAW_FILE = RAW_DIR / "constitution.txt"
OUT_FILE = PROCESSED_DIR / "india_constitution.jsonl"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEARCH_URL = "https://indiankanoon.org/search/"
DOC_URL = "https://indiankanoon.org/doc/{doc_id}/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NyayaLex-Ingestion/1.0; "
        "+https://github.com/rohan/nyayalex)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SLEEP_BETWEEN = 0.5          # seconds between every HTTP request
MAX_RETRIES = 5
BACKOFF_BASE = 2.0           # exponential backoff base (seconds)

# ---------------------------------------------------------------------------
# Constitution article table — (article_num, heading, part_label)
# Articles 1–395 + Preamble + 12 Schedules.
# Part labels from the official numbering of the Constitution of India.
# ---------------------------------------------------------------------------
PART_MAP = {
    range(1, 5):    "Part I — The Union and its Territory",
    range(5, 12):   "Part II — Citizenship",
    range(12, 36):  "Part III — Fundamental Rights",
    range(36, 52):  "Part IV — Directive Principles of State Policy",
    range(52, 53):  "Part IVA — Fundamental Duties",
    range(53, 80):  "Part V — The Union",
    range(80, 123): "Part VI — The States",
    range(123, 151):"Part VII — The States in Part B of the First Schedule (Repealed)",
    range(148, 152):"Part VIII — The Union Territories",
    range(152, 238):"Part VIII — The Union Territories",
    range(238, 242):"Part IX — The Panchayats",
    range(242, 243):"Part IXA — The Municipalities",
    range(243, 245):"Part IXB — Co-operative Societies",
    range(245, 264):"Part XI — Relations between the Union and the States",
    range(264, 301):"Part XII — Finance, Property, Contracts and Suits",
    range(301, 308):"Part XIII — Trade, Commerce and Intercourse within the Territory of India",
    range(308, 324):"Part XIV — Services under the Union and the States",
    range(324, 330):"Part XV — Elections",
    range(330, 343):"Part XVI — Special Provisions relating to certain Classes",
    range(343, 352):"Part XVII — Official Language",
    range(352, 361):"Part XVIII — Emergency Provisions",
    range(361, 368):"Part XIX — Miscellaneous",
    range(368, 369):"Part XX — Amendment of the Constitution",
    range(369, 393):"Part XXI — Temporary, Transitional and Special Provisions",
    range(393, 396):"Part XXII — Short Title, Commencement, Authoritative Text in Hindi and Repeals",
}

def _article_part(article_num: int) -> str:
    for r, label in PART_MAP.items():
        if article_num in r:
            return label
    return "Miscellaneous"


# Article headings for all 395 articles (abbreviated set for the most cited;
# remaining articles fall back to a generic heading derived by search result).
ARTICLE_HEADINGS = {
    "preamble": "Preamble",
    1: "Name and territory of the Union",
    2: "Admission or establishment of new States",
    3: "Formation of new States and alteration of areas, boundaries or names of existing States",
    4: "Laws made under articles 2 and 3 to provide for the amendment of the First and the Fourth Schedules and supplemental, incidental and consequential matters",
    5: "Citizenship at the commencement of the Constitution",
    6: "Rights of citizenship of certain persons who have migrated to India from Pakistan",
    7: "Rights of citizenship of certain migrants to Pakistan",
    8: "Rights of citizenship of certain persons of Indian origin residing outside India",
    9: "Persons voluntarily acquiring citizenship of a foreign State not to be citizens",
    10: "Continuance of the rights of citizenship",
    11: "Parliament to regulate the right of citizenship by law",
    12: "Definition of State",
    13: "Laws inconsistent with or in derogation of the Fundamental Rights",
    14: "Equality before law",
    15: "Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth",
    16: "Equality of opportunity in matters of public employment",
    17: "Abolition of Untouchability",
    18: "Abolition of titles",
    19: "Protection of certain rights regarding freedom of speech, etc.",
    20: "Protection in respect of conviction for offences",
    21: "Protection of life and personal liberty",
    21: "Protection of life and personal liberty",
    22: "Protection against arrest and detention in certain cases",
    23: "Prohibition of traffic in human beings and forced labour",
    24: "Prohibition of employment of children in factories, etc.",
    25: "Freedom of conscience and free profession, practice and propagation of religion",
    26: "Freedom to manage religious affairs",
    27: "Freedom as to payment of taxes for promotion of any particular religion",
    28: "Freedom as to attendance at religious instruction or religious worship in certain educational institutions",
    29: "Protection of interests of minorities",
    30: "Right of minorities to establish and administer educational institutions",
    31: "Compulsory acquisition of property (Repealed)",
    32: "Remedies for enforcement of rights conferred by this Part",
    33: "Power of Parliament to modify the rights conferred by this Part in their application to Forces, etc.",
    34: "Restriction on rights conferred by this Part while martial law is in force in any area",
    35: "Legislation to give effect to the provisions of this Part",
    36: "Definition",
    37: "Application of the provisions contained in this Part",
    38: "State to secure a social order for the promotion of welfare of the people",
    39: "Certain principles of policy to be followed by the State",
    40: "Organisation of village panchayats",
    41: "Right to work, to education and to public assistance in certain cases",
    42: "Provision for just and humane conditions of work and maternity relief",
    43: "Living wage, etc., for workers",
    44: "Uniform civil code for the citizens",
    45: "Provision for early childhood care and education to children below the age of six years",
    46: "Promotion of educational and economic interests of Scheduled Castes, Scheduled Tribes and other weaker sections",
    47: "Duty of the State to raise the level of nutrition and the standard of living and to improve public health",
    48: "Organisation of agriculture and animal husbandry",
    49: "Protection of monuments and places and objects of national importance",
    50: "Separation of judiciary from executive",
    51: "Promotion of international peace and security",
    51: "Promotion of international peace and security",
    52: "The President of India",
    53: "Executive power of the Union",
    54: "Election of President",
    55: "Manner of election of President",
    56: "Term of office of President",
    57: "Eligibility for re-election",
    58: "Qualifications for election as President",
    59: "Conditions of President's office",
    60: "Oath or affirmation by the President",
    61: "Procedure for impeachment of the President",
    62: "Time of holding election to fill vacancy in the office of President and the term of office of person elected to fill casual vacancy",
    63: "The Vice-President of India",
    64: "The Vice-President to be ex officio Chairman of the Council of States",
    65: "The Vice-President to act as President or to discharge his functions during casual vacancies in the office, or during the absence, of President",
    66: "Election of Vice-President",
    67: "Term of office of Vice-President",
    68: "Time of holding election to fill vacancy in the office of Vice-President and the term of office of person elected to fill casual vacancy",
    69: "Oath or affirmation by the Vice-President",
    70: "Discharge of President's functions in other contingencies",
    71: "Matters relating to, or connected with, the election of a President or Vice-President",
    72: "Power of President to grant pardons, etc., and to suspend, remit or commute sentences in certain cases",
    73: "Extent of executive power of the Union",
    74: "Council of Ministers to aid and advise President",
    75: "Other provisions as to Ministers",
    76: "Attorney-General for India",
    77: "Conduct of business of the Government of India",
    78: "Duties of Prime Minister as respects the furnishing of information to the President, etc.",
    79: "Constitution of Parliament",
    80: "Composition of the Council of States",
    81: "Composition of the House of the People",
    82: "Readjustment after each census",
    83: "Duration of Houses of Parliament",
    84: "Qualification for membership of Parliament",
    85: "Sessions of Parliament, prorogation and dissolution",
    86: "Right of President to address and send messages to Houses",
    87: "Special address by the President",
    88: "Rights of Ministers and Attorney-General as respects Houses",
    89: "The Chairman and Deputy Chairman of the Council of States",
    90: "Vacation and resignation of, and removal from, the office of Deputy Chairman",
    91: "Power of the Deputy Chairman or other person to perform the duties of the office of, or to act as, Chairman",
    92: "The Chairman or the Deputy Chairman not to preside while a resolution for his removal from office is under consideration",
    93: "The Speaker and Deputy Speaker of the House of the People",
    94: "Vacation and resignation of, and removal from, the offices of Speaker and Deputy Speaker",
    95: "Power of the Deputy Speaker or other person to perform the duties of the office of, or to act as, Speaker",
    96: "Speaker or the Deputy Speaker not to preside while a resolution for his removal from office is under consideration",
    97: "Salaries and allowances of the Chairman and Deputy Chairman and the Speaker and Deputy Speaker",
    98: "Secretariat of Parliament",
    99: "Oath or affirmation by members",
    100: "Voting in Houses, power of Houses to act notwithstanding vacancies and quorum",
    101: "Vacation of seats",
    102: "Disqualifications for membership",
    103: "Decision on questions as to disqualifications of members",
    104: "Penalty for sitting and voting before making oath or affirmation under article 99 or when not qualified or when disqualified",
    105: "Powers, privileges, etc., of the Houses of Parliament and of the members and committees thereof",
    106: "Salaries and allowances of members",
    107: "Provisions as to introduction and passing of Bills",
    108: "Joint sitting of both Houses in certain cases",
    109: "Special procedure in respect of Money Bills",
    110: "Definition of Money Bills",
    111: "Assent to Bills",
    112: "Annual financial statement",
    113: "Procedure in Parliament with respect to estimates",
    114: "Appropriation Bills",
    115: "Supplemental, additional or excess grants",
    116: "Votes on account, votes of credit and exceptional grants",
    117: "Special provisions as to financial Bills",
    118: "Rules of procedure",
    119: "Regulation by law of procedure in Parliament in relation to financial business",
    120: "Language to be used in Parliament",
    121: "Restriction on discussion in Parliament",
    122: "Courts not to inquire into proceedings of Parliament",
    123: "Power of President to promulgate Ordinances during recess of Parliament",
    124: "Establishment and constitution of Supreme Court",
    125: "Salaries, etc., of Judges",
    126: "Appointment of acting Chief Justice",
    127: "Appointment of ad hoc Judges",
    128: "Attendance of retired Judges at sittings of the Supreme Court",
    129: "Supreme Court to be a court of record",
    130: "Seat of Supreme Court",
    131: "Original jurisdiction of the Supreme Court",
    132: "Appellate jurisdiction of Supreme Court in appeals from High Courts in certain cases",
    133: "Appellate jurisdiction of Supreme Court in appeals from High Courts in regard to civil matters",
    134: "Appellate jurisdiction of Supreme Court in regard to criminal matters",
    135: "Jurisdiction and powers of the Federal Court under existing law to be exercisable by the Supreme Court",
    136: "Special leave to appeal by the Supreme Court",
    137: "Review of judgments or orders by the Supreme Court",
    138: "Enlargement of the jurisdiction of the Supreme Court",
    139: "Conferment on the Supreme Court of powers to issue certain writs",
    140: "Ancillary powers of Supreme Court",
    141: "Law declared by Supreme Court to be binding on all courts",
    142: "Enforcement of decrees and orders of Supreme Court and orders as to discovery, etc.",
    143: "Power of President to consult Supreme Court",
    144: "Civil and judicial authorities to act in aid of the Supreme Court",
    145: "Rules of Court, etc.",
    146: "Officers and servants and the expenses of the Supreme Court",
    147: "Interpretation",
    148: "Comptroller and Auditor-General of India",
    149: "Duties and powers of the Comptroller and Auditor-General",
    150: "Form of accounts of the Union and of the States",
    151: "Audit reports",
    152: "Definition",
    153: "Governors of States",
    154: "Executive power of State",
    155: "Appointment of Governor",
    156: "Term of office of Governor",
    157: "Qualifications for appointment as Governor",
    158: "Conditions of Governor's office",
    159: "Oath or affirmation by the Governor",
    160: "Discharge of the functions of the Governor in certain contingencies",
    161: "Power of Governor to grant pardons, etc., and to suspend, remit or commute sentences in certain cases",
    162: "Extent of executive power of State",
    163: "Council of Ministers to aid and advise Governor",
    164: "Other provisions as to Ministers",
    165: "Advocate-General for the State",
    166: "Conduct of business of the Government of a State",
    167: "Duties of Chief Minister as respects the furnishing of information to Governor, etc.",
    168: "Constitution of Legislatures in States",
    169: "Abolition or creation of Legislative Councils in States",
    170: "Composition of the Legislative Assemblies",
    171: "Composition of the Legislative Councils",
    172: "Duration of State Legislatures",
    173: "Qualification for membership of the State Legislature",
    174: "Sessions of the State Legislature, prorogation and dissolution",
    175: "Right of Governor to address and send messages to the House or Houses",
    176: "Special address by the Governor",
    177: "Rights of Ministers and Advocate-General as respects the Houses",
    213: "Power of Governor to promulgate Ordinances during recess of Legislature",
    214: "High Courts for States",
    215: "High Courts to be courts of record",
    216: "Constitution of High Courts",
    217: "Appointment and conditions of the office of a Judge of a High Court",
    218: "Application of certain provisions relating to Supreme Court to High Courts",
    219: "Oath or affirmation by Judges of High Courts",
    220: "Restriction on practice after being a permanent Judge",
    221: "Salaries, etc., of Judges",
    222: "Transfer of a Judge from one High Court to another",
    223: "Appointment of acting Chief Justice",
    224: "Appointment of additional and acting Judges",
    225: "Jurisdiction of existing High Courts",
    226: "Power of High Courts to issue certain writs",
    227: "Power of superintendence over all courts by the High Court",
    228: "Transfer of certain cases to High Court",
    229: "Officers and servants and the expenses of High Courts",
    230: "Extension of jurisdiction of High Courts to Union territories",
    231: "Establishment of a common High Court for two or more States",
    232: "Interpretation",
    233: "Appointment of district judges",
    234: "Recruitment of persons other than district judges to the judicial service",
    235: "Control over subordinate courts",
    236: "Interpretation",
    237: "Application of the provisions of this Chapter to certain class or classes of magistrates",
    239: "Administration of Union territories",
    243: "Definitions",
    244: "Administration of Scheduled Areas and Tribal Areas",
    245: "Extent of laws made by Parliament and by the Legislatures of States",
    246: "Subject-matter of laws made by Parliament and by the Legislatures of States",
    247: "Power of Parliament to provide for the establishment of certain additional courts",
    248: "Residuary powers of legislation",
    249: "Power of Parliament to legislate with respect to a matter in the State List in the national interest",
    250: "Power of Parliament to legislate with respect to any matter in the State List if a Proclamation of Emergency is in operation",
    251: "Inconsistency between laws made by Parliament under articles 249 and 250 and laws made by the Legislatures of States",
    252: "Power of Parliament to legislate for two or more States by consent and adoption of such legislation by any other State",
    253: "Legislation for giving effect to international agreements",
    254: "Inconsistency between laws made by Parliament and laws made by the Legislatures of States",
    255: "Requirements as to recommendations and previous sanctions to be regarded as matters of procedure only",
    256: "Obligation of States and the Union",
    257: "Control of the Union over States in certain cases",
    258: "Power of the Union to confer powers, etc., on States in certain cases",
    259: "Armed Forces in States in Part B of the First Schedule (Repealed)",
    260: "Jurisdiction of the Union in relation to territories outside India",
    261: "Public acts, records and judicial proceedings",
    262: "Adjudication of disputes relating to waters of inter-State rivers or river valleys",
    263: "Provisions with respect to an inter-State Council",
    264: "Interpretation",
    265: "Taxes not to be imposed save by authority of law",
    266: "Consolidated Funds and public accounts of India and of the States",
    267: "Contingency Fund",
    268: "Duties levied by the Union but collected and appropriated by the States",
    269: "Taxes levied and collected by the Union but assigned to the States",
    270: "Taxes levied and distributed between the Union and the States",
    271: "Surcharge on certain duties and taxes for purposes of the Union",
    272: "Taxes which are levied and collected by the Union and may be distributed between the Union and the States (Repealed)",
    273: "Grants in lieu of export duty on jute and jute products",
    274: "Prior recommendation of President required to Bills affecting taxation in which States are interested",
    275: "Grants from the Union to certain States",
    276: "Taxes on professions, trades, callings and employments",
    277: "Savings",
    278: "Agreement with States in Part B of the First Schedule with regard to certain financial matters (Repealed)",
    279: "Calculation of net proceeds, etc.",
    280: "Finance Commission",
    281: "Recommendations of the Finance Commission",
    282: "Expenditure defrayable by the Union or a State out of its revenues",
    283: "Custody, etc., of Consolidated Funds, Contingency Funds and moneys credited to the public accounts",
    284: "Custody of suitors' deposits and other moneys received by public servants and courts",
    285: "Exemption of property of the Union from State taxation",
    286: "Restrictions as to imposition of tax on the sale or purchase of goods",
    287: "Exemption from taxes on electricity",
    288: "Exemption from taxation by States in respect of water or electricity in certain cases",
    289: "Exemption of property and income of a State from Union taxation",
    290: "Adjustment in respect of certain expenses and pensions",
    291: "Privy purse sums of Rulers (Repealed)",
    292: "Borrowing by the Government of India",
    293: "Borrowing by States",
    294: "Succession to property, assets, rights, liabilities and obligations in certain cases",
    295: "Succession to property, assets, rights, liabilities and obligations in other cases",
    296: "Property accruing by escheat or lapse or as bona vacantia",
    297: "Things of value within territorial waters or continental shelf and resources of the exclusive economic zone to vest in the Union",
    298: "Power to carry on trade, etc.",
    299: "Contracts",
    300: "Suits and proceedings",
    301: "Freedom of trade, commerce and intercourse",
    302: "Power of Parliament to impose restrictions on trade, commerce and intercourse",
    303: "Restrictions on the legislative powers of the Union and of the States with regard to trade and commerce",
    304: "Restrictions on trade, commerce and intercourse among States",
    305: "Saving of existing laws and laws providing for State monopolies",
    306: "Power of certain States in Part B of the First Schedule to impose restrictions on trade and commerce (Repealed)",
    307: "Appointment of authority for carrying out the purposes of articles 301 to 304",
    308: "Interpretation",
    309: "Recruitment and conditions of service of persons serving the Union or a State",
    310: "Tenure of office of persons serving the Union or a State",
    311: "Dismissal, removal or reduction in rank of persons employed in civil capacities under the Union or a State",
    312: "All-India services",
    313: "Transitional provisions",
    314: "Provision for protection of existing officers of certain services (Repealed)",
    315: "Public Service Commissions for the Union and for the States",
    316: "Appointment and term of office of members",
    317: "Removal and suspension of a member of a Public Service Commission",
    318: "Power to make regulations as to conditions of service of members and staff of the Commission",
    319: "Prohibition as to the holding of offices by members of Public Service Commissions on ceasing to be such members",
    320: "Functions of Public Service Commissions",
    321: "Power to extend functions of Public Service Commissions",
    322: "Expenses of Public Service Commissions",
    323: "Reports of Public Service Commissions",
    324: "Superintendence, direction and control of elections to be vested in an Election Commission",
    325: "No person to be ineligible for inclusion in, or to claim to be included in a special, electoral roll on grounds of religion, race, caste or sex",
    326: "Elections to the House of the People and to the Legislative Assemblies of States to be on the basis of adult suffrage",
    327: "Power of Parliament to make provision with respect to elections to Legislatures",
    328: "Power of Legislature of a State to make provision with respect to elections to such Legislature",
    329: "Bar to interference by courts in electoral matters",
    330: "Reservation of seats for Scheduled Castes and Scheduled Tribes in the House of the People",
    331: "Representation of the Anglo-Indian Community in the House of the People",
    332: "Reservation of seats for Scheduled Castes and Scheduled Tribes in the Legislative Assemblies of the States",
    333: "Representation of the Anglo-Indian Community in the Legislative Assemblies of the States",
    334: "Reservation of seats and special representation to cease after certain period",
    335: "Claims of Scheduled Castes and Scheduled Tribes to services and posts",
    336: "Special provision for Anglo-Indian Community in certain services",
    337: "Special provision with respect to educational grants for the benefit of Anglo-Indian Community",
    338: "National Commission for Scheduled Castes",
    339: "Control of the Union over the administration of Scheduled Areas and the welfare of Scheduled Tribes",
    340: "Appointment of a Commission to investigate the conditions of backward classes",
    341: "Scheduled Castes",
    342: "Scheduled Tribes",
    343: "Official language of the Union",
    344: "Commission and Committee of Parliament on official language",
    345: "Official language or languages of a State",
    346: "Official language for communication between one State and another or between a State and the Union",
    347: "Special provision relating to language spoken by a section of the population of a State",
    348: "Language to be used in the Supreme Court and in the High Courts and for Acts, Bills, etc.",
    349: "Special procedure for enactment of certain laws relating to language",
    350: "Language to be used in representations for redress of grievances",
    351: "Directive for development of the Hindi language",
    352: "Proclamation of Emergency",
    353: "Effect of Proclamation of Emergency",
    354: "Application of provisions relating to distribution of revenues while a Proclamation of Emergency is in operation",
    355: "Duty of the Union to protect States against external aggression and internal disturbance",
    356: "Provisions in case of failure of constitutional machinery in State",
    357: "Exercise of legislative powers under Proclamation issued under article 356",
    358: "Suspension of provisions of article 19 during emergencies",
    359: "Suspension of the enforcement of the rights conferred by Part III during emergencies",
    360: "Provisions as to Financial Emergency",
    361: "Protection of President and Governors and Rajpramukhs",
    362: "Rights and privileges of Rulers of Indian States (Repealed)",
    363: "Bar to interference by courts in disputes arising out of certain treaties, agreements, etc.",
    364: "Special provisions as to major ports and aerodromes",
    365: "Effect of failure to comply with, or to give effect to, directions given by the Union",
    366: "Definitions",
    367: "Interpretation",
    368: "Power of Parliament to amend the Constitution and procedure therefor",
    369: "Temporary power to Parliament to make laws with respect to certain matters in the State List as if they were matters in the Concurrent List",
    370: "Temporary provisions with respect to the State of Jammu and Kashmir",
    371: "Special provision with respect to the States of Maharashtra and Gujarat",
    372: "Continuance in force of existing laws and their adaptation",
    373: "Power of President to make order in respect of persons under preventive detention in certain cases",
    374: "Provisions as to Judges of the Federal Court and proceedings pending in the Federal Court or before His Majesty in Council",
    375: "Courts, authorities and officers to continue to function subject to the provisions of the Constitution",
    376: "Provisions as to Judges of High Courts",
    377: "Provisions as to Comptroller and Auditor-General of India",
    378: "Provisions as to Public Service Commissions",
    379: "Provisions as to provisional Parliament and the Speaker and Deputy Speaker thereof (Repealed)",
    380: "Provisions as to provisional Legislatures of certain States (Repealed)",
    381: "Provisions as to Council of Ministers of certain States (Repealed)",
    382: "Provisions as to Governors and Rajpramukhs (Repealed)",
    383: "Provisions as to certain class of persons serving under the Crown (Repealed)",
    384: "Provisions as to certain elections (Repealed)",
    385: "Provisions as to certain Legislatures (Repealed)",
    386: "Provisions as to certain financial matters (Repealed)",
    387: "Provisions as to certain Bills (Repealed)",
    388: "Provisions as to certain orders, rules, etc. (Repealed)",
    389: "Provisions as to certain estimates (Repealed)",
    390: "Provisions as to certain loans (Repealed)",
    391: "Provisions as to references to His Majesty or the Governor-General (Repealed)",
    392: "Power of the President to remove difficulties",
    393: "Short title",
    394: "Commencement",
    395: "Repeals",
}

SCHEDULE_LABELS = {
    1: "First Schedule — Names of States and territories",
    2: "Second Schedule — Provisions as to certain Emoluments, Allowances, Privileges and Rights",
    3: "Third Schedule — Forms of Oaths or Affirmations",
    4: "Fourth Schedule — Allocation of seats in the Council of States",
    5: "Fifth Schedule — Provisions as to the Administration and Control of Scheduled Areas and Scheduled Tribes",
    6: "Sixth Schedule — Provisions as to the Administration of Tribal Areas in the States of Assam, Meghalaya, Tripura and Mizoram",
    7: "Seventh Schedule — Union List, State List, Concurrent List",
    8: "Eighth Schedule — Languages",
    9: "Ninth Schedule — Acts and Regulations",
    10: "Tenth Schedule — Provisions as to Disqualification on Ground of Defection",
    11: "Eleventh Schedule — Powers, authority and responsibilities of Panchayats",
    12: "Twelfth Schedule — Powers, authority and responsibilities of Municipalities",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None) -> requests.Response | None:
    """GET with exponential backoff on 429 / 5xx."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_BASE ** attempt
                time.sleep(wait)
                continue
            return None
        except requests.RequestException:
            wait = BACKOFF_BASE ** attempt
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _search_article(query: str) -> str | None:
    """
    Search Indian Kanoon for a query and return the first doc_id found.
    """
    params = {"formInput": query, "pagenum": 0}
    resp = _get(SEARCH_URL, params=params)
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    # result links look like /doc/1234567/
    for a in soup.select("div.result_title a[href]"):
        href = a["href"]
        m = re.match(r"^/doc/(\d+)/?$", href)
        if m:
            return m.group(1)
    return None


def _fetch_doc_text(doc_id: str) -> tuple[str, str]:
    """
    Fetch plain text of a document from Indian Kanoon.
    Returns (title, body_text).
    """
    url = DOC_URL.format(doc_id=doc_id)
    resp = _get(url)
    if not resp:
        return "", ""
    soup = BeautifulSoup(resp.text, "lxml")
    title_tag = soup.find("h2", class_="doc_title") or soup.find("title")
    title = title_tag.get_text(separator=" ", strip=True) if title_tag else ""
    # main doc body
    doc_div = soup.find("div", id="judgments") or soup.find("div", class_="judgments")
    if not doc_div:
        doc_div = soup.find("div", class_="doc_content") or soup.find("body")
    text = doc_div.get_text(separator="\n", strip=True) if doc_div else ""
    return title, text


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _chunks_for_article(
    article_key: str | int,
    heading: str,
    part: str,
    doc_id: str,
    body_text: str,
) -> list[dict]:
    """Build chunk_text metadata and return chunks for one article."""
    if article_key == "preamble":
        article_num_str = "Preamble"
        citation = "Preamble, Constitution of India"
    else:
        article_num_str = str(article_key)
        citation = f"Article {article_key}, Constitution of India"

    metadata = {
        "source": "india_constitution",
        "jurisdiction": "india",
        "collection": "india_constitution",
        "document": "Constitution of India",
        "part": part,
        "article_num": article_num_str,
        "article_heading": heading,
        "citation": citation,
        "doc_id": doc_id,
    }
    return chunk_text(body_text, metadata)


def ingest() -> int:
    """
    Download all Constitution of India articles from Indian Kanoon,
    chunk them, and write to data/processed/india_constitution.jsonl.

    Returns the total number of chunks written.
    """
    all_raw_lines: list[str] = []
    total_chunks = 0

    # Build ordered work list: preamble, articles 1-395, schedules 1-12
    work_items: list[tuple] = []

    # Preamble
    work_items.append(("preamble", "Preamble", "Preamble"))

    # Articles
    for n in range(1, 396):
        heading = ARTICLE_HEADINGS.get(n, f"Article {n}")
        part = _article_part(n)
        work_items.append((n, heading, part))

    # Schedules
    for s in range(1, 13):
        work_items.append((f"schedule_{s}", SCHEDULE_LABELS[s], "Schedules"))

    with jsonlines.open(OUT_FILE, mode="w") as writer:
        for key, heading, part in tqdm(work_items, desc="Constitution articles", unit="article"):
            # Build search query
            if key == "preamble":
                query = "preamble constitution of india"
            elif isinstance(key, int):
                query = f"constitution of india article {key} {heading[:40]}"
            else:  # schedule
                snum = key.split("_")[1]
                query = f"constitution of india schedule {snum}"

            doc_id = _search_article(query)
            time.sleep(SLEEP_BETWEEN)

            body_text = ""
            if doc_id:
                _, body_text = _fetch_doc_text(doc_id)
                time.sleep(SLEEP_BETWEEN)

            if not body_text:
                # Fallback: use heading as minimal text so we still emit a chunk
                body_text = f"{heading}. (Full text not retrieved.)"
                doc_id = doc_id or "unknown"

            all_raw_lines.append(f"--- {key} | {heading} ---\n{body_text}\n")

            chunks = _chunks_for_article(key, heading, part, doc_id or "unknown", body_text)
            for chunk in chunks:
                writer.write(chunk)
            total_chunks += len(chunks)

    # Write raw concatenated text
    RAW_FILE.write_text("\n".join(all_raw_lines), encoding="utf-8")
    print(f"\nDone. {total_chunks} chunks written to {OUT_FILE}")
    print(f"Raw text saved to {RAW_FILE}")
    return total_chunks


if __name__ == "__main__":
    ingest()
