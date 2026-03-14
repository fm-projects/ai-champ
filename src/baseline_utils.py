#!/usr/bin/env python3
"""Utilities for rule-based PII NER baseline."""

import ast
import csv
import hashlib
import os
import random
import re
from typing import Dict, List, Sequence, Tuple

Span = Tuple[int, int, str]

try:
    from natasha import MorphVocab, NamesExtractor  # type: ignore
except Exception:  # pragma: no cover
    MorphVocab = None
    NamesExtractor = None

try:
    from pymorphy3 import MorphAnalyzer as MorphAnalyzer3  # type: ignore
except Exception:  # pragma: no cover
    MorphAnalyzer3 = None

try:
    from pymorphy2 import MorphAnalyzer as MorphAnalyzer2  # type: ignore
except Exception:  # pragma: no cover
    MorphAnalyzer2 = None

try:
    import pycountry  # type: ignore
except Exception:  # pragma: no cover
    pycountry = None

try:
    from babel import Locale  # type: ignore
except Exception:  # pragma: no cover
    Locale = None


# =========================
# I/O and metrics
# =========================


def configure_determinism(seed: int = 42, threads: int = 1) -> None:
    """Configure reproducible execution for python + common numeric stack."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

def parse_target(value: str) -> List[Span]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("target must be a list")

    out: List[Span] = []
    for item in parsed:
        if not (isinstance(item, (tuple, list)) and len(item) == 3):
            raise ValueError(f"Bad entity tuple: {item}")
        start, end, category = item
        if not (isinstance(start, int) and isinstance(end, int) and isinstance(category, str)):
            raise ValueError(f"Bad entity types: {item}")
        out.append((start, end, category))
    return out


def load_train_tsv(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for idx, row in enumerate(reader):
            text = row["text"]
            target_parsed = parse_target(row["target"])
            for start, end, _ in target_parsed:
                if not (0 <= start < end <= len(text)):
                    raise ValueError(f"Bad span bounds at row {idx}: {(start, end)} for text len {len(text)}")
            rows.append(
                {
                    "id": str(idx),
                    "text": text,
                    "target": row["target"],
                    "target_parsed": target_parsed,
                    "entity": row.get("entity", ""),
                }
            )
    return rows


def load_text_csv(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if "id" not in row or "text" not in row:
                raise ValueError(f"File {path} must contain id,text columns")
            rows.append({"id": str(row["id"]), "text": row["text"]})
    return rows


def load_id_target_csv(path: str) -> Dict[str, List[Span]]:
    out: Dict[str, List[Span]] = {}
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if "id" not in row or "target" not in row:
                raise ValueError(f"File {path} must contain id,target columns")
            out[str(row["id"])] = parse_target(row["target"])
    return out


def save_id_target_csv(rows: Sequence[Dict], preds: Sequence[Sequence[Span]], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "target"])
        for row, pred in zip(rows, preds):
            writer.writerow([row["id"], repr(list(pred))])


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def hash_to_bucket(value: str, n_buckets: int, seed: int) -> int:
    digest = hashlib.md5((str(seed) + value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def strict_micro_f1(y_true: Sequence[Sequence[Span]], y_pred: Sequence[Sequence[Span]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for true_row, pred_row in zip(y_true, y_pred):
        true_set = set((int(s), int(e), str(c)) for s, e, c in true_row)
        pred_set = set((int(s), int(e), str(c)) for s, e, c in pred_row)
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def per_category_f1(y_true: Sequence[Sequence[Span]], y_pred: Sequence[Sequence[Span]]) -> List[Tuple[str, float, int]]:
    categories = sorted({cat for row in y_true for _, _, cat in row})
    report: List[Tuple[str, float, int]] = []

    for category in categories:
        true_cat = [[e for e in row if e[2] == category] for row in y_true]
        pred_cat = [[e for e in row if e[2] == category] for row in y_pred]
        metrics = strict_micro_f1(true_cat, pred_cat)
        support = sum(len(row) for row in true_cat)
        report.append((category, metrics["f1"], support))

    return sorted(report, key=lambda item: (-item[1], item[0]))


# =========================
# Shared helpers
# =========================

def _near_keywords(text: str, start: int, end: int, keywords: Sequence[str], window: int = 44) -> bool:
    if not keywords:
        return True
    left = max(0, start - window)
    right = min(len(text), end + window)
    chunk = text[left:right].lower()
    return any(keyword in chunk for keyword in keywords)


def _trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start] in ' \t\n\r"\'():,;':
        start += 1
    while end > start and text[end - 1] in ' \t\n\r"\'():,;':
        end -= 1
    return start, end


def _add_span(spans: List[Span], text: str, start: int, end: int, category: str) -> None:
    start, end = _trim_span(text, start, end)
    if start < end:
        spans.append((start, end, category))


def _looks_like_place(value: str) -> bool:
    cleaned = value.strip(" \"'«»:,.;")
    if len(cleaned) < 3:
        return False
    if re.search(r"\d", cleaned):
        return False
    lower = cleaned.lower()
    if any(month in lower for month in MONTH_WORDS):
        return False
    if WEEKDAY_RX.search(lower):
        return False
    return bool(re.search(r"[А-ЯЁ]", cleaned))


def _find_with_context(
    text: str,
    regex: re.Pattern,
    category: str,
    keywords: Sequence[str],
    window: int = 44,
    group: int = 0,
) -> List[Span]:
    spans: List[Span] = []
    for match in regex.finditer(text):
        start, end = match.span(group)
        if _near_keywords(text, start, end, keywords, window=window):
            _add_span(spans, text, start, end, category)
    return spans


def _init_natasha_extractor():
    if NamesExtractor is None or MorphVocab is None:
        return None
    try:
        return NamesExtractor(MorphVocab())
    except Exception:
        return None


def _init_morph():
    if MorphAnalyzer3 is not None:
        try:
            return MorphAnalyzer3()
        except Exception:
            pass
    if MorphAnalyzer2 is not None:
        try:
            return MorphAnalyzer2()
        except Exception:
            pass
    return None


def _load_country_forms() -> set[str]:
    out: set[str] = set()

    if pycountry is not None:
        try:
            for country in pycountry.countries:
                for attr in ("name", "official_name", "common_name"):
                    value = getattr(country, attr, None)
                    if value:
                        out.add(str(value).lower())
        except Exception:
            pass

    if Locale is not None:
        try:
            ru_locale = Locale.parse("ru")
            for value in ru_locale.territories.values():
                if isinstance(value, str):
                    out.add(value.lower())
        except Exception:
            pass

    return out


NATASHA_EXTRACTOR = _init_natasha_extractor()
MORPH = _init_morph()
COUNTRY_FORMS = _load_country_forms()


# =========================
# Base regex rules
# =========================

# tuple: (category, compiled_regex, context_keywords)
BASE_RULES: List[Tuple[str, re.Pattern, Tuple[str, ...]]] = [
    ("Email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), ()),
    ("Номер телефона", re.compile(r"(?<!\w)(?:\+7|8)[\s\-\(]*\d[\d\s\-\(\)]{8,16}\d(?!\w)"), ()),
    ("Дата окончания срока действия карты", re.compile(r"(?<!\d)(0[1-9]|1[0-2])/[0-9]{2}(?!\d)"), ("кар", "срок", "действ")),
    ("CVV/CVC", re.compile(r"(?<!\d)\d{3}(?!\d)"), ("cvv", "cvc", "код", "security")),
    ("ПИН код", re.compile(r"(?<!\d)\d{4}(?!\d)"), ("pin", "пин")),
    ("Одноразовые коды", re.compile(r"(?<!\d)\d{6}(?!\d)"), ("однораз", "sms", "смс", "otp", "код")),
    ("Номер карты", re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)"), ("карт", "pan", "номер карты")),
    ("Номер банковского счета", re.compile(r"(?<!\d)\d{20}(?!\d)"), ("счет", "счёт", "расчет", "расчёт", "р/с", "банков")),
    ("Сведения об ИНН", re.compile(r"(?<!\d)(?:\d{10}|\d{12})(?!\d)"), ("инн",)),
    ("СНИЛС клиента", re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}\s\d{2}(?!\d)"), ()),
    ("СНИЛС клиента", re.compile(r"(?<!\d)\d{11}(?!\d)"), ("снилс",)),
    ("Разрешение на работу / визу", re.compile(r"(?<!\d)\d{2}\s\d{7}(?!\d)"), ("разреш", "виз")),
    ("Водительское удостоверение", re.compile(r"(?<!\d)\d{2}\s\d{2}\s\d{6}(?!\d)"), ("водитель", "удостовер", "права", "ву")),
    ("Водительское удостоверение", re.compile(r"(?<!\d)\d{4}\s\d{6}(?!\d)"), ("водитель", "удостовер", "права", "ву")),
    ("Серия и номер вида на жительство", re.compile(r"(?<!\d)\d{2}\s\d{7,8}(?!\d)"), ("вид на жительство", "внж")),
    ("Паспортные данные", re.compile(r"(?<!\d)\d{4}\s\d{6}(?!\d)"), ("паспорт", "серия", "выдан")),
    ("Дата рождения", re.compile(r"(?<!\d)(?:19\d{2}|20[01]\d|202[0-6])(?!\d)"), ("рожд", "дата рождения")),
    ("Дата рождения", re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./](?:0?[1-9]|1[0-2])[./](?:19\d{2}|20\d{2})(?!\d)"), ("рожд", "дата рождения")),
    ("Дата регистрации по месту жительства или пребывания", re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./](?:0?[1-9]|1[0-2])[./](?:19\d{2}|20\d{2})(?!\d)"), ("регистрац", "прописк", "месту жительства", "пребывания")),
    ("API ключи", re.compile(r"(?<!\S)AIza[0-9A-Za-z_\-]{20,}(?!\S)"), ()),
    ("API ключи", re.compile(r"(?<!\S)\d{8,10}:[0-9A-Za-z_\-]{20,}(?!\S)"), ()),
    ("API ключи", re.compile(r"(?<!\S)(?:sk|pk)_[A-Za-z0-9_\-]{16,}(?!\S)"), ()),
    ("API ключи", re.compile(r"(?<!\S)[A-Za-z0-9_\-]{24,}(?!\S)"), ("api", "key", "token", "secret", "access")),
]


# =========================
# Weak classes config
# =========================

BANK_ROOT_PATTERNS = [
    r"сбербанк\w*",
    r"втб",
    r"альфа-банк\w*",
    r"газпромбанк\w*",
    r"тинькофф",
    r"райффайзенбанк\w*",
    r"росбанк\w*",
    r"открытие",
    r"совкомбанк\w*",
    r"уралсиб\w*",
    r"юникредит",
    r"рнкб",
    r"вбрр",
    r"ситибанк\w*",
    r"дельтакредит",
    r"ренессанс кредит",
    r"убрир",
    r"смп",
    r"отп",
    r"мтс",
    r"бкс",
    r"бинбанк\w*",
    r"абсолют",
    r"дом\.рф",
    r"ак барс",
    r"авангард",
    r"интеза",
    r"русский стандарт",
    r"санкт-петербург",
    r"урал фд",
    r"финсервис",
    r"центр-инвест",
    r"возрождение",
    r"зенит",
    r"точка",
    r"восточный",
    r"восточного",
    r"инвестторгбанк\w*",
    r"кредит европа",
    r"кубань кредит",
    r"металлинвестбанк\w*",
    r"московский кредитный",
    r"народный банк\w* казахстана",
    r"новикомбанк\w*",
    r"почта",
    r"промсвязьбанк\w*",
    r"рнко платежный центр",
    r"россельхозбанк\w*",
    r"сдм-банк\w*",
    r"солид",
    r"транскапиталбанк\w*",
    r"хоум кредит",
    r"halyk",
    r"kaspi",
]
BANK_ROOT_REGEXES = [
    re.compile(rf"(?<![a-zа-яё0-9])(?:{pattern})(?![a-zа-яё0-9])", re.IGNORECASE)
    for pattern in BANK_ROOT_PATTERNS
]

COUNTRY_ROOTS = [
    "росси", "армени", "белорус", "беларус", "казахстан", "кыргыз", "киргиз", "узбекистан", "таджикистан",
    "азербайдж", "грузи", "молд", "молдав", "украин", "литв", "латви", "эстони", "польш", "герман", "франци",
    "итал", "испан", "португал", "греци", "болгар", "серб", "черногор", "македон", "венгри", "румын", "австр",
    "швейцар", "нидерланд", "бельги", "великобрит", "англи", "шотланд", "ирланд", "норвег", "швец", "финлянд",
    "дан", "исланд", "сша", "америк", "канад", "мексик", "бразил", "аргентин", "чили", "перу", "колумб", "куб",
    "егип", "марок", "тунис", "алжир", "ливан", "израил", "иран", "ирак", "турци", "сауд", "оаэ", "катар", "кита",
    "япон", "коре", "инд", "пакистан", "таиланд", "вьетнам", "малайз", "сингапур", "индонез", "филиппин",
    "австрал", "новозеланд", "монгол",
]

FIO_CONTEXT = ("фио", "фамил", "имя", "отчеств", "получател", "клиент", "держател")
CITIZEN_CONTEXT = ("гражданств", "гражданин", "подданств", "национальн")
BIRTH_CONTEXT = ("место рождения", "родился", "родилась", "урожен", "уроженка", "родом")
TEMP_ID_CONTEXT = ("временн", "удостовер", "личности")
CARD_HOLDER_CONTEXT = (
    "имя держателя",
    "держател",
    "holder name",
    "name on card",
    "карты",
    "карте",
    "карта",
    "именем",
)
BIRTH_CERT_CONTEXT = ("свидетельств", "о рождении")
AUTO_CONTEXT = ("vin", "госномер", "кузов", "двигател", "марка", "модель", "автомоб")

PASSWORD_CONTEXT = ("парол", "password", "pwd", "код доступа", "учетн", "учётн")
ORG_INN_CONTEXT = ("инн", "юрлиц", "организац")
ORG_KPP_BIK_CONTEXT = ("кпп", "бик")
ORG_OGRN_CONTEXT = ("огрн",)
ORG_ACCOUNT_CONTEXT = ("расчет", "расчёт", "р/с")

ADDRESS_TOKENS = (
    "г.", "город", "ул.", "улица", "просп", "пр-кт", "переул", "пер.", "дом", "д.", "кв.", "квартира",
    "корп.", "обл.", "область", "район", "оф.", "офис", "индекс",
)

PASSWORD_RX = re.compile(r"(?<!\S)[A-Za-z0-9@#$%^&*()_+\-=[\]{};:'\",.<>/?!]{8,64}(?!\S)")
TRACK1_RX = re.compile(r"%B\d{13,19}\^[A-ZА-ЯЁ/ ]{2,26}\^\d{4,}\?")
TRACK2_RX = re.compile(r"\d{13,19}=\d{4,}\d*")
EMV_TAG_RX = re.compile(r"9F[0-9A-F]{2}[0-9A-F]{2,}", re.IGNORECASE)

CAP_WORD_RX = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\b")
ADDR_COMPONENT_RX = re.compile(
    r"\b(?:г\.|город|ул\.|улица|просп(?:ект)?|пр-кт|пер\.|переулок|дом(?![а-яё])|д\.|кв\.|квартира|корп\.|"
    r"обл\.|область|район|оф\.|офис|стр\.|строение|пос\.|поселок|респ\.|республика|край)\s*"
    r"[A-Za-zА-Яа-яЁё0-9\- ]{1,40}"
)
POSTAL_RX = re.compile(r"(?<!\d)\d{6}(?!\d)")
BIRTH_AFTER_RX = re.compile(r"(?:место рождения\s*[:\-]?\s*|родил(?:ся|ась)\s+в\s+|урожен(?:ец|ка)?\s+)([А-ЯЁ][^,.;\n]{2,40})")
BIRTH_PLACE_QUOTED_RX = re.compile(r"[\"«]([А-ЯЁ][А-Яа-яЁё\- ]{2,45})[\"»]")
REG_DATE_TEXT_RX = re.compile(
    r"(?:(?:с|до)\s+)?((?:0?[1-9]|[12]\d|3[01])\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"(?:\s+(?:19\d{2}|20\d{2})(?:\s+года)?)?)",
    re.IGNORECASE,
)
WEEKDAY_RX = re.compile(
    r"\b(?:понедельник(?:а|у|ом)?|вторник(?:а|у|ом)?|сред(?:а|у|ы|ой)|четверг(?:а|у|ом)?|"
    r"пятниц(?:а|у|ы|ей)|суббот(?:а|у|ы|ой)|воскресень(?:е|я|ю|ем))\b",
    re.IGNORECASE,
)
TIME_RX = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)")
DATE_TIME_RX = re.compile(
    r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./](?:0?[1-9]|1[0-2])[./](?:19\d{2}|20\d{2})"
    r"(?:\s*(?:в\s*)?\d{1,2}:\d{2}(?::\d{2})?)?(?!\d)"
)
DATE_TEXT_WITH_TIME_RX = re.compile(
    r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"\s+(?:19\d{2}|20\d{2})(?:\s+года)?(?:\s+в\s+\d{1,2}:\d{2}(?::\d{2})?)?",
    re.IGNORECASE,
)
BIRTH_COMPONENTS_RX = re.compile(
    r"(?<!\d)(0?[1-9]|[12]\d|3[01])\s+"
    r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"\s+((?:19\d{2}|20\d{2}))(?!\d)",
    re.IGNORECASE,
)
REG_CONTEXT = ("регистрац", "прописк", "месту жительства", "пребывания", "зарегистр")
ORG_ADDRESS_RX = re.compile(
    r"(?:юридическ(?:ий|ого)\s+адрес|почтов(?:ый|ого)\s+адрес|адрес)\s*[:\-]?\s*([^.;\n]{12,95})",
    re.IGNORECASE,
)
GEO_PREP_RX = re.compile(r"(?:\bв|\bиз|\bдо|\bчерез|\bпо|\bдля)\s*$", re.IGNORECASE)

TEMP_ID_PATTERNS = [
    re.compile(r"(?<!\d)\d{2}\s\d{2}\s\d{6}(?!\d)"),
    re.compile(r"(?<!\d)\d{4}\s\d{6}(?!\d)"),
    re.compile(r"(?<!\d)\d{10}(?!\d)"),
    re.compile(r"(?<!\d)\d{8}(?!\d)"),
    re.compile(r"(?<!\d)\d{12}(?!\d)"),
    re.compile(r"(?<!\w)[А-ЯA-Z]{1,2}\d{8}(?!\w)"),
]
CARD_HOLDER_RX = re.compile(r"(?<![A-Z])[A-Z]{2,}(?:\s+[A-Z]{2,}){1,3}(?![A-Z])")
CODE_WORD_WITH_CONTEXT_RX = re.compile(
    r"(?:кодов(?:ое|ого|ым)?\s+слов(?:о|а)|секретн(?:ое|ого)?\s+слов(?:о|а))\s*[:\-]?\s*[\"'«]?"
    r"([A-Za-zА-Яа-яЁё0-9_!@#$%^&*+\-]{4,20})",
    re.IGNORECASE,
)
BIRTH_CERT_PATTERNS = [
    re.compile(r"(?<!\d)[IVXLCDM]{1,6}-?[А-ЯA-Z]{1,3}\s?\d{6}(?!\d)"),
    re.compile(r"(?<!\d)\d{2}\s?\d{2}\s?\d{6}(?!\d)"),
]
VNZ_PATTERNS = [
    re.compile(r"(?<!\d)\d{2}\s?\d{7}(?!\d)"),
    re.compile(r"(?<!\d)\d{2}\-\d{7}(?!\d)"),
    re.compile(r"(?<!\w)\d{2}№\d{7}(?!\w)"),
    re.compile(r"(?<!\w)\d{2}\s?№\s?\d{7}(?!\w)"),
    re.compile(r"(?<!\d)\d{9}(?!\d)"),
]
VIN_RX = re.compile(r"(?<!\w)[A-HJ-NPR-Z0-9]{17}(?!\w)")
CAR_PLATE_RX = re.compile(r"(?<!\w)[АВЕКМНОРСТУХABEKMHOPCTYX]\d{3}[АВЕКМНОРСТУХABEKMHOPCTYX]{2}\d{2,3}(?!\w)")
CAR_YEAR_RX = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
ORG_INN_RX = re.compile(r"(?<!\d)\d{10}(?!\d)")
ORG_KPP_BIK_RX = re.compile(r"(?<!\d)\d{9}(?!\d)")
ORG_OGRN_RX = re.compile(r"(?<!\d)\d{13}(?!\d)")
ORG_ACCOUNT_RX = re.compile(r"(?<!\d)\d{20}(?!\d)")
ORG_LEGAL_FORM_RX = re.compile(r"\b(?:ООО|АО|ПАО|ИП)\s+[\"«]?[A-Za-zА-Яа-яЁё0-9 .\-]{2,40}[\"»]?")
PASSPORT_DATE_RX = re.compile(
    r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./](?:0?[1-9]|1[0-2])[./](?:19\d{2}|20\d{2})(?!\d)"
)
PASSPORT_TEXT_DATE_RX = re.compile(
    r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"\s+(?:19\d{2}|20\d{2})(?:\s+года)?",
    re.IGNORECASE,
)
PASSPORT_SERIES_NUMBER_RX = re.compile(r"(?<!\d)\d{4}\s?\d{6}(?!\d)")
PASSPORT_CODE_RX = re.compile(r"(?<!\d)\d{6}(?!\d)")
PASSPORT_CODE_DASH_RX = re.compile(r"(?<!\d)\d{3}-\d{3}(?!\d)")
PASSPORT_AUTHORITY_RX = re.compile(
    r"\b(?:УФМС|ОУФМС|ОВД|УВД|МВД|ГУ МВД|Управлением по вопросам миграции)[^,.;\n]{0,80}",
    re.IGNORECASE,
)
SERIES_CONTEXT_RX = re.compile(r"(?:серия|серии)\s*[:\-]?\s*(\d{4})", re.IGNORECASE)
NUMBER_CONTEXT_RX = re.compile(r"(?:номер|номером)\s*[:\-]?\s*(\d{6})", re.IGNORECASE)

MONTH_WORDS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


# =========================
# Detectors
# =========================

def predict_rules_one(text: str) -> List[Span]:
    spans: List[Span] = []
    for category, regex, keywords in BASE_RULES:
        spans.extend(_find_with_context(text, regex, category, keywords))
    return sorted(set(spans), key=lambda item: (item[0], item[1], item[2]))


def detect_passwords(text: str) -> List[Span]:
    spans: List[Span] = []
    for match in PASSWORD_RX.finditer(text):
        token = match.group(0)
        has_letters = bool(re.search(r"[A-Za-z]", token))
        has_digit_or_symbol = bool(re.search(r"\d|[@#$%^&*()_+\-=[\]{};:'\",.<>/?!]", token))
        if has_letters and has_digit_or_symbol and _near_keywords(text, match.start(), match.end(), PASSWORD_CONTEXT):
            _add_span(spans, text, match.start(), match.end(), "Пароли")
    return spans


def detect_magstripe(text: str) -> List[Span]:
    spans: List[Span] = []
    for regex in (TRACK1_RX, TRACK2_RX, EMV_TAG_RX):
        for match in regex.finditer(text):
            _add_span(spans, text, match.start(), match.end(), "Содержимое магнитной полосы")
    return spans


def detect_bank_names(text: str) -> List[Span]:
    spans: List[Span] = []
    for regex in BANK_ROOT_REGEXES:
        for match in regex.finditer(text):
            _add_span(spans, text, match.start(), match.end(), "Наименование банка")
    # Generic bank name shapes in Russian and Latin.
    generic_bank_patterns = [
        re.compile(r"\b[А-ЯЁA-Z][А-Яа-яЁёA-Za-z\- ]{2,35}\s+Банк(?:а|у|е|ом)?\b"),
        re.compile(r"\bБанк(?:а|у|е|ом)?\s+[А-ЯЁA-Z][А-Яа-яЁёA-Za-z\- ]{2,35}\b"),
        re.compile(r"\b[A-Z][A-Za-z]{2,20}\s+Bank\b"),
        re.compile(r"\b[А-ЯЁ][А-Яа-яЁёA-Za-z\-]{4,30}банк(?:а|у|е|ом)?\b"),
    ]
    for regex in generic_bank_patterns:
        for match in regex.finditer(text):
            _add_span(spans, text, match.start(), match.end(), "Наименование банка")
    return spans


def detect_fio(text: str) -> List[Span]:
    spans: List[Span] = []
    lower_text = text.lower()

    if NATASHA_EXTRACTOR is not None:
        try:
            matches = NATASHA_EXTRACTOR(text)
            for match in matches:
                start, end = match.span
                # Dataset often marks parts of FIO as separate entities.
                segment = text[start:end]
                for token in CAP_WORD_RX.finditer(segment):
                    tok_start = start + token.start()
                    tok_end = start + token.end()
                    _add_span(spans, text, tok_start, tok_end, "ФИО")
        except Exception:
            pass

    for keyword in FIO_CONTEXT:
        for m in re.finditer(re.escape(keyword), lower_text):
            right = text[m.end(): min(len(text), m.end() + 45)]
            right = re.split(r"[,.!?;\n]", right, maxsplit=1)[0]
            if len(right) < 2:
                continue
            for token in CAP_WORD_RX.finditer(right):
                start = m.end() + token.start()
                end = m.end() + token.end()
                if MORPH is not None:
                    try:
                        parsed = MORPH.parse(text[start:end])[0]
                        if not ({"Name", "Surn", "Patr"} & set(parsed.tag.grammemes)):
                            continue
                    except Exception:
                        pass
                _add_span(spans, text, start, end, "ФИО")

    # Common templates in dataset: "на имя <ФИО>", "для <ФИО>", "счет <ФИО>", "открыт на <ФИО>".
    fio_context_patterns = [
        re.compile(r"(?:на имя|для|открыт на|счет|счёт)\s+([А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){0,2})"),
        re.compile(r"(?:мои данные[,:\s]+)([А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,2})"),
        re.compile(r"(?:на|для)\s+([А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?)"),
    ]
    for pattern in fio_context_patterns:
        for match in pattern.finditer(text):
            phrase_start, _ = match.span(1)
            phrase = match.group(1)
            for token in CAP_WORD_RX.finditer(phrase):
                start = phrase_start + token.start()
                end = phrase_start + token.end()
                if MORPH is not None:
                    try:
                        parsed = MORPH.parse(text[start:end])[0]
                        if not ({"Name", "Surn", "Patr"} & set(parsed.tag.grammemes)):
                            continue
                    except Exception:
                        pass
                _add_span(spans, text, start, end, "ФИО")
    fio_phrase_rx = re.compile(r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,2}\b")
    person_ctx = (
        "фио",
        "фамил",
        "имя",
        "отчеств",
        "клиент",
        "получател",
        "счет",
        "счёт",
        "карта",
        "перевод",
        "вклад",
    )
    for match in fio_phrase_rx.finditer(text):
        if _near_keywords(text, match.start(), match.end(), person_ctx, window=44):
            for token in CAP_WORD_RX.finditer(match.group(0)):
                _add_span(
                    spans,
                    text,
                    match.start() + token.start(),
                    match.start() + token.end(),
                    "ФИО",
                )
    return spans


def detect_address(text: str) -> List[Span]:
    spans: List[Span] = []
    for match in ADDR_COMPONENT_RX.finditer(text):
        _add_span(spans, text, match.start(), match.end(), "Полный адрес")
    for match in POSTAL_RX.finditer(text):
        if _near_keywords(text, match.start(), match.end(), ADDRESS_TOKENS):
            _add_span(spans, text, match.start(), match.end(), "Полный адрес")
    return spans


def detect_birth_place(text: str) -> List[Span]:
    spans: List[Span] = []
    place_rx = re.compile(r"\b(?:г\.|город|область|республика|край)?\s*[А-ЯЁ][А-Яа-яЁё\- ]{2,35}")

    for match in BIRTH_AFTER_RX.finditer(text):
        start, end = match.span(1)
        if _looks_like_place(text[start:end]):
            _add_span(spans, text, start, end, "Место рождения")

    lower_text = text.lower()
    if "место рождения" in lower_text:
        for match in BIRTH_PLACE_QUOTED_RX.finditer(text):
            start, end = match.span(1)
            if _looks_like_place(text[start:end]):
                _add_span(spans, text, start, end, "Место рождения")

    strict_context = ("место рождения", "урожен", "уроженка", "родом")
    for keyword in strict_context:
        for m in re.finditer(re.escape(keyword), text.lower()):
            right = text[m.end(): min(len(text), m.end() + 45)]
            right = re.split(r"[.;\n]", right, maxsplit=1)[0]
            for token in place_rx.finditer(right):
                start = m.end() + token.start()
                end = m.end() + token.end()
                if _looks_like_place(text[start:end]):
                    _add_span(spans, text, start, end, "Место рождения")

    return spans


def detect_citizenship(text: str) -> List[Span]:
    spans: List[Span] = []
    token_rx = re.compile(r"\b[А-Яа-яЁёA-Za-z\-]{2,30}\b")

    for match in token_rx.finditer(text):
        token = match.group(0)
        token_lower = token.lower()
        if token_lower in {"рф", "снг", "ес"}:
            _add_span(spans, text, match.start(), match.end(), "Гражданство и названия стран")
            continue
        is_country = any(root in token_lower for root in COUNTRY_ROOTS) or token_lower in COUNTRY_FORMS
        if not is_country and MORPH is not None:
            try:
                lemma = MORPH.parse(token)[0].normal_form.lower()
                is_country = lemma in COUNTRY_FORMS or any(root in lemma for root in COUNTRY_ROOTS)
            except Exception:
                pass
        if not is_country:
            continue
        left = text[max(0, match.start() - 16): match.start()]
        near_citizen = _near_keywords(text, match.start(), match.end(), CITIZEN_CONTEXT, window=65)
        near_geo = GEO_PREP_RX.search(left) is not None
        near_travel = _near_keywords(
            text,
            match.start(),
            match.end(),
            ("посольств", "аэропорт", "виз", "перевод", "страхов", "границ"),
            window=35,
        )
        if near_citizen or (token[0].isupper() and (near_geo or near_travel)):
            _add_span(spans, text, match.start(), match.end(), "Гражданство и названия стран")

    return spans


def detect_vnz(text: str) -> List[Span]:
    spans: List[Span] = []
    lower_text = text.lower()
    if "внж" not in lower_text and "жительств" not in lower_text and "вид на" not in lower_text:
        return spans
    for regex in VNZ_PATTERNS:
        for match in regex.finditer(text):
            if _near_keywords(text, match.start(), match.end(), ("внж", "жительств", "вид на"), window=32):
                _add_span(spans, text, match.start(), match.end(), "Серия и номер вида на жительство")
    return spans


def detect_birth_date_tokens(text: str) -> List[Span]:
    spans: List[Span] = []
    lower_text = text.lower()
    if "рожд" not in lower_text:
        return spans

    for regex in (DATE_TIME_RX, DATE_TEXT_WITH_TIME_RX):
        for match in regex.finditer(text):
            if _near_keywords(text, match.start(), match.end(), ("рожд", "день рождения"), window=45):
                _add_span(spans, text, match.start(), match.end(), "Дата рождения")

    for match in TIME_RX.finditer(text):
        if _near_keywords(text, match.start(), match.end(), ("рожд", "время рождения"), window=40):
            _add_span(spans, text, match.start(), match.end(), "Дата рождения")

    for match in WEEKDAY_RX.finditer(text):
        if _near_keywords(text, match.start(), match.end(), ("рожд", "день рождения"), window=45):
            _add_span(spans, text, match.start(), match.end(), "Дата рождения")

    for match in BIRTH_COMPONENTS_RX.finditer(text):
        if not _near_keywords(text, match.start(), match.end(), ("рожд",), window=45):
            continue
        day_start, day_end = match.span(1)
        month_start, month_end = match.span(2)
        year_start, year_end = match.span(3)
        _add_span(spans, text, day_start, day_end, "Дата рождения")
        _add_span(spans, text, month_start, month_end, "Дата рождения")
        _add_span(spans, text, year_start, year_end, "Дата рождения")

    for month in MONTH_WORDS:
        for match in re.finditer(rf"\b{month}\b", lower_text):
            if _near_keywords(text, match.start(), match.end(), ("рожд", "дата рождения", "день рождения"), window=40):
                _add_span(spans, text, match.start(), match.end(), "Дата рождения")
    for match in re.finditer(r"(?<!\d)(?:19\d{2}|20[01]\d|202[0-6])(?!\d)", text):
        if _near_keywords(text, match.start(), match.end(), ("рожд",), window=40):
            _add_span(spans, text, match.start(), match.end(), "Дата рождения")
    for match in re.finditer(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])(?!\d)", text):
        if _near_keywords(text, match.start(), match.end(), MONTH_WORDS + ("рожд",), window=18):
            _add_span(spans, text, match.start(), match.end(), "Дата рождения")

    return spans


def detect_registration_dates(text: str) -> List[Span]:
    spans: List[Span] = []
    lower_text = text.lower()
    if not any(key in lower_text for key in ("регистрац", "прописк", "жительства", "пребывания", "зарегистр")):
        return spans

    for match in PASSPORT_DATE_RX.finditer(text):
        if _near_keywords(text, match.start(), match.end(), REG_CONTEXT, window=52):
            _add_span(spans, text, match.start(), match.end(), "Дата регистрации по месту жительства или пребывания")
    for match in REG_DATE_TEXT_RX.finditer(text):
        # group 1 is the actual date without leading preposition "с"/"до"
        s, e = match.span(1) if match.lastindex else (match.start(), match.end())
        if _near_keywords(text, s, e, REG_CONTEXT, window=52):
            _add_span(spans, text, s, e, "Дата регистрации по месту жительства или пребывания")
    return spans


def detect_temp_id(text: str) -> List[Span]:
    spans: List[Span] = []
    lower_text = text.lower()
    if not ("временн" in lower_text and "удостовер" in lower_text):
        return spans
    for regex in TEMP_ID_PATTERNS:
        spans.extend(_find_with_context(text, regex, "Временное удостоверение личности", TEMP_ID_CONTEXT, window=28))
    return spans


def detect_card_holder_name(text: str) -> List[Span]:
    return _find_with_context(
        text,
        CARD_HOLDER_RX,
        "Имя держателя карты",
        CARD_HOLDER_CONTEXT,
        window=80,
    )


def detect_code_words(text: str) -> List[Span]:
    return _find_with_context(text, CODE_WORD_WITH_CONTEXT_RX, "Кодовые слова", (), group=1)


def detect_birth_certificate(text: str) -> List[Span]:
    spans: List[Span] = []
    for regex in BIRTH_CERT_PATTERNS:
        spans.extend(_find_with_context(text, regex, "Свидетельство о рождении", BIRTH_CERT_CONTEXT))
    return spans


def detect_auto_data(text: str) -> List[Span]:
    spans: List[Span] = []
    for regex in (VIN_RX, CAR_YEAR_RX, CAR_PLATE_RX):
        spans.extend(_find_with_context(text, regex, "Данные об автомобиле клиента", AUTO_CONTEXT))
    return spans


def detect_org_data(text: str) -> List[Span]:
    spans: List[Span] = []
    category = "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)"

    spans.extend(_find_with_context(text, ORG_INN_RX, category, ORG_INN_CONTEXT))
    spans.extend(_find_with_context(text, ORG_KPP_BIK_RX, category, ORG_KPP_BIK_CONTEXT))
    spans.extend(_find_with_context(text, ORG_OGRN_RX, category, ORG_OGRN_CONTEXT))
    spans.extend(_find_with_context(text, ORG_ACCOUNT_RX, category, ORG_ACCOUNT_CONTEXT))

    for match in ORG_LEGAL_FORM_RX.finditer(text):
        _add_span(spans, text, match.start(), match.end(), category)

    return spans


def detect_passport_details(text: str) -> List[Span]:
    spans: List[Span] = []
    lower_text = text.lower()
    has_passport_context = ("паспорт" in lower_text) or ("серия" in lower_text and "номер" in lower_text)
    if not has_passport_context:
        return spans

    spans.extend(
        _find_with_context(
            text,
            PASSPORT_SERIES_NUMBER_RX,
            "Паспортные данные",
            ("паспорт", "серия", "номер"),
            window=40,
        )
    )
    spans.extend(
        _find_with_context(
            text,
            PASSPORT_DATE_RX,
            "Паспортные данные",
            ("паспорт", "выдан", "срок действия"),
            window=44,
        )
    )
    spans.extend(
        _find_with_context(
            text,
            PASSPORT_TEXT_DATE_RX,
            "Паспортные данные",
            ("паспорт", "выдан", "срок действия", "загранпаспорт"),
            window=44,
        )
    )
    spans.extend(
        _find_with_context(
            text,
            PASSPORT_CODE_RX,
            "Паспортные данные",
            ("код подразделения", "паспорт"),
            window=36,
        )
    )
    spans.extend(
        _find_with_context(
            text,
            PASSPORT_CODE_DASH_RX,
            "Паспортные данные",
            ("код подразделения", "паспорт"),
            window=36,
        )
    )
    for regex, group_size in ((SERIES_CONTEXT_RX, 1), (NUMBER_CONTEXT_RX, 1)):
        for match in regex.finditer(text):
            start, end = match.span(group_size)
            _add_span(spans, text, start, end, "Паспортные данные")
    for match in PASSPORT_AUTHORITY_RX.finditer(text):
        _add_span(spans, text, match.start(), match.end(), "Паспортные данные")
    return spans


def predict_weak_classes_one(text: str) -> List[Span]:
    spans: List[Span] = []
    for detector in WEAK_DETECTORS:
        spans.extend(detector(text))
    return sorted(set(spans), key=lambda item: (item[0], item[1], item[2]))


WEAK_DETECTORS = [
    detect_passport_details,
    detect_vnz,
    detect_birth_date_tokens,
    detect_registration_dates,
    detect_temp_id,
    detect_card_holder_name,
    detect_code_words,
    detect_birth_certificate,
    detect_auto_data,
    detect_org_data,
    detect_passwords,
    detect_magstripe,
    detect_bank_names,
    detect_fio,
    detect_address,
    detect_birth_place,
    detect_citizenship,
]

ALL_DETECTORS = [predict_rules_one, *WEAK_DETECTORS]


# =========================
# Conflict resolver
# =========================

PRIORITY_RULE_CATS = {
    "Email",
    "CVV/CVC",
    "Дата окончания срока действия карты",
    "Номер карты",
    "Номер банковского счета",
    "Сведения об ИНН",
    "Номер телефона",
    "ПИН код",
    "СНИЛС клиента",
    "API ключи",
    "Разрешение на работу / визу",
    "Водительское удостоверение",
    "Серия и номер вида на жительство",
    "Одноразовые коды",
    "Содержимое магнитной полосы",
    "Наименование банка",
    "Временное удостоверение личности",
    "Имя держателя карты",
    "Кодовые слова",
    "Свидетельство о рождении",
    "Данные об автомобиле клиента",
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)",
}

CATEGORY_RANK = {
    "Временное удостоверение личности": 12,
    "Содержимое магнитной полосы": 11,
    "Свидетельство о рождении": 10,
    "Имя держателя карты": 9,
    "Кодовые слова": 8,
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)": 7,
}

def span_overlaps(first: Span, second: Span) -> bool:
    return not (first[1] <= second[0] or first[0] >= second[1])


def pick_better_span(current: Span, candidate: Span) -> Span:
    cur_start, cur_end, cur_cat = current
    new_start, new_end, new_cat = candidate

    cur_len = cur_end - cur_start
    new_len = new_end - new_start

    if new_cat in PRIORITY_RULE_CATS and cur_cat not in PRIORITY_RULE_CATS:
        return candidate
    if cur_cat in PRIORITY_RULE_CATS and new_cat not in PRIORITY_RULE_CATS:
        return current

    if CATEGORY_RANK.get(new_cat, 0) > CATEGORY_RANK.get(cur_cat, 0):
        return candidate
    if CATEGORY_RANK.get(cur_cat, 0) > CATEGORY_RANK.get(new_cat, 0):
        return current

    if new_len > cur_len:
        return candidate
    if cur_len > new_len:
        return current

    if new_start < cur_start:
        return candidate

    return current


def resolve_conflicts(spans: Sequence[Span]) -> List[Span]:
    if not spans:
        return []

    ordered = sorted(spans, key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    selected: List[Span] = []

    for candidate in ordered:
        replaced = False
        for idx, current in enumerate(selected):
            if span_overlaps(current, candidate):
                selected[idx] = pick_better_span(current, candidate)
                replaced = True
                break
        if not replaced:
            selected.append(candidate)

    selected = sorted(set(selected), key=lambda item: (item[0], item[1], item[2]))

    final: List[Span] = []
    for candidate in selected:
        if not final or not span_overlaps(final[-1], candidate):
            final.append(candidate)
        else:
            final[-1] = pick_better_span(final[-1], candidate)

    return sorted(set(final), key=lambda item: (item[0], item[1], item[2]))


def predict_one(text: str) -> List[Span]:
    spans: List[Span] = []
    for detector in ALL_DETECTORS:
        spans.extend(detector(text))
    return resolve_conflicts(spans)
