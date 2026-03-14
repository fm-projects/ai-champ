import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm

from baseline_utils import predict_rules_one as rules_predict

try:
    from baseline_utils import detect_passport_details  # type: ignore
except Exception:
    detect_passport_details = None

try:
    from baseline_utils import detect_vnz  # type: ignore
except Exception:
    detect_vnz = None

try:
    from baseline_utils import detect_birth_date_tokens  # type: ignore
except Exception:
    detect_birth_date_tokens = None

# --- КОНФИГУРАЦИЯ ---
MODEL_PATH = "../models/rubert_ner/best_model"
TEST_PATH = "../data/private_test_dataset.csv"
OUTPUT_PATH = "../submission.csv"
MAX_LEN = 256
BATCH_SIZE = 32


# --- НАСТРОЙКИ ЭВРИСТИК ---

HEURISTIC_TARGET_WHITELIST = {
    # "Email",
    # "Дата окончания срока действия карты",
    # "CVV/CVC",
    # "ПИН код",
    # "Номер карты",
    # "Номер банковского счета",
    # "Водительское удостоверение",
    # "Серия и номер вида на жительство",
    "Паспортные данные",
    # "Дата рождения",
    # "Дата регистрации по месту жительства или пребывания",
    "API ключи",
    "Данные об автомобиле клиента",
}

ENABLE_BASE_RULES = True            # базовые регулярки
ENABLE_PASSPORT_HEURISTIC = True    # доп паспорт
ENABLE_VNZ_HEURISTIC = False         # доп "Серия и номер вида на жительство"
ENABLE_BIRTH_DATE_HEURISTIC = False  # доп с датами рождения


def _safe_collect_spans(detector_fn, text):
    """
    Безопасно вызывает эвристику и возвращает список span.
    """
    if detector_fn is None:
        return []

    try:
        spans = detector_fn(text)
        if not spans:
            return []
        return [s for s in spans if isinstance(s, (list, tuple)) and len(s) == 3]
    except Exception:
        return []


def apply_heuristics(text):
    """
    Применяет включенные эвристики и оставляет только категории из whitelist.
    
    Управление:
    - HEURISTIC_TARGET_WHITELIST: какие классы вообще разрешены
    - ENABLE_*: какие эвристики реально включены
    """
    spans = []

    heuristic_sources = [
        (ENABLE_BASE_RULES, rules_predict),
        (ENABLE_PASSPORT_HEURISTIC, detect_passport_details),
        (ENABLE_VNZ_HEURISTIC, detect_vnz),
        (ENABLE_BIRTH_DATE_HEURISTIC, detect_birth_date_tokens),
    ]

    for is_enabled, detector_fn in heuristic_sources:
        if is_enabled:
            spans.extend(_safe_collect_spans(detector_fn, text))

    # оставляем только разрешенные категории
    spans = [s for s in spans if s[2] in HEURISTIC_TARGET_WHITELIST]

    # дедупликация
    uniq = {}
    for start, end, label in spans:
        uniq[(start, end, label)] = (start, end, label)

    return list(uniq.values())


def predict_ner(texts, model, tokenizer, id2label, device):
    """
    Прогоняет батч текстов через BERT и склеивает токены обратно в символы.
    """
    encodings = tokenizer(
        texts,
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offsets = encodings.pop("offset_mapping").cpu().numpy()
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=2).cpu().numpy()

    batch_entities = []

    for i, seq_preds in enumerate(preds):
        entities = []
        current_entity = None

        for j, pred in enumerate(seq_preds):
            label = id2label[pred]
            start_char, end_char = offsets[i][j]
            start_char, end_char = int(start_char), int(end_char)

            # Пропускаем спецтокены
            if start_char == 0 and end_char == 0:
                continue

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                category = label[2:]
                current_entity = (start_char, end_char, category)

            elif label.startswith("I-"):
                category = label[2:]
                if current_entity and current_entity[2] == category:
                    current_entity = (current_entity[0], end_char, category)
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = (start_char, end_char, category)

            elif label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        batch_entities.append(entities)

    return batch_entities


def spans_overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])

def resolve_conflicts(bert_entities, heuristic_entities):
    """
    Более аккуратное объединение:
    1. эвристики добавляем сразу
    2. BERT-спан выкидываем только если он пересекается
       с ЭВРИСТИКОЙ ТОГО ЖЕ КЛАССА
    3. если классы разные, оставляем оба
    """
    final_entities = heuristic_entities.copy()

    for bert_ent in bert_entities:
        b_start, b_end, b_label = bert_ent
        drop_bert = False

        for h_ent in heuristic_entities:
            h_start, h_end, h_label = h_ent

            if spans_overlap(bert_ent, h_ent):
                # выбиваем BERT только если класс тот же
                if b_label == h_label:
                    drop_bert = True
                    break

        if not drop_bert:
            final_entities.append(bert_ent)

    # дедупликация
    final_entities = list(dict.fromkeys(final_entities))

    if not final_entities:
        return "[]"

    final_entities = sorted(final_entities, key=lambda x: (x[0], x[1], x[2]))
    return str(final_entities)


def main():
    print("Загрузка данных...")
    test_df = pd.read_csv(TEST_PATH)

    print("Загрузка модели...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(device)

    id2label = model.config.id2label

    predictions = []
    texts = test_df["text"].astype(str).tolist()

    print(f"Начинаем инференс ({len(texts)} текстов)...")

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i: i + BATCH_SIZE]

        # 1. Предсказания BERT
        bert_preds = predict_ner(batch_texts, model, tokenizer, id2label, device)

        # 2. Эвристики + merge
        for j, text in enumerate(batch_texts):
            heur_preds = apply_heuristics(text)
            final_pred_str = resolve_conflicts(bert_preds[j], heur_preds)
            predictions.append(final_pred_str)

    sub_df = pd.DataFrame({
        "id": test_df["id"],
        "Prediction": predictions
    })

    sub_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nГотово! Файл для отправки сохранен в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()