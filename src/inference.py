import pandas as pd
import torch
import ast
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm

from baseline_utils import predict_rules_one as rules_predict
# Импортируем метрики коллеги для оценки
from baseline_utils import strict_micro_f1, per_category_f1 

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
TRAIN_PATH = "../data/train_dataset.tsv"  # Путь к трейну для оценки
OUTPUT_PATH = "../submission.csv"
MAX_LEN = 256
BATCH_SIZE = 32

EVALUATE_ON_TRAIN = True # Переключатель: считать ли метрики перед инференсом


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
    # "Паспортные данные",
    # "Дата рождения",
    # "Дата регистрации по месту жительства или пребывания",
    # "API ключи"
}

ENABLE_BASE_RULES = False            # базовые регулярки
ENABLE_PASSPORT_HEURISTIC = False    # доп паспорт
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


def clean_boundaries(text: str, start: int, end: int) -> tuple:
    """Отщипывает знаки препинания и пробелы на краях найденных сущностей."""
    bad_chars = ' \t\n\r.,;:"\'«»()[]{}'
    
    # Сдвигаем левую границу вправо, если там мусор
    while start < end and text[start] in bad_chars:
        start += 1
    # Сдвигаем правую границу влево, если там мусор
    while end > start and text[end - 1] in bad_chars:
        end -= 1
        
    return start, end


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
    Объединение: BERT-спан выкидываем только если он пересекается
    с ЭВРИСТИКОЙ ТОГО ЖЕ КЛАССА.
    ВОЗВРАЩАЕТ СПИСОК (List[tuple]), А НЕ СТРОКУ!
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
    final_entities = sorted(final_entities, key=lambda x: (x[0], x[1], x[2]))
    
    return final_entities


def evaluate_on_train(model, tokenizer, id2label, device):
    """
    Прогоняет трейновый датасет через пайплайн и считает метрики.
    """
    print(f"\n[{'='*40}]\nЗАПУСК ОЦЕНКИ НА TRAIN ДАТАСЕТЕ\n[{'='*40}]")
    try:
        train_df = pd.read_csv(TRAIN_PATH, sep='\t')
    except Exception as e:
        print(f"Ошибка загрузки {TRAIN_PATH}: {e}")
        return

    def parse_target(val):
        if pd.isna(val) or val == 'empty': return []
        try: return ast.literal_eval(val)
        except: return []
        
    train_df['target_list'] = train_df['target'].apply(parse_target)
    
    texts = train_df["text"].astype(str).tolist()
    y_true = train_df["target_list"].tolist()
    y_pred = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Оценка (Train)"):
        batch_texts = texts[i: i + BATCH_SIZE]
        bert_preds = predict_ner(batch_texts, model, tokenizer, id2label, device)

        for j, text in enumerate(batch_texts):
            heur_preds = apply_heuristics(text)
            final_pred_list = resolve_conflicts(bert_preds[j], heur_preds)
            y_pred.append(final_pred_list)

    # Считаем общие метрики
    overall = strict_micro_f1(y_true, y_pred)
    print("\n--- ОБЩИЕ МЕТРИКИ (Strict Span) ---")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1-score:  {overall['f1']:.4f}")
    print("-" * 35)

    # Считаем метрики по классам
    print("\n--- МЕТРИКИ ПО КЛАССАМ ---")
    print(f"{'Класс':<60} | {'F1':<6} | {'Support'}")
    print("-" * 80)
    cat_metrics = per_category_f1(y_true, y_pred)
    for cat, f1, support in cat_metrics:
        print(f"{cat:<60} | {f1:.4f} | {support}")
    print(f"[{'='*40}]\n")


def main():
    print("Загрузка модели...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(device)
    id2label = model.config.id2label

    # Оценка на Train
    if EVALUATE_ON_TRAIN:
        evaluate_on_train(model, tokenizer, id2label, device)

    # Инференс на Test
    print("Загрузка тестовых данных...")
    test_df = pd.read_csv(TEST_PATH)
    predictions = []
    texts = test_df["text"].astype(str).tolist()

    print(f"Начинаем инференс ({len(texts)} текстов)...")

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Инференс (Test)"):
        batch_texts = texts[i: i + BATCH_SIZE]

        # 1. Предсказания BERT
        bert_preds = predict_ner(batch_texts, model, tokenizer, id2label, device)

        # 2. Эвристики + merge
        for j, text in enumerate(batch_texts):
            heur_preds = apply_heuristics(text)
            final_pred_list = resolve_conflicts(bert_preds[j], heur_preds)
            
            # В CSV мы должны писать строковое представление (даже если пусто "[]")
            if not final_pred_list:
                predictions.append("[]")
            else:
                predictions.append(str(final_pred_list))

    sub_df = pd.DataFrame({
        "id": test_df["id"],
        "Prediction": predictions
    })

    sub_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nГотово! Файл для отправки сохранен в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
