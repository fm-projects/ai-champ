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


def apply_heuristics(text):
    """
    Используем высокоточные эвристики из baseline_utils и оставляем
    только целевые категории, которые хотим добавить в пайплайн.
    """
    selected = {
        "Email",
        "Дата окончания срока действия карты",
        "CVV/CVC",
        "ПИН код",
        "Номер карты",
        "Номер банковского счета",
        "Водительское удостоверение",
        "Серия и номер вида на жительство",
        "Паспортные данные",
        "Дата рождения",
        "Дата регистрации по месту жительства или пребывания",
    }

    spans = []

    # Базовые правила
    try:
        spans.extend(rules_predict(text))
    except Exception:
        pass

    # Усиление по паспорту
    if detect_passport_details is not None:
        try:
            spans.extend(detect_passport_details(text))
        except Exception:
            pass

    # Усиление по ВНЖ
    if detect_vnz is not None:
        try:
            spans.extend(detect_vnz(text))
        except Exception:
            pass

    # Усиление по дате рождения
    if detect_birth_date_tokens is not None:
        try:
            spans.extend(detect_birth_date_tokens(text))
        except Exception:
            pass

    # Оставляем только нужные категории
    spans = [s for s in spans if len(s) == 3 and s[2] in selected]

    # Убираем дубликаты
    uniq = {}
    for s in spans:
        uniq[(s[0], s[1], s[2])] = s

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


def resolve_conflicts(bert_entities, heuristic_entities):
    """
    Объединяет предсказания нейросети и эвристик.
    При пересечении приоритет у эвристик.
    """
    final_entities = heuristic_entities.copy()

    for bert_ent in bert_entities:
        b_start, b_end, _ = bert_ent
        has_overlap = False

        for h_ent in heuristic_entities:
            h_start, h_end, _ = h_ent
            if max(b_start, h_start) < min(b_end, h_end):
                has_overlap = True
                break

        if not has_overlap:
            final_entities.append(bert_ent)

    if not final_entities:
        return "[]"

    final_entities = sorted(final_entities, key=lambda x: x[0])
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