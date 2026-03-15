import pandas as pd
import ast
import numpy as np
import torch
import random
import os
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed # <--- ИМПОРТИРУЕМ ФУНКЦИЮ ДЛЯ СИДА
)
import evaluate

# --- КОНФИГУРАЦИЯ ---
MODEL_NAME = "DeepPavlov/rubert-base-cased"
DATA_PATH = "../data/train_dataset.tsv"
NEGATIVE_DATA_PATH = "../data/negative_train_dataset.tsv" # <--- ДОБАВЛЕН ПУТЬ К НЕГАТИВАМ
PSEUDO_DATA_PATH = "../data/pseudo_train_dataset.tsv"
OUTPUT_DIR = "../models/rubert_ner"
MAX_LEN = 256
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 5e-5

# МАГИЧЕСКОЕ ЧИСЛО (ЗЕРНО)
SEED = 42

def parse_targets(s):
    if pd.isna(s) or s == 'empty': return []
    try: return ast.literal_eval(s)
    except: return []

def resolve_overlaps(spans):
    """Оставляет самые длинные сущности при конфликте (пересечении) координат."""
    if len(spans) < 2: return spans
    # Сортируем по старту, а при равенстве - по убыванию длины
    spans = sorted(spans, key=lambda x: (x[0], -(x[1]-x[0])))
    resolved = []
    for span in spans:
        # Проверяем, не пересекается ли текущий спан с уже добавленными
        if not any(max(span[0], r[0]) < min(span[1], r[1]) for r in resolved):
            resolved.append(span)
    return resolved

# ВАЖНО: теперь функция принимает df (датафрейм), а не df_path (путь)
def prepare_data(df):
    print("Очистка и извлечение классов...")
    df['target_list'] = df['target'].apply(parse_targets)
    df['target_list'] = df['target_list'].apply(resolve_overlaps)
    
    # Собираем все уникальные классы для создания BIO-тегов
    unique_cats = set()
    for targets in df['target_list']:
        for t in targets:
            unique_cats.add(t[2])
            
    # Формируем словари (label2id и id2label)
    labels = ["O"]
    for cat in sorted(unique_cats):
        labels.extend([f"B-{cat}", f"I-{cat}"])
        
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    return df, label2id, id2label, labels

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, texts, targets, tokenizer, label2id, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        spans = self.targets[idx]
        
        # return_offsets_mapping=True - магия, которая связывает токены с символами
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        offsets = encoding.pop("offset_mapping").squeeze().tolist()
        item = {key: val.squeeze() for key, val in encoding.items()}
        
        # Инициализируем метки нулями (класс "O")
        labels = [self.label2id["O"]] * self.max_len
        
        # Расставляем BIO-теги
        for i, offset in enumerate(offsets):
            start_char, end_char = offset
            # Если токен специальный ([CLS], [SEP], [PAD]), ставим -100 (игнорируется лоссом)
            if start_char == 0 and end_char == 0:
                labels[i] = -100
                continue
                
            for span_start, span_end, category in spans:
                # Если токен полностью или частично попал внутрь спана сущности
                if start_char >= span_start and end_char <= span_end:
                    # Если это первый токен сущности - ставим B-
                    if start_char == span_start or labels[i-1] == self.label2id["O"] or labels[i-1] == -100:
                        labels[i] = self.label2id[f"B-{category}"]
                    # Иначе I- (продолжение)
                    else:
                        labels[i] = self.label2id[f"I-{category}"]
                    break # Сущность найдена, идем к следующему токену
                    
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item

def main():
    # === ГЛОБАЛЬНАЯ ФИКСАЦИЯ СИДА ===
    # Это сделает всё обучение полностью детерминированным
    set_seed(SEED)
    
    # 1. Загрузка данных (Основной + Негативный)
    print("Загрузка основного датасета...")
    df_main = pd.read_csv(DATA_PATH, sep='\t')
    
    print("Загрузка негативного датасета (Hard Negatives)...")
    try:
        df_neg = pd.read_csv(NEGATIVE_DATA_PATH, sep='\t')
    except FileNotFoundError:
        print(f"Файл {NEGATIVE_DATA_PATH} не найден. Убедитесь, что он существует.")
        return

    df_pseudo = pd.read_csv(PSEUDO_DATA_PATH, sep='\t')

    # Склеиваем и тщательно перемешиваем, чтобы негативы распределились равномерно
    df_combined = pd.concat([df_main, df_neg, df_pseudo], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Итоговый размер обучающей выборки: {len(df_combined)} текстов.")
    
    # Передаем склеенный датафрейм на обработку
    df, label2id, id2label, label_list = prepare_data(df_combined)
    print(f"Всего классов (с учетом B/I): {len(label2id)}")
    
    # 2. Разбиение на Train/Val (стратифицируем по наличию сущностей для баланса)
    df['has_entity'] = df['target_list'].apply(lambda x: len(x) > 0)
    # Тут тоже используем наш SEED для одинакового сплита. 
    # Stratify честно распределит наши пустые негативы (has_entity=False) в train и val
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df['has_entity'])
    
    # 3. Инициализация Токенизатора и Датасетов
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = NERDataset(train_df['text'].values, train_df['target_list'].values, tokenizer, label2id, MAX_LEN)
    val_dataset = NERDataset(val_df['text'].values, val_df['target_list'].values, tokenizer, label2id, MAX_LEN)
    
    # 4. Инициализация Модели
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
        ignore_mismatched_sizes=True
        
    )
    
    # 5. Метрики (используем seqeval для точного расчета F1 по сущностям)
    metric = evaluate.load("seqeval")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Убираем -100 (спец-токены) из расчетов
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        # Базовые (общие) метрики
        metrics_dict = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
        # Добавляем F1 для каждого класса в отдельности
        for key, value in results.items():
            if isinstance(value, dict) and "f1" in value:
                # key - это имя класса (например "Паспортные данные")
                metrics_dict[f"{key}_f1"] = value["f1"]
                
        return metrics_dict

    # 6. Настройки обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        
        # === ФИКСАЦИЯ СИДА В TRAINER ===
        seed=SEED,
        data_seed=SEED,
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none" # Отключаем wandb, если он не настроен
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # 7. Запуск!
    print("Начинаем обучение...")
    trainer.train()
    
    # 8. Сохранение лучшей модели
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    print(f"Модель успешно сохранена в {OUTPUT_DIR}/best_model")

if __name__ == "__main__":
    main()