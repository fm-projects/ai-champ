from collections import defaultdict
from typing import List, Tuple, Dict, Set
import ast
import glob
import math
import os
import pandas as pd
from collections import Counter
import numpy as np

SUBMISSION_SCORES = {
    'submission (61).csv': 0.97522,
    'submission (62).csv': 0.97369,
    'submission (68).csv': 0.97552,
    'submission (70).csv': 0.97395,
    'submission (71).csv': 0.97700,
    'submission (74).csv': 0.97569,
    'submission (76).csv': 0.97700,
    'submission (77).csv': 0.97689,
    'submission (78).csv': 0.97808,
    'submission (79).csv': 0.97465,
    'submission (80).csv': 0.97288,
    'submission_ruro.csv': 0.96885,
}

HISTORICAL_TOP7 = [
    'submission (61).csv',
    'submission (68).csv',
    'submission (71).csv',
    'submission (74).csv',
    'submission (76).csv',
    'submission (77).csv',
    'submission (78).csv',
]

CATEGORY_THRESHOLDS = {
    'Паспортные данные': 0.6,
    'Номер карты': 0.62,
    'CVV/CVC': 0.8,
    'ПИН код': 0.8,
    'Пароли': 0.72,
    'API ключи': 0.72,
    'Содержимое магнитной полосы': 0.82,
}

DEFAULT_CATEGORY_THRESHOLD = 0.5

def load_submission(filepath: str) -> Dict[int, List[Tuple]]:
    """Загрузка сабмита"""
    df = pd.read_csv(filepath)
    submissions = {}
    for _, row in df.iterrows():
        pred_id = int(row['id'])
        try:
            if pd.isna(row['Prediction']) or row['Prediction'] == '[]':
                submissions[pred_id] = []
            else:
                submissions[pred_id] = ast.literal_eval(row['Prediction'])
        except:
            submissions[pred_id] = []
    return submissions

# ==================== СТРАТЕГИЯ 1: MAJORITY VOTING ====================
def blend_majority_voting(submissions_dict: Dict[str, Dict], threshold: float = 0.5) -> Dict:
    """
    Блендинг через голосование большинства.
    Сущность включается если её поддерживают >= threshold доля моделей.
    """
    all_ids = set()
    for sub in submissions_dict.values():
        all_ids.update(sub.keys())
    
    n_models = len(submissions_dict)
    min_votes = math.ceil(n_models * threshold)
    
    blended = {}
    for id_ in all_ids:
        vote_counter = defaultdict(int)
        
        for sub in submissions_dict.values():
            for pred in sub.get(id_, []):
                vote_counter[pred] += 1
        
        blended[id_] = [pred for pred, votes in vote_counter.items() if votes >= min_votes]
    
    return blended

def blend_majority_by_votes(submissions_dict: Dict[str, Dict], min_votes: int) -> Dict:
    """
    Majority voting с фиксированным числом голосов.
    Нужен для воспроизведения исторически удачных схем вроде 4-of-7.
    """
    all_ids = set()
    for sub in submissions_dict.values():
        all_ids.update(sub.keys())
    
    blended = {}
    for id_ in all_ids:
        vote_counter = defaultdict(int)
        for sub in submissions_dict.values():
            for pred in sub.get(id_, []):
                vote_counter[pred] += 1
        blended[id_] = [pred for pred, votes in vote_counter.items() if votes >= min_votes]
    
    return blended

# ==================== СТРАТЕГИЯ 2: UNION (Объединение) ====================
def blend_union(submissions_dict: Dict[str, Dict]) -> Dict:
    """
    Объединение всех предсказаний (максимальный recall).
    """
    all_ids = set()
    for sub in submissions_dict.values():
        all_ids.update(sub.keys())
    
    blended = {}
    for id_ in all_ids:
        all_preds = set()
        for sub in submissions_dict.values():
            all_preds.update(sub.get(id_, []))
        blended[id_] = list(all_preds)
    
    return blended

# ==================== СТРАТЕГИЯ 3: INTERSECTION (Пересечение) ====================
def blend_intersection(submissions_dict: Dict[str, Dict]) -> Dict:
    """
    Пересечение всех предсказаний (максимальная precision).
    """
    all_ids = set()
    for sub in submissions_dict.values():
        all_ids.update(sub.keys())
    
    n_models = len(submissions_dict)
    blended = {}
    for id_ in all_ids:
        pred_sets = [set(sub.get(id_, [])) for sub in submissions_dict.values()]
        blended[id_] = list(set.intersection(*pred_sets) if pred_sets else set())
    
    return blended

# ==================== СТРАТЕГИЯ 4: SPAN OVERLAP MERGE ====================
def blend_span_overlap(submissions_dict: Dict[str, Dict], overlap_threshold: float = 0.5) -> Dict:
    """
    Объединение перекрывающихся спанов с голосованием по категории.
    """
    all_ids = set()
    for sub in submissions_dict.values():
        all_ids.update(sub.keys())
    
    def spans_overlap(s1, e1, s2, e2):
        """Проверка перекрытия спанов"""
        intersection = max(0, min(e1, e2) - max(s1, s2))
        union = (e1 - s1) + (e2 - s2) - intersection
        return intersection / union if union > 0 else 0
    
    blended = {}
    for id_ in all_ids:
        all_preds = []
        for sub in submissions_dict.values():
            all_preds.extend(sub.get(id_, []))
        
        if not all_preds:
            blended[id_] = []
            continue
        
        # Группируем перекрывающиеся спаны
        merged = []
        used = [False] * len(all_preds)
        
        for i, (s1, e1, c1) in enumerate(all_preds):
            if used[i]:
                continue
            
            group = [(s1, e1, c1)]
            used[i] = True
            
            for j, (s2, e2, c2) in enumerate(all_preds):
                if not used[j] and spans_overlap(s1, e1, s2, e2) >= overlap_threshold:
                    group.append((s2, e2, c2))
                    used[j] = True
            
            # Выбираем наиболее частую категорию и усредняем границы
            categories = Counter([g[2] for g in group])
            best_cat = categories.most_common(1)[0][0]
            avg_start = int(np.mean([g[0] for g in group]))
            avg_end = int(np.mean([g[1] for g in group]))
            
            merged.append((avg_start, avg_end, best_cat))
        
        blended[id_] = merged
    
    return blended

# ==================== СТРАТЕГИЯ 5: WEIGHTED VOTING ====================
def blend_weighted_voting(submissions_dict: Dict[str, Dict], 
                          weights: Dict[str, float] = None) -> Dict:
    """
    Взвешенное голосование (можно задать веса для каждой модели).
    """
    if weights is None:
        weights = {name: 1.0 for name in submissions_dict.keys()}
    
    all_ids = set()
    for sub in submissions_dict.values():
        all_ids.update(sub.keys())
    
    blended = {}
    for id_ in all_ids:
        score_counter = defaultdict(float)
        
        for name, sub in submissions_dict.items():
            weight = weights.get(name, 1.0)
            for pred in sub.get(id_, []):
                score_counter[pred] += weight
        
        # Включаем предсказания сscore > 0.5 * total_weight
        total_weight = sum(weights.values())
        blended[id_] = [pred for pred, score in score_counter.items() 
                       if score >= total_weight * 0.5]
    
    return blended

def build_score_weights(
    submissions_dict: Dict[str, Dict],
    scores: Dict[str, float],
    scale: float = 400.0,
) -> Dict[str, float]:
    """
    Преобразует leaderboard score в веса.
    Экспонента сильно штрафует слабые сабмиты и почти обнуляет хвост.
    """
    available_scores = {name: scores[name] for name in submissions_dict if name in scores}
    if not available_scores:
        return {name: 1.0 for name in submissions_dict}
    
    min_score = min(available_scores.values())
    weights = {}
    for name in submissions_dict:
        score = scores.get(name, min_score)
        weights[name] = math.exp((score - min_score) * scale)
    return weights

def blend_weighted_category_specific(
    submissions_dict: Dict[str, Dict],
    weights: Dict[str, float],
    category_thresholds: Dict[str, float] = None,
    default_threshold: float = DEFAULT_CATEGORY_THRESHOLD,
) -> Dict:
    """
    Взвешенное голосование с разными порогами по категориям.
    """
    if category_thresholds is None:
        category_thresholds = CATEGORY_THRESHOLDS
    
    all_ids = set()
    for sub in submissions_dict.values():
        all_ids.update(sub.keys())
    
    total_weight = sum(weights.values())
    blended = {}
    for id_ in all_ids:
        score_counter = defaultdict(float)
        category_map = {}
        
        for name, sub in submissions_dict.items():
            weight = weights.get(name, 1.0)
            for pred in sub.get(id_, []):
                score_counter[pred] += weight
                category_map[pred] = pred[2]
        
        result = []
        for pred, score in score_counter.items():
            cat = category_map[pred]
            threshold = category_thresholds.get(cat, default_threshold)
            if score >= total_weight * threshold:
                result.append(pred)
        
        blended[id_] = result
    
    return blended

# ==================== СТРАТЕГИЯ 6: CATEGORY-SPECIFIC VOTING ====================
def blend_category_specific(submissions_dict: Dict[str, Dict]) -> Dict:
    """
    Разные пороги голосования для разных категорий.
    """
    all_ids = set()
    for sub in submissions_dict.values():
        all_ids.update(sub.keys())
    
    n_models = len(submissions_dict)
    blended = {}
    
    for id_ in all_ids:
        vote_counter = defaultdict(int)
        category_map = {}
        
        for sub in submissions_dict.values():
            for pred in sub.get(id_, []):
                vote_counter[pred] += 1
                category_map[pred] = pred[2]
        
        result = []
        for pred, votes in vote_counter.items():
            cat = category_map[pred]
            threshold = CATEGORY_THRESHOLDS.get(cat, DEFAULT_CATEGORY_THRESHOLD)
            if votes >= n_models * threshold:
                result.append(pred)
        
        blended[id_] = result
    
    return blended

def select_submissions(
    submissions_dict: Dict[str, Dict],
    names: List[str],
) -> Dict[str, Dict]:
    """Выбирает подмножество сабмитов по именам."""
    return {name: submissions_dict[name] for name in names if name in submissions_dict}

def blend_from_existing_files(filepaths: List[str], min_votes: int) -> Dict:
    """
    Блендинг уже готовых сабмитов/блендов из csv-файлов.
    """
    loaded = {
        os.path.basename(path): load_submission(path)
        for path in filepaths
        if os.path.exists(path)
    }
    return blend_majority_by_votes(loaded, min_votes=min_votes)

# ==================== СОХРАНЕНИЕ РЕЗУЛЬТАТА ====================
def save_submission(blended: Dict, filepath: str) -> None:
    """Сохранение блендинга в формате сабмита"""
    rows = []
    for id_ in sorted(blended.keys()):
        preds = blended[id_]
        rows.append({
            'id': id_,
            'Prediction': str(preds) if preds else '[]'
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Сохранено {len(df)} строк в {filepath}")

# ==================== ЗАПУСК БЛЕНДИНГА ====================
if __name__ == "__main__":
    # Загрузка всех сабмитов
    submissions = {
        os.path.basename(x): load_submission(x)
        for x in sorted(glob.glob('submits/*.csv'))
    }
    
    print("Запущен блендинг сабмитов...")
    print(f"Найдено {len(submissions)} сабмитов: {', '.join(submissions.keys())}")
    
    score_weights = build_score_weights(submissions, SUBMISSION_SCORES, scale=400.0)
    print("Веса по score:")
    for name, weight in sorted(score_weights.items(), key=lambda item: item[1], reverse=True):
        score = SUBMISSION_SCORES.get(name)
        print(f"  {name}: score={score}, weight={weight:.4f}")
    
    # Стратегия 1: Majority Voting (50%)
    blended1 = blend_majority_voting(submissions, threshold=0.5)
    save_submission(blended1, 'blended_majority_50.csv')
    
    # Стратегия 1b: Majority Voting (60%)
    blended2 = blend_majority_voting(submissions, threshold=0.6)
    save_submission(blended2, 'blended_majority_60.csv')
    
    # Стратегия 2: Union
    blended3 = blend_union(submissions)
    save_submission(blended3, 'blended_union.csv')
    
    # Стратегия 3: Intersection
    blended4 = blend_intersection(submissions)
    save_submission(blended4, 'blended_intersection.csv')
    
    # Стратегия 4: Span Overlap
    blended5 = blend_span_overlap(submissions, overlap_threshold=0.5)
    save_submission(blended5, 'blended_span_overlap.csv')
    
    # Стратегия 5a: Weighted Voting по leaderboard score
    blended6 = blend_weighted_voting(submissions, weights=score_weights)
    save_submission(blended6, 'blended_weighted_score.csv')
    
    # Стратегия 5b: Более строгий score-weighted voting
    blended6_strict = {}
    total_weight = sum(score_weights.values())
    all_ids = set()
    for sub in submissions.values():
        all_ids.update(sub.keys())
    for id_ in all_ids:
        score_counter = defaultdict(float)
        for name, sub in submissions.items():
            weight = score_weights.get(name, 1.0)
            for pred in sub.get(id_, []):
                score_counter[pred] += weight
        blended6_strict[id_] = [
            pred for pred, score in score_counter.items()
            if score >= total_weight * 0.58
        ]
    save_submission(blended6_strict, 'blended_weighted_score_strict.csv')
    
    # Стратегия 6: Category-Specific
    blended7 = blend_category_specific(submissions)
    save_submission(blended7, 'blended_category_specific.csv')
    
    # Стратегия 7: Weighted Category-Specific
    blended8 = blend_weighted_category_specific(submissions, weights=score_weights)
    save_submission(blended8, 'blended_weighted_category_specific.csv')
    
    historical_top7 = select_submissions(submissions, HISTORICAL_TOP7)
    if len(historical_top7) == len(HISTORICAL_TOP7):
        print("\nГенерирую точечные кандидаты на сильном историческом подмножестве...")
        
        # Исторически лучший режим: 4-of-7.
        blended_top7_majority = blend_majority_by_votes(historical_top7, min_votes=4)
        save_submission(blended_top7_majority, 'blended_top7_historical_4of7.csv')
        
        # Для этого набора category_specific даёт тот же размер, но оставим отдельным кандидатом.
        blended_top7_category = blend_category_specific(historical_top7)
        save_submission(blended_top7_category, 'blended_top7_category_specific.csv')
        
        top8_with_79 = select_submissions(submissions, HISTORICAL_TOP7 + ['submission (79).csv'])
        if len(top8_with_79) == 8:
            blended_top8_with_79 = blend_majority_by_votes(top8_with_79, min_votes=4)
            save_submission(blended_top8_with_79, 'blended_top8_add79_4of8.csv')
        
        top8_with_62 = select_submissions(submissions, HISTORICAL_TOP7 + ['submission (62).csv'])
        if len(top8_with_62) == 8:
            blended_top8_with_62 = blend_majority_by_votes(top8_with_62, min_votes=4)
            save_submission(blended_top8_with_62, 'blended_top8_add62_4of8.csv')
        
        top8_with_80 = select_submissions(submissions, HISTORICAL_TOP7 + ['submission (80).csv'])
        if len(top8_with_80) == 8:
            blended_top8_with_80 = blend_majority_by_votes(top8_with_80, min_votes=4)
            save_submission(blended_top8_with_80, 'blended_top8_add80_4of8.csv')
    
    # Лучший найденный кластер вокруг submission (69).csv
    best_cluster = select_submissions(
        submissions,
        [
            'submission (61).csv',
            'submission (68).csv',
            'submission (69).csv',
            'submission (71).csv',
            'submission (74).csv',
            'submission (76).csv',
            'submission (77).csv',
            'submission (78).csv',
            'submission (80).csv',
        ],
    )
    if len(best_cluster) == 9:
        blended_top9_add69_80_5of9 = blend_majority_by_votes(best_cluster, min_votes=5)
        save_submission(blended_top9_add69_80_5of9, 'blended_top9_add69_80_5of9.csv')
        
        top9_add69_ruro = select_submissions(
            submissions,
            [
                'submission (61).csv',
                'submission (68).csv',
                'submission (69).csv',
                'submission (71).csv',
                'submission (74).csv',
                'submission (76).csv',
                'submission (77).csv',
                'submission (78).csv',
                'submission_ruro.csv',
            ],
        )
        if len(top9_add69_ruro) == 9:
            blended_top9_add69_ruro_5of9 = blend_majority_by_votes(top9_add69_ruro, min_votes=5)
            save_submission(blended_top9_add69_ruro_5of9, 'blended_top9_add69_ruro_5of9.csv')
        
        top9_add69_70 = select_submissions(
            submissions,
            [
                'submission (61).csv',
                'submission (68).csv',
                'submission (69).csv',
                'submission (70).csv',
                'submission (71).csv',
                'submission (74).csv',
                'submission (76).csv',
                'submission (77).csv',
                'submission (78).csv',
            ],
        )
        if len(top9_add69_70) == 9:
            blended_top9_add69_70_5of9 = blend_majority_by_votes(top9_add69_70, min_votes=5)
            save_submission(blended_top9_add69_70_5of9, 'blended_top9_add69_70_5of9.csv')
        
        top8_add69 = select_submissions(
            submissions,
            [
                'submission (61).csv',
                'submission (68).csv',
                'submission (69).csv',
                'submission (71).csv',
                'submission (74).csv',
                'submission (76).csv',
                'submission (77).csv',
                'submission (78).csv',
            ],
        )
        if len(top8_add69) == 8:
            blended_top8_add69_category = blend_category_specific(top8_add69)
            save_submission(blended_top8_add69_category, 'blended_top8_add69_category_specific.csv')
        
        best3_inputs = [
            'blended_top9_add69_80_5of9.csv',
            'blended_top9_add69_ruro_5of9.csv',
            'blended_top9_add69_70_5of9.csv',
        ]
        if all(os.path.exists(path) for path in best3_inputs):
            blended_best3_majority_2of3 = blend_from_existing_files(best3_inputs, min_votes=2)
            save_submission(blended_best3_majority_2of3, 'blended_best3_majority_2of3.csv')
        
        best4_inputs = best3_inputs + ['blended_top8_add69_category_specific.csv']
        if all(os.path.exists(path) for path in best4_inputs):
            blended_best4_majority_3of4 = blend_from_existing_files(best4_inputs, min_votes=3)
            save_submission(blended_best4_majority_3of4, 'blended_best4_majority_3of4.csv')
    
    print("\nВсе стратегии блендинга выполнены!")
    print("Файлы сохранены: blended_*.csv")
