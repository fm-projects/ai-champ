from collections import defaultdict
from typing import List, Tuple, Dict, Set
import ast
import glob
import pandas as pd
from collections import Counter
import numpy as np

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
    min_votes = int(n_models * threshold)
    
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

# ==================== СТРАТЕГИЯ 6: CATEGORY-SPECIFIC VOTING ====================
def blend_category_specific(submissions_dict: Dict[str, Dict]) -> Dict:
    """
    Разные пороги голосования для разных категорий.
    """
    # Более строгие пороги для критичных категорий
    category_thresholds = {
        'Паспортные данные': 0.6,
        'Номер карты': 0.6,
        'CVV/CVC': 0.8,
        'ПИН код': 0.8,
        'Пароли': 0.7,
        'API ключи': 0.7,
        'Содержимое магнитной полосы': 0.8,
    }
    default_threshold = 0.5
    
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
            threshold = category_thresholds.get(cat, default_threshold)
            if votes >= n_models * threshold:
                result.append(pred)
        
        blended[id_] = result
    
    return blended

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
    submissions = {x.split('/')[-1]: load_submission(x) for x in glob.glob('submits/*')}
    print(submissions)
    
    print("Запущен блендинг сабмитов...")
    
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
    
    # # Стратегия 5: Weighted Voting
    # weights = {x: 1}
    # blended6 = blend_weighted_voting(submissions, weights=weights)
    # save_submission(blended6, 'blended_weighted.csv')
    
    # Стратегия 6: Category-Specific
    blended7 = blend_category_specific(submissions)
    save_submission(blended7, 'blended_category_specific.csv')
    
    print("\nВсе стратегии блендинга выполнены!")
    print("Файлы сохранены: blended_*.csv")