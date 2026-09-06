TARGETS = [
        'math_score',
        'history_score',
        'physics_score',
        'chemistry_score',
        'biology_score',
        'english_score',
        'geography_score',
    ]

BEST_MODEL_PARAMS = {
    "random_forest": {
    "model__n_estimators": 200,
    "model__max_depth": 10,
    "model__min_samples_split": 5,
    "model__min_samples_leaf": 4,
    "model__max_features": "log2"
    }
}