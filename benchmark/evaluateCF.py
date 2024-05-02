from surprise import Reader, Dataset, KNNBasic, SVD, NMF
from surprise.model_selection import GridSearchCV, cross_validate, KFold

import numpy as np
from sklearn.metrics import ndcg_score

def find_best_params(data, param_grid, algo=KNNBasic):
 
    gs = GridSearchCV(algo, measures=['RMSE'], param_grid=param_grid)
    gs.fit(data)
    return gs.best_score['rmse'], gs.best_params['rmse']



def get_ndcg(surprise_predictions, k_highest_scores=None):
    uids = [p.uid for p in surprise_predictions]
    iids = [p.iid for p in surprise_predictions]
    r_uis = [p.r_ui for p in surprise_predictions]
    ests = [p.est for p in surprise_predictions]

    # Construct 2D arrays from the lists
    true_relevance = np.array([r_uis])
    predicted_scores = np.array([ests])

    # Calculate NDCG@k
    return ndcg_score(y_true=true_relevance, y_score=predicted_scores, k=k_highest_scores)