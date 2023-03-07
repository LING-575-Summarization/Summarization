from .lexrank import *

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from evaluate import load

class LexrankClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            vector_type: str,
            threshold: Optional[float] = 0.,
            error: Optional[float] = 1e-16,
            dampening_factor: Optional[float] = 0.15,
            min_length: Optional[int] = None, 
            min_jaccard_dist: Optional[float] = None
    ) -> None:
        self.vector_type = vector_type
        self.threshold = threshold
        self.error = error
        self.dampening_factor = dampening_factor
        self.min_length = min_length
        self.min_jaccard_dist = min_jaccard_dist

    def fit(self, docs: Tuple["docset", "indices"], **kwargs):
        print("Fitting...")
        LexRank = LexRankFactory(self.vector_type)
        self.lexrank_driver = LexRank(docs[0], docs[1], **kwargs)
        self._X = docs[0]
        self.indices = docs[1]
        return self
    
    def predict(self, X: Tuple["docset", "indices"]):
        check_is_fitted(self)
        print("Predicting...")
        self.lexrank_driver.replace_evaldocs(X[0], X[1])
        return self.lexrank_driver.obtain_summary()

    def score(self, X: Tuple["docset", "indices"], y: List["summaries"]):
        print("Scoring...")
        if not hasattr(self, 'rouge_score'):
            self.rouge_score = load('rouge')
        summary = self.predict(X)
        all_scores = []
        for gold_sum in y:
            all_scores.append(
                self.rouge_score(predictions=summary, references=gold_sum)
            )
        rouge1 = [x['rouge1'] for x in all_scores]
        return np.mean(rouge1)

        