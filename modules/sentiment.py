class SentimentAnalyzer:
    def __init__(self):
        pass

    def score(self, text):
        from transformers import pipeline
        clf = pipeline("sentiment-analysis")

        result = clf(text)[0]
        score = result["score"]

        if result["label"] == "NEGATIVE":
            score = -score

        return score