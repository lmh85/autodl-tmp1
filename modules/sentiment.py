# src/modules/sentiment.py
"""
情感连续化模块：将文本情感从分类标签转为连续分值 e_τ ∈ [-1, 1]
支持两种后端：
  - 'transformers'（默认，精度高）
  - 'textblob'（轻量，无需 GPU）
"""

import torch


class SentimentAnalyzer:
    def __init__(self, backend="transformers", device=None):
        """
        backend: "transformers" | "textblob"
        device:  torch.device，默认自动检测
        """
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None  # lazy init

    def _load_model(self):
        if self._model is not None:
            return
        if self.backend == "transformers":
            from transformers import pipeline
            # 使用轻量级 distilbert 模型，避免显存压力
            self._model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" else -1,
            )
        elif self.backend == "textblob":
            from textblob import TextBlob
            self._model = "textblob"
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def score(self, text: str) -> float:
        """
        输入：用户话语文本
        输出：e_τ ∈ [-1.0, 1.0]
          正值 = 正向情感（喜欢）
          负值 = 负向情感（不喜欢）
          0    = 中性
        """
        if not text or not text.strip():
            return 0.0

        self._load_model()

        if self.backend == "transformers":
            result = self._model(text[:512])[0]  # 截断防止超长
            raw_score = float(result["score"])   # 置信度 ∈ (0, 1]
            # POSITIVE → 正值，NEGATIVE → 负值
            if result["label"] == "NEGATIVE":
                raw_score = -raw_score
            return round(raw_score, 4)

        elif self.backend == "textblob":
            from textblob import TextBlob
            polarity = TextBlob(text).sentiment.polarity  # ∈ [-1, 1]
            return round(float(polarity), 4)

    def batch_score(self, texts: list) -> list:
        """批量打分，适合训练阶段"""
        return [self.score(t) for t in texts]


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = SentimentAnalyzer(backend="textblob")
    tests = [
        "I really loved that movie, it was fantastic!",
        "The film was terrible and boring.",
        "It was okay, nothing special.",
    ]
    for t in tests:
        print(f"[{analyzer.score(t):+.3f}] {t}")