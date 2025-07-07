from utils.config_manager import get_config
from utils.logger import get_logger
from ai.reinforcement_learning import RLTrainer
from ai.deep_learning_trainer import DeepLearningTrainer
from ai.simple_traditional_ml import TraditionalMLTrainer

class ModelManager:
    """
    จัดการการฝึกและเรียกใช้งานโมเดล RL, DL, และ Traditional ML
    """
    def __init__(self):
        cfg = get_config()
        self.logger = get_logger()
        self.rl_cfg = cfg.get('ai_models.rl')
        self.dl_cfg = cfg.get('ai_models.deep_learning')
        self.ml_cfg = cfg.get('ai_models.traditional_ml')
        self.ensemble_weights = cfg.get('ai_models.ensemble_weights', [1.0])

    def train_all(self, data: dict[str, any]) -> dict:
        self.logger.info("Training all models (RL, DL, ML)")
        rl_model = RLTrainer(self.rl_cfg, self.logger).train(data)
        dl_model = DeepLearningTrainer(self.dl_cfg, self.logger).train(data)
        ml_model = TraditionalMLTrainer(self.ml_cfg, self.logger).train(data)
        return {'rl': rl_model, 'dl': dl_model, 'ml': ml_model}

    def predict_ensemble(self, models: dict, obs: any) -> float:
        preds = [models['rl'].predict(obs),
                 models['dl'].predict(obs),
                 models['ml'].predict(obs)]
        weighted = sum(w * p for w, p in zip(self.ensemble_weights, preds))
        return weighted / sum(self.ensemble_weights)