"""M2 explicit-epoch execution using Task 6/A3 shared subject-level utilities."""
from .clinical_trainer import ClinicalTrainer, ClinicalEvaluation, seed_training
from .config import FUSION_A_DEFAULTS
from .fusion_a import EEGClinicalRepresentationFusionRegressor
from .training_utils import build_role_optimizer, prepare_eeg_inputs

FusionAEvaluation = ClinicalEvaluation


def build_optimizer(model):
    if not isinstance(model,EEGClinicalRepresentationFusionRegressor):
        raise ValueError("M2 optimizer requires Architecture A")
    options={**FUSION_A_DEFAULTS,**model.backbone.resolved_config.settings.get("fusion_a",{})}
    return build_role_optimizer(model,[
        ("eeg_pooling",model.eeg_encoder.pooling,options["lr_pooling"]),
        ("eeg_projection",model.eeg_encoder.projection,options["lr_eeg_projection"]),
        ("clinical_projection",model.clinical_encoder.projection,options["lr_clinical_projection"]),
        ("fusion",model.head,options["lr_fusion"]),
        ("feature_adapter",model.eeg_encoder.feature_adapter,options["lr_feature_adapter"]),
    ])


class FusionATrainer(ClinicalTrainer):
    """Same split/loader API as A3; never generates splits or selects checkpoints."""

    def _model_device(self):
        return self.model.head.norm.weight.device

    def _build_optimizer(self):
        return build_optimizer(self.model)

    def _model_identifier(self):
        return f"fusion_a_llm:{self.serializer.feature_set}"

    def _batch(self,batch,expected,seen):
        ids,targets=self._batch_targets(batch,expected,seen)
        inputs=prepare_eeg_inputs(batch,n_subjects=len(ids),device=self.device,
                                  dtype=self.model.pooling.input_projection.weight.dtype)
        return ids,inputs,targets

    def _forward(self,inputs):
        return self.model(**inputs)
