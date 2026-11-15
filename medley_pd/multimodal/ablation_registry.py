"""Pure OFAT specification over effective Task 1 configurations, no torch/model IO."""
from __future__ import annotations
import copy
from dataclasses import dataclass, field

from .config import ResolvedConfig, validate_config, _validate_medley, apply_overrides, load_medley_config
from .model_registry import model_config, RunnerError
from .experiment_protocol import fingerprint


FAMILIES={
    "pooling": (("M2","M3"), ("attention","mean","max"), ("eeg.pooling.type",), "architecture_and_capacity"),
    "feature_adapter": (("M2","M3"), (False,True), ("eeg.feature_adapter.enabled",), "capacity"),
    "soft_tokens": (("M3",), (1,4,8), ("eeg.pooling.num_queries",), "architecture_and_capacity"),
    "adaptation": (("M1","M2","M3"), ("frozen","lora","qlora"), ("llm.tuning","llm.quantization"), "training_and_precision"),
    "numeric_skip": (("M2",), (False,True), ("fusion.numeric_skip",), "architecture_and_capacity"),
    "clinical_set": (("M1","M2","M3"), ("C0","C1","C2"), ("clinical_set",), "input_representation"),
    "encoder": (("M2","M3"), ("cbramod","scerebro"), ("eeg.encoder","eeg.feature_dim"), "input_representation_and_capacity"),
}


def semantic_diff(a,b,prefix=""):
    if isinstance(a,dict) and isinstance(b,dict):
        return {path:values for key in sorted(a.keys()|b.keys()) for path,values in
                semantic_diff(a.get(key),b.get(key),f"{prefix}.{key}".strip(".")).items()}
    return {} if a==b else {prefix:{"reference":copy.deepcopy(a),"candidate":copy.deepcopy(b)}}


@dataclass(frozen=True)
class AblationVariant:
    variant_id: str
    model: str
    family: str
    reference: ResolvedConfig=field(repr=False)
    candidate: ResolvedConfig=field(repr=False)
    changes: dict
    metadata: dict

    def assert_valid(self):
        rebuilt=resolve_variant(self.reference,model=self.model,family=self.family,
                                value=self.metadata["value"],candidate=self.candidate)
        if rebuilt.variant_id!=self.variant_id or rebuilt.changes!=self.changes or rebuilt.metadata!=self.metadata:
            raise RunnerError("variant_changed","Variant configuration or metadata changed after resolution")

    def summary(self):
        return {"variant_id":self.variant_id,"model":self.model,"family":self.family,"changes":copy.deepcopy(self.changes),
                **copy.deepcopy(self.metadata)}


def resolve_variant(config,*,model,family,value,candidate=None):
    if family not in FAMILIES:
        raise RunnerError("ablation_family","Unknown ablation family")
    models,values,paths,kind=FAMILIES[family]
    if model not in models or not any(type(value)==type(v) and value==v for v in values):
        raise RunnerError("ablation_applicability","Unsupported model/factor/value combination")
    reference=model_config(config,model)
    reference.settings["ablation"]["enabled"]=True
    settings=copy.deepcopy(reference.settings);medley=copy.deepcopy(reference.medley)
    updates={paths[0]:value}
    if family=="adaptation": updates["llm.quantization"]="nf4" if value=="qlora" else "none"
    if family=="encoder":
        updates["eeg.feature_dim"]={"cbramod":3800,"scerebro":3420}[value]
        resolved=load_medley_config(settings["dataset"],value,settings["task"])
        medley["encoder"]=resolved["encoder"]
    for path,item in updates.items():
        node=settings;parts=path.split(".")
        for key in parts[:-1]: node=node[key]
        node[parts[-1]]=item
    intended=ResolvedConfig(settings,medley,reference.split_validation)
    actual=copy.deepcopy(candidate) if candidate is not None else intended
    validate_config(reference.settings);validate_config(actual.settings)
    _validate_medley(reference.settings,reference.medley);_validate_medley(actual.settings,actual.medley)
    if family=="pooling" and reference.settings["eeg"]["pooling"]["num_queries"]!=1:
        raise RunnerError("pooling_ofat","Pooling-only comparisons require a K=1 reference; changing K is a separate factor")
    if family=="soft_tokens" and reference.settings["eeg"]["pooling"]["type"]!="attention":
        raise RunnerError("soft_tokens","K ablation requires learned-query attention")
    if semantic_diff(actual.to_dict(),intended.to_dict()):
        raise RunnerError("ofat","Candidate has undeclared or incorrectly bundled differences")
    changes=semantic_diff(reference.to_dict(),actual.to_dict())
    if not changes: raise RunnerError("ofat","A candidate must change the declared factor")
    metadata={"value":value,"factor_kind":kind,"declared_settings_bundle":list(paths),
              "structural_control":"single_factor_declared_bundle","capacity":"may_differ_measure_at_construction",
              "trainable_parameters":None,"computational_cost":None,
              "cohort_policy":"C1 exact_source; compare complete population and target identities before execution",
              "provenance":"planning_only_unverified_until_C1_C2_validation",
              "reference_fingerprint":fingerprint(reference.to_dict()),"candidate_fingerprint":fingerprint(actual.to_dict()),
              "seed_scope":list(reference.settings["evaluation"]["seeds"]),"fold_scope":list(reference.settings["evaluation"]["folds"])}
    identity={"schema":"medley_ofat_variant_v1","model":model,"family":family,"reference":reference.to_dict(),
              "candidate":actual.to_dict(),"changes":changes,"metadata":metadata}
    return AblationVariant("ablation-"+fingerprint(identity),model,family,reference,actual,changes,metadata)
