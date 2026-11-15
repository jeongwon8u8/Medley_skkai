"""Exact paired outcomes and explicit comparability levels, never a silent join."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy

import pandas as pd

from .experiment_evaluation import AnalysisError,PredictionSeries,KEYS,validate_prediction_frame
from .experiment_protocol import fingerprint


def _flatten(value, prefix="settings"):
    if isinstance(value,dict):
        return {path:item for key,entry in value.items() for path,item in _flatten(entry,f"{prefix}.{key}").items()}
    return {prefix:value}


def _shared_settings(run):
    settings=copy.deepcopy(run["manifest"]["resolved_config"]["settings"])
    chosen={key:settings[key] for key in ("dataset","task","eeg","clinical_set","clinical_text","clinical_regression","llm","loss","training")}
    for key in ("cache_dir","local_files_only","remote_code_reason","trust_remote_code"):
        chosen["llm"].pop(key,None)
    chosen["runner"]={key:settings.get("runner",{}).get(key,default) for key,default in
                      (("selection_metric","MAE"),("min_delta",0.),("shuffle_train",True),("num_workers",0))}
    return _flatten(chosen)


def _base_known(run, synthetic):
    cp=run["checkpoint_identity"]
    if cp is None: return False
    info=run["manifest"]["identity"]
    if cp.get("model_id") and info["llm_revision_status"]=="immutable_sha_supplied_not_verified":
        return bool(cp.get("revision"))
    return bool(synthetic and cp.get("synthetic_frozen_digest"))


def comparison_conditions(candidate,reference,runs,*,synthetic=False,allowed_differences=()):
    checks={"population":True,"target_definition":True,"outer_splits":True,"eeg_source":True,
            "clinical_source":True,"llm_base":True,"tokenizer":True,"training_protocol":True,"code_environment":True}
    observed_allowed=set();disallowed=set();architecture_equal=True;intrinsic_equal=True
    for key in sorted(runs):
        a,b=candidate.runs.get(key),reference.runs.get(key)
        if a is None or b is None:
            return {"checks":{k:False for k in checks},"recorded_protocol_comparable":False,
                    "fully_controlled_configuration":False,"allowed_differences":[],"unverified_or_different":["run_provenance_unavailable"]}
        ia,ib=(r["manifest"]["identity"] for r in (a,b))
        checks["population"] &= all(ia[k]==ib[k] for k in ("cohort_fingerprint","source_subject_fingerprint","source_cohort_n"))
        checks["target_definition"] &= ia["target"]==ib["target"] and ia["unit"]==ib["unit"]
        checks["outer_splits"] &= ia["outer_sha256"]==ib["outer_sha256"] and ia["assignment_fingerprint"]==ib["assignment_fingerprint"]
        sa,sb=_shared_settings(a),_shared_settings(b)
        differences={path for path in sa.keys()|sb.keys() if sa.get(path)!=sb.get(path)}
        allowed=differences & set(allowed_differences)
        observed_allowed.update(allowed);disallowed.update(differences-set(allowed_differences))
        encoder_ablation={"settings.eeg.encoder","settings.eeg.feature_dim"} <= allowed
        clinical_ablation=bool({p for p in allowed if p=="settings.clinical_set" or p.startswith("settings.clinical_text.")})
        checks["eeg_source"] &= ia["cache_fingerprint"]==ib["cache_fingerprint"] or encoder_ablation
        checks["clinical_source"] &= ia["clinical_fingerprint"]==ib["clinical_fingerprint"] or clinical_ablation
        baseline=ia["model"]=="M0" or ib["model"]=="M0"
        checks["training_protocol"] &= not baseline and (a["manifest"]["subjects"]==b["manifest"]["subjects"]
            or "settings.training.validation_fraction" in allowed)
        checks["code_environment"] &= all(ia["runtime"][p]==ib["runtime"][p] for p in ("python","libraries","application_source_fingerprint"))
        if not baseline:
            ca,cb=a["checkpoint_identity"],b["checkpoint_identity"]
            llm_ablation=bool({"settings.llm.model_id","settings.llm.revision"} & allowed)
            same_base=all(ca.get(p)==cb.get(p) for p in ("model_id","revision","synthetic_frozen_digest","base_config_digest"))
            if synthetic and "settings.llm.tuning" in allowed:
                canonical_a,canonical_b=(cp.get("canonical_synthetic_base_digest") for cp in (ca,cb))
                same_base |= bool(canonical_a and canonical_a==canonical_b and all(ca.get(p)==cb.get(p) for p in ("model_id","revision","base_config_digest")))
            checks["llm_base"] &= _base_known(a,synthetic) and _base_known(b,synthetic) and (same_base or llm_ablation)
            checks["tokenizer"] &= bool(ca.get("tokenizer_digest") and cb.get("tokenizer_digest")) and (ca["tokenizer_digest"]==cb["tokenizer_digest"] or llm_ablation)
        else:
            checks["llm_base"]=False;checks["tokenizer"]=False
        architecture_equal &= ia["architecture"]==ib["architecture"]
        config_a,config_b=(r["manifest"]["resolved_config"]["settings"] for r in (a,b))
        intrinsic_equal &= all(config_a.get(name)==config_b.get(name) for name in ("fusion","fusion_a","fusion_b_training","ablation"))
    reasons=sorted([name for name,value in checks.items() if not value]+list(disallowed))
    comparable=all(checks.values()) and not disallowed
    return {"checks":checks,"recorded_protocol_comparable":comparable,
            "fully_controlled_configuration":bool(comparable and architecture_equal and intrinsic_equal and not observed_allowed),
            "intrinsic_model_settings_equal":intrinsic_equal,"allowed_differences":sorted(observed_allowed),"unverified_or_different":reasons,
            "limits":"recorded input/protocol equivalence, not clinical extraction truth or equal model capacity"}


@dataclass(frozen=True)
class MatchedPair:
    candidate: str
    reference: str
    _frame: pd.DataFrame=field(repr=False)
    diagnostics: dict
    conditions: dict

    @property
    def frame(self): return self._frame.copy(deep=True)

    def public_metadata(self):
        return {"candidate":self.candidate,"reference":self.reference,
                **{k:v for k,v in self.diagnostics.items() if k not in ("excluded_candidate_keys","excluded_reference_keys")},
                "conditions":copy.deepcopy(self.conditions)}


def match_predictions(candidate,reference,*,matching="strict",expected_runs=None,synthetic=False,allowed_differences=()):
    if matching not in ("strict","matched_subset") or candidate.label==reference.label:
        raise AnalysisError("matching","Select distinct candidate/reference labels and a supported matching policy")
    candidate.assert_unchanged();reference.assert_unchanged()
    a,b=validate_prediction_frame(candidate.frame),validate_prediction_frame(reference.frame)
    if a.model.nunique()!=1 or b.model.nunique()!=1:
        raise AnalysisError("series","Each comparison dimension must contain exactly one underlying model")
    all_a,all_b=len(a),len(b)
    if expected_runs is not None:
        scope=set(expected_runs)
        a=a.loc[[tuple(k) in scope for k in a[["seed","fold"]].itertuples(index=False,name=None)]]
        b=b.loc[[tuple(k) in scope for k in b[["seed","fold"]].itertuples(index=False,name=None)]]
    else: scope=set(a[["seed","fold"]].itertuples(index=False,name=None))|set(b[["seed","fold"]].itertuples(index=False,name=None))
    ar=set(a[["seed","fold"]].itertuples(index=False,name=None));br=set(b[["seed","fold"]].itertuples(index=False,name=None))
    ak=set(a[KEYS].itertuples(index=False,name=None));bk=set(b[KEYS].itertuples(index=False,name=None))
    if matching=="strict" and (ar!=scope or br!=scope or ak!=bk):
        raise AnalysisError("missing_matches","Strict analysis requires every requested run and exact subject coverage")
    # Targets/subgroups are fixed per subject, even if a bad fold assignment would
    # otherwise remove that row from an intersection.
    am=a.drop_duplicates("subject").set_index("subject");bm=b.drop_duplicates("subject").set_index("subject")
    common=sorted(set(am.index)&set(bm.index))
    if any(am.loc[s,"target"]!=bm.loc[s,"target"] or am.loc[s,"subgroup"]!=bm.loc[s,"subgroup"] for s in common):
        raise AnalysisError("outcome_metadata","Subject targets or subgroups disagree; no intersection can repair them")
    if not ak & bk:
        raise AnalysisError("no_matches","No exact seed/fold/subject outcome pairs are available")
    joined=a.merge(b,on=KEYS,how="inner",validate="one_to_one",suffixes=("_candidate","_reference")).sort_values(KEYS).reset_index(drop=True)
    if not (joined.target_candidate==joined.target_reference).all():
        raise AnalysisError("target","Exact paired outcomes require equal original-unit targets")
    frame=joined.loc[:,KEYS].copy()
    frame["target"]=joined.target_candidate;frame["subgroup"]=joined.subgroup_candidate
    frame["candidate_prediction"]=joined.yhat_candidate;frame["reference_prediction"]=joined.yhat_reference
    matched_runs=set(frame[["seed","fold"]].itertuples(index=False,name=None))
    conditions=comparison_conditions(candidate,reference,matched_runs,synthetic=synthetic,allowed_differences=allowed_differences)
    if matching=="strict" and not conditions["recorded_protocol_comparable"]:
        error=AnalysisError("provenance","Strict comparison requires verified compatible recorded conditions")
        error.conditions=conditions
        raise error
    diagnostics={"matching":matching,"population_label":"full_requested_scope" if matching=="strict" else "matched subset",
        "source_candidate_rows":all_a,"source_reference_rows":all_b,"candidate_rows":len(a),"reference_rows":len(b),
        "matched_rows":len(frame),"unique_subjects":frame.subject.nunique(),"matched_runs":len(matched_runs),
        "missing_candidate_runs":[list(k) for k in sorted(scope-ar)],"missing_reference_runs":[list(k) for k in sorted(scope-br)],
        "excluded_candidate_keys":[list(k) for k in sorted(ak-bk)],"excluded_reference_keys":[list(k) for k in sorted(bk-ak)],
        "excluded_candidate_rows":len(ak-bk),"excluded_reference_rows":len(bk-ak),
        "outside_scope_candidate_rows":all_a-len(a),"outside_scope_reference_rows":all_b-len(b)}
    if matching=="matched_subset": conditions["fully_controlled_configuration"]=False
    return MatchedPair(candidate.label,reference.label,frame,diagnostics,conditions)
