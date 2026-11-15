"""C1 sequential synthetic protocol gates; no models, network or GPU."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from test_multimodal_data_adapter import files as adapter_files
from multimodal.config import CONFIG_DIR, load_multimodal_config
from multimodal.data_adapter import MedleyDataAdapter
from multimodal.experiment_protocol import ReadOnlyOuterSplits, ProtocolError


def setup_source(adapter_files):
    cfg, table, cache, path, ids = adapter_files
    cfg.settings["clinical_text"]["sex_code_mapping"] = {"0":"female", "1":"male"}
    return cfg, MedleyDataAdapter(cfg), path, ids


def test_c11_original_style_outer_assignments_unchanged(adapter_files):
    cfg, adapter, path, ids = setup_source(adapter_files)
    raw = path.read_bytes()
    manager = ReadOnlyOuterSplits(cfg, adapter)
    for seed in (0,1):
        for fold in (1,2,5):
            outer = manager.get_outer_split(seed=seed, fold=fold)
            assert outer == adapter.get_split(seed,fold)
            assert set(outer.train_subjects).isdisjoint(outer.test_subjects)
            assert set(outer.train_subjects + outer.test_subjects) == set(ids)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("case", ["missing", "duplicate", "bad_header", "missing_subject", "mutated", "adapter_overlap"])
def test_c11_invalid_or_changed_source_fails(adapter_files,case):
    cfg,adapter,path,ids=setup_source(adapter_files)
    if case == "missing":
        path.unlink()
        with pytest.raises(ProtocolError): ReadOnlyOuterSplits(cfg,adapter)
    elif case == "duplicate":
        path.write_text(path.read_text()+path.read_text().splitlines()[1]+"\n")
        with pytest.raises(ProtocolError): ReadOnlyOuterSplits(cfg,adapter)
    elif case == "bad_header":
        path.write_text("subject,seed,seed\n")
        with pytest.raises(ProtocolError): ReadOnlyOuterSplits(cfg,adapter)
    else:
        manager=ReadOnlyOuterSplits(cfg,adapter)
        if case == "missing_subject":
            manager.adapter=SimpleNamespace(subject_ids=ids[1:],get_split=adapter.get_split)
        elif case == "mutated": path.write_bytes(path.read_bytes()+b"\n")
        elif case == "adapter_overlap":
            outer=adapter.get_split(0,1)
            bad=replace(outer,test_subjects=outer.train_subjects[:1])
            manager.adapter=SimpleNamespace(subject_ids=ids,get_split=lambda *args:bad)
        with pytest.raises(ProtocolError): manager.get_outer_split(seed=0,fold=1)


def test_c11_noncontiguous_actual_seed_and_fold_selection(adapter_files):
    import pandas as pd
    cfg,adapter,path,ids=setup_source(adapter_files)
    frame=pd.read_csv(path);frame.seed=frame.seed.map({0:1,1:7});frame.to_csv(path,index=False)
    cfg.settings["evaluation"]["seeds"]=[1,7]
    cfg.settings["evaluation"]["folds"]=[2,5]
    adapter=MedleyDataAdapter(cfg)
    manager=ReadOnlyOuterSplits(cfg,adapter)
    assert manager.get_outer_split(seed=7,fold=5)==adapter.get_split(7,5)
    with pytest.raises(ProtocolError): manager.get_outer_split(seed=0,fold=1)


from multimodal.experiment_cohort import validate_cohort,compare_cohorts,validate_predictions,audit_prediction_file


def test_c12_complete_cohort_and_content_fingerprints(adapter_files):
    cfg,adapter,path,ids=setup_source(adapter_files)
    first=validate_cohort(cfg,adapter,ids)
    second=validate_cohort(cfg,adapter,list(reversed(ids)))
    assert first==second and first.n==10
    assert all(sid not in repr(first) for sid in ids)
    assert first.provenance_scope=="validated_decoded_records_not_extraction_history"
    assert compare_cohorts({"M1":first,"M2":second,"M3":first}).matched


@pytest.mark.parametrize("case",["missing","extra","duplicate","target","subgroup","cache","clinical","sex_mapping"])
def test_c12_invalid_records_never_silently_filter(adapter_files,case):
    import numpy as np
    cfg,adapter,path,ids=setup_source(adapter_files)
    base=validate_cohort(cfg,adapter,ids)
    record=adapter.get_subject(ids[0])
    if case in ("missing","extra","duplicate"):
        membership={"missing":ids[1:],"extra":[*ids,"synthetic-extra"],"duplicate":[*ids,ids[0]]}[case]
        bad=SimpleNamespace(subject_ids=membership,get_subject=adapter.get_subject)
    else:
        if case=="target": record=replace(record,moca=float("nan"))
        elif case=="subgroup": record=replace(record,subgroup=123)
        elif case=="cache": record=replace(record,eeg_features=np.ones((2,5),np.float32))
        elif case=="clinical": record=replace(record,clinical={"moca":20})
        elif case=="sex_mapping": cfg.settings["clinical_text"]["sex_code_mapping"]=None
        bad=SimpleNamespace(subject_ids=ids,get_subject=lambda sid:record if sid==ids[0] else adapter.get_subject(sid))
    with pytest.raises(ValueError): validate_cohort(cfg,bad,ids)
    assert base.n==10


@pytest.mark.parametrize("change",["target","subgroup","subjects"])
def test_c12_comparison_reports_inconsistency(adapter_files,change):
    cfg,adapter,path,ids=setup_source(adapter_files)
    base=validate_cohort(cfg,adapter,ids)
    if change=="subjects": other=replace(base,subjects=base.subjects[:-1],metadata=base.metadata[:-1])
    else:
        row=base.metadata[0]
        row=replace(row,**({"target":row.target+1} if change=="target" else {"subgroup":"OTHER"}))
        other=replace(base,metadata=(row,*base.metadata[1:]))
    assert not compare_cohorts({"M1":base,"M0":other},strict=False).matched
    with pytest.raises(ProtocolError) as error: compare_cohorts({"M1":base,"M0":other})
    assert not error.value.report.matched


def predictions(adapter,outer):
    import pandas as pd
    return pd.DataFrame([dict(seed=outer.seed,fold=outer.fold,subject=sid,model="clinical:C1",
        subgroup=adapter.get_subject(sid).subgroup,y=adapter.get_subject(sid).moca,
        target=adapter.get_subject(sid).moca,yhat=22.) for sid in outer.test_subjects])


@pytest.mark.parametrize("case",["valid","duplicate","missing","extra","wrong_fold","target","subgroup","nonfinite"])
def test_c12_baseline_prediction_alignment_only(adapter_files,case,tmp_path):
    import pandas as pd
    cfg,adapter,path,ids=setup_source(adapter_files)
    cohort=validate_cohort(cfg,adapter,ids);outer=adapter.get_split(0,1);frame=predictions(adapter,outer)
    if case=="valid":
        file=tmp_path / "baseline.csv.gz";frame.to_csv(file,index=False)
        raw=file.read_bytes()
        result=audit_prediction_file(file,cohort,outer,model="clinical:C1")
        assert result["population_aligned"] and result["training_provenance"]=="not_verified"
        assert file.read_bytes()==raw
        return
    if case=="duplicate": frame=pd.concat([frame,frame.iloc[:1]])
    elif case=="missing": frame=frame.iloc[:1]
    elif case=="extra": frame.loc[0,"subject"]=outer.train_subjects[0]
    elif case=="wrong_fold": frame.loc[0,"fold"]=2
    elif case=="target": frame.loc[0,"y"]+=1
    elif case=="subgroup": frame.loc[0,"subgroup"]="OTHER"
    elif case=="nonfinite": frame.loc[0,"yhat"]=float("nan")
    with pytest.raises(ProtocolError): validate_predictions(frame,cohort,outer,model="clinical:C1")


from multimodal.experiment_protocol import ExperimentProtocol,select_inner_validation,derive_seed,dataloader_generator,seed_run


def test_c13_inner_split_deterministic_complete_and_global_rng_independent(adapter_files):
    import random
    import numpy as np
    import torch
    cfg,adapter,path,ids=setup_source(adapter_files)
    protocol=ExperimentProtocol(cfg,adapter)
    first=protocol.prepare(seed=0,fold=1)
    random.seed(991);np.random.seed(17);torch.manual_seed(41)
    random.random();np.random.rand(100);torch.rand(100)
    assert protocol.prepare(seed=0,fold=1)==first
    p=first.partition
    assert p.counts==dict(train=6,validation=2,test=2)
    assert set(p.train_subjects)|set(p.validation_subjects)==set(first.outer.train_subjects)
    for a,b in [(p.train_subjects,p.validation_subjects),(p.train_subjects,p.test_subjects),(p.validation_subjects,p.test_subjects)]:
        assert set(a).isdisjoint(b)
    assert p.test_subjects==first.outer.test_subjects
    shuffled=replace(first.outer,train_subjects=tuple(reversed(first.outer.train_subjects)))
    assert set(select_inner_validation(shuffled).validation_subjects)==set(p.validation_subjects)


@pytest.mark.parametrize("n,fraction,nval",[(2,.2,1),(5,.2,1),(8,.2,2),(10,.3,3),(20,.15,3)])
def test_c13_decimal_ceil_policy(n,fraction,nval):
    from multimodal.data_adapter import SubjectSplit
    outer=SubjectSplit(7,5,tuple(f"synthetic-{i}" for i in range(n)),("synthetic-test",))
    result=select_inner_validation(outer,fraction=fraction)
    assert len(result.validation_subjects)==nval and len(result.train_subjects)==n-nval


@pytest.mark.parametrize("n,fraction",[(1,.2),(2,.99),(3,.9),(5,0),(5,1),(5,float("nan")),(5,True)])
def test_c13_small_or_invalid_policy_errors(n,fraction):
    from multimodal.data_adapter import SubjectSplit
    outer=SubjectSplit(7,5,tuple(f"synthetic-{i}" for i in range(n)),("synthetic-test",))
    with pytest.raises(ProtocolError): select_inner_validation(outer,fraction=fraction)


def test_c13_architecture_independent_partitions(adapter_files):
    cfg,adapter,path,ids=setup_source(adapter_files)
    partitions=[]
    for architecture,fusion in [("clinical_only","none"),("fusion_a","representation"),("fusion_b","soft_token")]:
        other=copy.deepcopy(cfg);other.settings["architecture"]=architecture;other.settings["fusion"]["type"]=fusion
        partitions.append(ExperimentProtocol(other,MedleyDataAdapter(other)).prepare(seed=1,fold=5).partition)
    assert partitions[0]==partitions[1]==partitions[2]


def test_c13_seed_domain_separation_and_loader_generator():
    import random
    import numpy as np
    import torch
    assert derive_seed(7,5)==derive_seed(7,5)
    assert len({derive_seed(7,5),derive_seed(7,2),derive_seed(1,5),derive_seed(7,5,purpose="training")})==4
    before=torch.random.get_rng_state().clone()
    a=torch.randperm(20,generator=dataloader_generator(7,5))
    b=torch.randperm(20,generator=dataloader_generator(7,5))
    assert torch.equal(a,b) and torch.equal(before,torch.random.get_rng_state())
    seed_run(7,5);a=(random.random(),np.random.rand(),torch.rand(1))
    seed_run(7,5);b=(random.random(),np.random.rand(),torch.rand(1))
    assert a[0]==b[0] and a[1]==b[1] and torch.equal(a[2],b[2])


def test_review_inner_selection_independent_of_decimal_context():
    from decimal import localcontext,ROUND_DOWN
    from multimodal.data_adapter import SubjectSplit
    outer=SubjectSplit(7,5,tuple(f"synthetic-{i}" for i in range(11)),("synthetic-test",))
    normal=select_inner_validation(outer,fraction=.2)
    assert len(normal.validation_subjects)==3
    with localcontext() as context:
        context.prec=1;context.rounding=ROUND_DOWN
        assert select_inner_validation(outer,fraction=.2)==normal


@pytest.mark.parametrize("column,value",[("seed",True),("fold",True),("seed",1.),("model",None)])
def test_review_prediction_integer_metadata_is_not_boolean_or_float(adapter_files,column,value):
    cfg,adapter,path,ids=setup_source(adapter_files)
    outer=adapter.get_split(1,1);cohort=validate_cohort(cfg,adapter,ids);frame=predictions(adapter,outer)
    frame[column]=value
    with pytest.raises(ProtocolError): validate_predictions(frame,cohort,outer,model="clinical:C1")
