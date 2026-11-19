"""C2 planning and CLI safety, then bounded actual tiny-model integrations."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_multimodal_data_adapter import files as adapter_files
from test_multimodal_experiment_protocol import setup_source
from multimodal.config import CONFIG_DIR,load_multimodal_config,save_resolved_config
from multimodal.experiment_runner import plan_experiments,execution_guard,main
from multimodal.model_registry import registration,RunnerError


def test_c21_planning_whole_grid_and_model_selection_no_side_effects(tmp_path):
    cfg=load_multimodal_config(CONFIG_DIR / "fusion_b.json")
    original=copy.deepcopy(cfg.to_dict())
    plans=plan_experiments(cfg,models=["M0","M1","M2","M3"],output_dir=tmp_path / "outputs")
    assert len(plans)==200 and not (tmp_path / "outputs").exists()
    assert cfg.to_dict()==original and all(p.summary()["status"]=="planned_not_executed" for p in plans)
    one=plan_experiments(cfg,models=["M2"],seeds=[7],folds=[5])
    assert len(one)==1 and one[0].model=="M2" and one[0].seed==7 and one[0].fold==5
    assert one[0].config.settings["architecture"]=="fusion_a"
    assert plan_experiments(cfg,models=["M2"],seeds=[7],folds=[5])==one
    with pytest.raises(RunnerError): execution_guard(plans,max_runs=1)


@pytest.mark.parametrize("kwargs",[{"models":["M4"]},{"seeds":[99]},{"folds":[0]},{"models":["M1","M1"]}])
def test_c21_invalid_selection(kwargs):
    cfg=load_multimodal_config(CONFIG_DIR / "fusion_b.json")
    with pytest.raises(ValueError): plan_experiments(cfg,**kwargs)


@pytest.mark.parametrize("arguments",[["--execute"],["--execute","--model","M1"],
    ["--execute","--model","M1","--ack-grid","--max-runs","1"]])
def test_c21_cli_unbounded_execution_blocked(arguments):
    with pytest.raises(SystemExit) as error: main(["--config",str(CONFIG_DIR / "fusion_b.json"),*arguments])
    assert error.value.code==2


def test_c21_dry_run_cli_does_not_load_trainers_or_downloads(tmp_path):
    repo=Path(__file__).resolve().parents[2]
    env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
    result=subprocess.run([sys.executable,"-m","medley_pd.multimodal.experiment_runner","--config",
        str(CONFIG_DIR / "fusion_b.json"),"--model","M1","--seed","0","--fold","1","--output-dir",str(tmp_path / "unused")],
        cwd=repo,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report["planned_runs"]==1 and not report["execution_enabled"]
    assert not (tmp_path / "unused").exists()
    for name in ("M1","M2","M3"): assert registration(name).checkpoint_module
    with pytest.raises(RunnerError): registration("invalid")


@pytest.fixture(scope="module",autouse=True)
def threads():
    import torch
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tiny_factory():
    from test_multimodal_fusion_b import backbone
    from multimodal.llm_backbone import prepare_llm_backbone
    base=backbone()
    def create(config):
        model = copy.deepcopy(base.model).to(config.settings["llm"]["device"])
        return prepare_llm_backbone(config, model, copy.deepcopy(base.tokenizer))
    return create


def execution_inputs(adapter_files,tmp_path,model="M1",mode="frozen"):
    from multimodal.model_registry import model_config
    from multimodal.experiment_protocol import ExperimentProtocol
    cfg,adapter,path,ids=setup_source(adapter_files)
    cfg.settings["protocol"]["manifest_dir"]=str(tmp_path / "runs")
    cfg.settings["llm"].update(tuning=mode,quantization="none",compute_dtype="float32",device="cpu",lora_dropout=0.)
    cfg.settings["training"].update(max_epochs=2,early_stopping_patience=1,batch_size=2,gradient_checkpointing=False)
    cfg.settings["eeg"]["pooling"]["hidden_dim"]=16
    cfg.settings["fusion_a"].update(eeg_projection_dim=16,clinical_projection_dim=16)
    cfg.settings["fusion"].update(fusion_dim=16,dropout=0.)
    cfg=model_config(cfg,model)
    return cfg,adapter,ExperimentProtocol(cfg,adapter)


def test_c22_one_real_m1_epoch_uses_c1_loaders_and_validation_only(adapter_files,tmp_path,tiny_factory):
    from torch.utils.data import DataLoader
    from multimodal.experiment_protocol import seed_run
    from multimodal.dataset import MedleySubjectDataset,MedleySubjectCollator
    from multimodal.model_registry import create_runtime
    from multimodal.training_control import train_epoch_with_validation
    cfg,adapter,protocol=execution_inputs(adapter_files,tmp_path)
    prepared=protocol.prepare(seed=0,fold=1);seed_run(0,1)
    runtime=create_runtime("M1",cfg,prepared,backbone_factory=tiny_factory)
    def loader(ids): return DataLoader(MedleySubjectDataset(adapter,ids),batch_size=2,collate_fn=MedleySubjectCollator())
    before=runtime.model.head.weight.detach().clone()
    info=train_epoch_with_validation(runtime,loader(prepared.partition.train_subjects),loader(prepared.partition.validation_subjects))
    assert info["training"]["n"]==6 and info["validation"]["n"]==2
    assert not runtime.model.head.weight.equal(before)
    assert runtime.trainer.optimizer_steps==1


@pytest.mark.parametrize("model",["M1","M2","M3"])
@pytest.mark.parametrize("tty,disabled,device", [
    (True, False, "cpu"), (True, True, "cpu"), (False, False, "cpu"), (True, False, "cuda"),
])
def test_c24_real_model_checkpoint_export_and_one_test_pass(
    model, tty, disabled, device, adapter_files, tmp_path, tiny_factory, monkeypatch, capsys,
):
    import sys
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(sys.stderr, "isatty", lambda: tty)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TQDM_DISABLE", "1" if disabled else "0")
    from multimodal.runner_execution import execute_plan
    from multimodal.run_state import inspect_completed,read_json
    from multimodal.clinical_trainer import ClinicalTrainer
    calls=[];original=ClinicalTrainer.evaluate
    def record(self,loader,*,role="validation"):
        assert self.device.type == device
        calls.append(role)
        return original(self,loader,role=role)
    monkeypatch.setattr(ClinicalTrainer,"evaluate",record)
    cfg,adapter,protocol=execution_inputs(adapter_files,tmp_path,model)
    cfg.settings["llm"]["device"] = device
    plans=plan_experiments(cfg,models=[model],seeds=[0],folds=[1])
    result=execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)[0]
    stderr = capsys.readouterr().err
    if tty and not disabled:
        for entry in (
            "Experiments", f"model={model} seed=0 fold=1", "1/1", "Data preparation",
            "Model initialization", "Data loader setup", "Epochs", "train_loss=", "lr=",
            "optimizer_steps=", "Training batches", "3/3", "avg_loss=", "Validation",
            "loss=", "Test", "MAE=", "RMSE=",
        ):
            assert entry in stderr
    else:
        for entry in ("Experiments", "Epochs", "Training batches", "Validation:", "Test:"):
            assert entry not in stderr
    assert result.metrics["n"]==2 and calls.count("test")==1
    assert result.completion["status"]=="completed" and not result.reused
    report=inspect_completed(result.directory)
    assert report["integrity"]=="validated" and not report["training_resume"]
    assert main(["--inspect-completed",str(result.directory)])==0
    assert json.loads(capsys.readouterr().out)["integrity"]=="validated"
    attempt=result.directory / result.completion["attempt"]
    selection=read_json(attempt / "selection.json")
    history=read_json(attempt / "history.json")
    assert selection["validation_MAE"]==min(row["validation"]["MAE"] for row in history)
    assert selection["restoration"]=="verified_exact_trainable_weights" and not selection["test_used_for_selection"]
    with pytest.raises(RunnerError): execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)
    def must_not_load(cfg): raise AssertionError("completed reuse loaded a model")
    reused=execute_plan(plans,existing="resume",adapter_factory=lambda cfg:adapter,backbone_factory=must_not_load)[0]
    assert reused.reused and reused.metrics==result.metrics and calls.count("test")==1


@pytest.mark.parametrize("model",["M1","M2","M3"])
def test_real_lora_runtime_preserves_optimizer_groups(model,adapter_files,tmp_path,tiny_factory):
    from multimodal.runner_execution import execute_plan
    from multimodal.run_state import read_json
    cfg,adapter,_=execution_inputs(adapter_files,tmp_path,model,mode="lora")
    plans=plan_experiments(cfg,models=[model],seeds=[0],folds=[1])
    def factory(config):
        result=tiny_factory(config)
        assert result.mode=="lora"
        return result
    result=execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=factory)[0]
    checkpoint=next((result.directory / result.completion["attempt"] / "checkpoints").glob("*/trainable.safetensors"))
    from safetensors.torch import load_file
    assert any("lora_" in name for name in load_file(str(checkpoint)))
    assert result.completion["status"]=="completed"


def test_m0_import_readonly_no_model_or_checkpoint(adapter_files,tmp_path):
    import pandas as pd
    from multimodal.runner_execution import execute_plan
    from multimodal.run_state import inspect_completed,read_json
    from test_multimodal_experiment_protocol import predictions
    cfg,adapter,protocol=execution_inputs(adapter_files,tmp_path,"M0")
    source=tmp_path / "existing_m0.csv.gz"
    frame=predictions(adapter,protocol.get_outer_split(seed=0,fold=1));frame.to_csv(source,index=False)
    before=source.read_bytes()
    cfg.settings["runner"].update(baseline_path=str(source),baseline_model="clinical:C1")
    plans=plan_experiments(cfg,models=["M0"],seeds=[0],folds=[1])
    def forbidden(config): raise AssertionError("M0 constructed a model")
    result=execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=forbidden)[0]
    assert source.read_bytes()==before and result.completion["status"]=="imported"
    attempt=result.directory / result.completion["attempt"]
    assert not (attempt / "checkpoints").exists()
    assert read_json(attempt / "baseline_audit.json")["training_provenance"]=="not_verified"
    assert inspect_completed(result.directory)["status"]=="imported"
    reused=execute_plan(plans,existing="resume",adapter_factory=lambda cfg:adapter,backbone_factory=forbidden)[0]
    assert reused.reused and source.read_bytes()==before


def test_interrupt_requires_restart_preserves_failed_attempt_and_sanitizes_errors(adapter_files,tmp_path,tiny_factory,monkeypatch):
    import multimodal.runner_execution as execution
    from multimodal.run_state import read_json
    cfg,adapter,_=execution_inputs(adapter_files,tmp_path)
    plans=plan_experiments(cfg,models=["M1"],seeds=[0],folds=[1])
    real=execution.train_select_restore
    def interrupted(*args,**kwargs): raise KeyboardInterrupt("SENSITIVE_PATIENT_VALUE")
    monkeypatch.setattr(execution,"train_select_restore",interrupted)
    with pytest.raises(KeyboardInterrupt): execution.execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)
    roots=list((tmp_path / "runs").glob("*/executions/*"));assert len(roots)==1
    root=roots[0];assert not (root / "completed.json").exists()
    old=list((root / "attempts").iterdir())[0]
    assert read_json(old / "state.json")["status"]=="failed"
    assert "SENSITIVE" not in (old / "state.json").read_text()
    for policy in ("fail","resume"):
        with pytest.raises(RunnerError): execution.execute_plan(plans,existing=policy,adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)
    monkeypatch.setattr(execution,"train_select_restore",real)
    result=execution.execute_plan(plans,existing="restart",adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)[0]
    assert result.directory==root and len(list((root / "attempts").iterdir()))==2 and old.exists()
    assert read_json(old / "state.json")["status"]=="failed"


@pytest.mark.parametrize("artifact",["predictions.csv.gz","metrics.json","checkpoint","completed.json"])
def test_corrupt_completed_run_cannot_be_reused(artifact,adapter_files,tmp_path,tiny_factory):
    from multimodal.runner_execution import execute_plan
    cfg,adapter,_=execution_inputs(adapter_files,tmp_path)
    plans=plan_experiments(cfg,models=["M1"],seeds=[0],folds=[1])
    result=execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)[0]
    attempt=result.directory / result.completion["attempt"]
    if artifact=="checkpoint": path=next((attempt / "checkpoints").glob("*/trainable.safetensors"))
    elif artifact=="completed.json": path=result.directory / artifact
    else: path=attempt / artifact
    path.write_bytes(path.read_bytes()+b"CORRUPTED")
    with pytest.raises(RunnerError): execute_plan(plans,existing="resume",adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)
    with pytest.raises(RunnerError): execute_plan(plans,existing="restart",adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)


def test_validation_best_epoch_is_restored_before_only_test(adapter_files,tmp_path,tiny_factory,monkeypatch):
    import torch
    from multimodal.runner_execution import execute_plan
    from multimodal.clinical_trainer import ClinicalTrainer,ClinicalEvaluation
    cfg,adapter,_=execution_inputs(adapter_files,tmp_path)
    cfg.settings["training"].update(max_epochs=3,early_stopping_patience=5)
    original=ClinicalTrainer.evaluate;scores=iter([3.,1.,2.]);best={};roles=[]
    def controlled(self,loader,*,role="validation"):
        roles.append(role)
        if role=="test": torch.testing.assert_close(self.model.head.weight,best["weight"],rtol=0,atol=0)
        result=original(self,loader,role=role)
        if role=="validation":
            value=next(scores)
            if value==1.: best["weight"]=self.model.head.weight.detach().clone()
            return ClinicalEvaluation({**result.metrics,"MAE":value},result.predictions)
        return result
    monkeypatch.setattr(ClinicalTrainer,"evaluate",controlled)
    plans=plan_experiments(cfg,models=["M1"],seeds=[0],folds=[1])
    result=execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)[0]
    from multimodal.run_state import read_json
    assert read_json(result.directory / result.completion["attempt"] / "selection.json")["best_epoch"]==2
    assert roles==["validation"]*3+["test"]


def test_no_test_or_completion_when_restore_is_noop(adapter_files,tmp_path,tiny_factory,monkeypatch):
    from multimodal.runner_execution import execute_plan
    from multimodal.clinical_trainer import ClinicalTrainer,ClinicalEvaluation
    from multimodal.model_registry import ModelRuntime
    cfg,adapter,_=execution_inputs(adapter_files,tmp_path)
    cfg.settings["training"]["early_stopping_patience"]=5
    original=ClinicalTrainer.evaluate;scores=iter([1.,2.]);roles=[]
    def controlled(self,loader,*,role="validation"):
        roles.append(role);result=original(self,loader,role=role)
        return ClinicalEvaluation({**result.metrics,"MAE":next(scores)},result.predictions)
    monkeypatch.setattr(ClinicalTrainer,"evaluate",controlled)
    monkeypatch.setattr(ModelRuntime,"restore",lambda self,path:{"epochs_completed":1})
    plans=plan_experiments(cfg,models=["M1"],seeds=[0],folds=[1])
    with pytest.raises(RunnerError,match="restore"): execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)
    assert "test" not in roles and not list((tmp_path / "runs").glob("*/executions/*/completed.json"))


def test_guards_before_adapter_model_and_plan_mutation(adapter_files,tmp_path,tiny_factory):
    from multimodal.runner_execution import execute_plan
    cfg,adapter,_=execution_inputs(adapter_files,tmp_path)
    plans=plan_experiments(cfg,models=["M1"],seeds=[0,1],folds=[1])
    def forbidden(cfg): raise AssertionError("guard ran after construction")
    with pytest.raises(RunnerError): execute_plan(plans,max_runs=1,adapter_factory=forbidden,backbone_factory=forbidden)
    plan=plans[0];plan.config.settings["training"]["lr_new_modules"]*=2
    with pytest.raises(RunnerError,match="changed"): execute_plan([plan],adapter_factory=forbidden,backbone_factory=forbidden)


def test_execution_missing_model_id_fails_actionably_without_run_files(tmp_path,capsys):
    with pytest.raises(SystemExit): main(["--config",str(CONFIG_DIR / "fusion_b.json"),"--model","M1",
        "--seed","0","--fold","1","--execute","--output-dir",str(tmp_path / "unused")])
    assert "llm.model_id" in capsys.readouterr().err
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("change",["earlier_epoch","selection_identity","runtime_config","completion_seed"])
def test_review_rehashed_completion_must_match_selection_policy_and_config(change,adapter_files,tmp_path,tiny_factory,monkeypatch):
    from multimodal.runner_execution import execute_plan
    from multimodal.clinical_trainer import ClinicalTrainer,ClinicalEvaluation
    from multimodal.run_state import read_json,write_json,validate_completion,_attempt_hashes
    cfg,adapter,_=execution_inputs(adapter_files,tmp_path)
    cfg.settings["training"].update(max_epochs=3,early_stopping_patience=5)
    original=ClinicalTrainer.evaluate;scores=iter([3.,2.,1.])
    def controlled(self,loader,*,role="validation"):
        result=original(self,loader,role=role)
        return ClinicalEvaluation({**result.metrics,"MAE":next(scores)},result.predictions) if role=="validation" else result
    monkeypatch.setattr(ClinicalTrainer,"evaluate",controlled)
    plans=plan_experiments(cfg,models=["M1"],seeds=[0],folds=[1])
    result=execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)[0]
    attempt=result.directory / result.completion["attempt"]
    selection=read_json(attempt / "selection.json")
    if change=="earlier_epoch":
        selection.update(best_epoch=1,validation_MAE=3.,checkpoint="checkpoints/epoch-0001")
    elif change=="selection_identity": selection["c1_run_id"]="other_run"
    elif change=="runtime_config":
        runtime=read_json(attempt / "runtime_config.json");runtime["settings"]["training"]["lr_new_modules"]*=2
        write_json(attempt / "runtime_config.json",runtime)
        checkpoint=attempt / selection["checkpoint"]
        write_json(checkpoint / "resolved_config.json",runtime)
        cp=read_json(checkpoint / "manifest.json")
        from multimodal.clinical_checkpoint import _config_digest
        cp["identity"]["config_digest"]=_config_digest(runtime)
        write_json(checkpoint / "manifest.json",cp)
    write_json(attempt / "selection.json",selection)
    completion=read_json(result.directory / "completed.json")
    completion["artifacts"]=_attempt_hashes(result.directory,attempt)
    if change=="completion_seed": completion["seed"]=False
    with pytest.raises(RunnerError): validate_completion(result.directory,completion)


def test_runner_config_roundtrip_and_missing_dependency(tmp_path,monkeypatch):
    from multimodal.model_registry import check_readiness
    from multimodal.config import ConfigError
    cfg=load_multimodal_config(CONFIG_DIR / "fusion_a.json",["runner.min_delta=0.25","runner.shuffle_train=false"])
    snapshot=tmp_path / "config.json";save_resolved_config(cfg,snapshot)
    assert load_multimodal_config(snapshot).to_dict()==cfg.to_dict()
    for option in ("runner.selection_metric=RMSE","runner.num_workers=2","runner.min_delta=-1","runner.shuffle_train=1"):
        with pytest.raises(ConfigError): load_multimodal_config(CONFIG_DIR / "fusion_b.json",[option])
    cfg.settings["llm"].update(model_id="synthetic-local-only-model",revision="a"*40)
    import multimodal.model_registry as registry
    original=registry.importlib.util.find_spec
    monkeypatch.setattr(registry.importlib.util,"find_spec",lambda name:None if name=="bitsandbytes" else original(name))
    with pytest.raises(RunnerError,match="bitsandbytes"): check_readiness(cfg)


def test_dry_run_imports_no_torch_or_backbone_in_fresh_process():
    repo=Path(__file__).resolve().parents[2]
    code="""
import sys
from medley_pd.multimodal.experiment_runner import main
main(['--config',sys.argv[1],'--model','M1','--seed','0','--fold','1'])
assert 'torch' not in sys.modules
assert not any(name.endswith('.llm_backbone') for name in sys.modules)
"""
    completed=subprocess.run([sys.executable,"-c",code,str(CONFIG_DIR / "fusion_b.json")],cwd=repo,
                             capture_output=True,text=True,env={**os.environ,"HF_HUB_OFFLINE":"1"})
    assert completed.returncode==0,completed.stderr


def test_exception_after_atomic_completion_preserves_published_run(adapter_files,tmp_path,tiny_factory,monkeypatch):
    from multimodal.runner_execution import execute_plan
    from multimodal.run_state import RunStore,inspect_completed
    cfg,adapter,_=execution_inputs(adapter_files,tmp_path)
    plans=plan_experiments(cfg,models=["M1"],seeds=[0],folds=[1])
    original=RunStore.complete
    def completed_then_interrupted(self,**kwargs):
        original(self,**kwargs)
        raise RuntimeError("post-publication exception must not mutate completed artifacts")
    monkeypatch.setattr(RunStore,"complete",completed_then_interrupted)
    with pytest.raises(RuntimeError): execute_plan(plans,adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)
    root=next((tmp_path / "runs").glob("*/executions/*"))
    assert inspect_completed(root)["status"]=="completed"
    reused=execute_plan(plans,existing="resume",adapter_factory=lambda cfg:adapter,backbone_factory=tiny_factory)[0]
    assert reused.reused
