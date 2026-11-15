"""Run ownership, atomic records and incomplete-state safety; synthetic only."""
from pathlib import Path

import pytest

from test_multimodal_data_adapter import files as adapter_files
from test_multimodal_experiment_runner import execution_inputs
from multimodal.experiment_manifest import build_manifest
from multimodal.run_state import RunStore,write_json,read_json,inspect_completed
from multimodal.model_registry import RunnerError


def test_atomic_exclusive_write_never_overwrites(tmp_path):
    path=tmp_path / "completion.json"
    write_json(path,{"first":1},exclusive=True)
    with pytest.raises(FileExistsError): write_json(path,{"second":2},exclusive=True)
    assert read_json(path)=={"first":1} and not list(tmp_path.glob("*.tmp"))


def test_parallel_ownership_incomplete_rejection_and_restart(adapter_files,tmp_path):
    cfg,adapter,protocol=execution_inputs(adapter_files,tmp_path)
    manifest=build_manifest(protocol,seed=0,fold=1)
    first=RunStore(manifest);second=RunStore(manifest)
    with first.claim():
        with pytest.raises(RunnerError,match="owns"):
            with second.claim(existing="restart"): pass
        with pytest.raises(RunnerError): first.complete(prediction_model="clinical_llm:C1")
    assert read_json(first.attempt / "state.json")["status"]=="incomplete"
    with pytest.raises(RunnerError): inspect_completed(first.root)
    with pytest.raises(RunnerError):
        with second.claim(existing="resume"): pass
    with second.claim(existing="restart"): pass
    assert len(list((first.root / "attempts").iterdir()))==2


def test_review_existing_manifest_cannot_bypass_protected_source_root(adapter_files,tmp_path):
    from multimodal.experiment_protocol import ExperimentProtocol
    from multimodal.experiment_manifest import save_manifest
    cfg,adapter,_=execution_inputs(adapter_files,tmp_path)
    cfg.settings["protocol"]["manifest_dir"]=cfg.medley["paths"]["data_root"]
    manifest=build_manifest(ExperimentProtocol(cfg,adapter),seed=0,fold=1)
    with pytest.raises(ValueError): save_manifest(manifest)
    namespace=Path(manifest.to_dict()["output_namespace"])
    namespace.mkdir();(namespace / "manifest.json").write_text(manifest.to_json())
    with pytest.raises(RunnerError): RunStore(manifest)
    assert not (namespace / "executions").exists()


def test_review_namespace_and_execution_symlinks_rejected(adapter_files,tmp_path):
    cfg,adapter,protocol=execution_inputs(adapter_files,tmp_path)
    manifest=build_manifest(protocol,seed=0,fold=1)
    namespace=Path(manifest.to_dict()["output_namespace"])
    other=tmp_path / "redirect";other.mkdir()
    namespace.parent.mkdir(parents=True);namespace.symlink_to(other,target_is_directory=True)
    with pytest.raises(RunnerError): RunStore(manifest)
    namespace.unlink()
    store=RunStore(manifest)
    (namespace / "executions").symlink_to(other,target_is_directory=True)
    with pytest.raises(RunnerError):
        with store.claim(): pass
    assert not list(other.iterdir())


def test_no_state_or_completion_write_without_ownership(adapter_files,tmp_path):
    cfg,adapter,protocol=execution_inputs(adapter_files,tmp_path)
    store=RunStore(build_manifest(protocol,seed=0,fold=1))
    with pytest.raises(RunnerError): store.state("completed")
    with pytest.raises(RunnerError): store.complete(prediction_model="clinical_llm:C1")
    with store.claim():
        with pytest.raises(RunnerError):
            with store.claim(existing="restart"): pass
    with pytest.raises(RunnerError): store.state("completed")
    assert not (store.root / "completed.json").exists()


def test_bootstrap_interruption_can_publish_missing_manifest_without_overwrite(adapter_files,tmp_path):
    from multimodal.experiment_manifest import read_manifest
    cfg,adapter,protocol=execution_inputs(adapter_files,tmp_path)
    manifest=build_manifest(protocol,seed=0,fold=1)
    namespace=Path(manifest.to_dict()["output_namespace"])
    namespace.mkdir(parents=True)
    orphan=namespace / ".manifest.interrupted.tmp";orphan.write_text("interrupted private metadata")
    store=RunStore(manifest)
    assert read_manifest(namespace / "manifest.json")==manifest.to_dict()
    assert orphan.read_text()=="interrupted private metadata"
    with store.claim(): pass
    assert not (store.root / "completed.json").exists()
