"""frozen S-CEReBrO (pulp-bio/BioFoundation PR #18, commit 361c017)

BioFoundation 코드를 그대로 불러와 사전학습 조건에 맞춘다.
  - 입력: 200 Hz, 1초(200샘플) 패치. 공통 전처리(0.3–75 Hz, 노치, 5초 구간)는 make_mumtaz.py 와 동일
  - 정규화: 구간마다 채널별 min-max → [-1, 1], 절대 엡실론 1e-6 (LMDBDataset._min_max_normalize 와 동일, μV 기준)
  - 전극 좌표: (활성 전극, 기준전극) 3D 좌표 [C, 2, 3] — make_datasets.common.referential_coordinates 와 동일
    기준전극은 데이터셋 yaml 의 reference (PD-EEG: CZ, Mumtaz: LE, 평균 기준: AR)
  - 가중치: S_CEReBrO.safetensors. 'model.' 접두어는 떼고 'model_head.' 는 버림. 모양이 맞는 텐서만 로드
  - 출력 토큰 [B, C·P, D] → pool=channel: 패치 평균 후 펼침 (C·D, CBraMod 와 같은 방식) / pool=mean: 전체 평균 (D)
"""
import importlib.util
import os
import sys

import numpy as np

from . import Encoder, register


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def referential_coordinates(repo, channels, reference):
    """BioFoundation make_datasets.common.referential_coordinates 와 같은 계산 (lmdb 등 의존성 없이)"""
    ep = _load_module(os.path.join(repo, "make_datasets", "electrode_positions.py"), "bf_electrode_positions")

    def coord(name):
        if name in ep.ELECTRODE_ANGLES:
            a = ep.ELECTRODE_ANGLES[name]
            return ep.get_electrode_3d_positions(a["theta"], a["phi"])
        if name in ep.SPECIAL_REFERENCE_POSITIONS:
            return ep.SPECIAL_REFERENCE_POSITIONS[name]
        raise KeyError(f"전극 '{name}' 의 좌표가 BioFoundation 표에 없습니다 (원본은 원점으로 대체하지만 여기서는 멈춤)")

    ref = coord(reference.upper())
    out = np.zeros((len(channels), 2, 3), dtype=np.float32)
    for i, c in enumerate(channels):
        out[i, 0] = coord(c.upper())
        out[i, 1] = ref
    return out


def minmax(x, eps=1e-6):
    """[N, C, T] 구간마다 채널별 [-1, 1]"""
    mn = x.min(axis=-1, keepdims=True)
    mx = x.max(axis=-1, keepdims=True)
    return (((x - mn) / (mx - mn + eps)) - 0.5) * 2


@register("scerebro")
class SCEReBrO(Encoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        repo, weights = cfg.get("repo"), cfg.get("weights")
        if not repo or not os.path.exists(os.path.join(repo, "models", "s_cerebro.py")):
            raise FileNotFoundError(f"BioFoundation 저장소(PR #18, commit 361c017) 경로를 encoder.repo 에 지정하세요: {repo}")
        if not weights or not os.path.exists(weights):
            raise FileNotFoundError(f"S_CEReBrO.safetensors 경로를 encoder.weights 에 지정하세요: {weights}")
        reference = cfg.get("reference")
        if not reference:
            raise ValueError("데이터셋의 기준전극(dataset yaml 의 reference)이 비어 있습니다. 예: PD-EEG CZ, 평균 기준 AR")
        import torch
        from safetensors.torch import load_file
        self.torch = torch
        sys.path.insert(0, repo)
        from models.s_cerebro import SCerebroEncoder

        arch = dict(cfg.get("arch") or {})
        arch["num_channels"] = len(cfg["channels"])
        m = SCerebroEncoder(**arch)
        sd = {}
        for k, v in load_file(weights).items():
            if k.startswith("model_head."):
                continue
            sd[k[len("model."):] if k.startswith("model.") else k] = v
        cur = m.state_dict()
        loaded = [k for k, v in sd.items() if k in cur and cur[k].shape == v.shape]
        skipped = [k for k, v in sd.items() if k in cur and cur[k].shape != v.shape]
        unexpected = [k for k in sd if k not in cur]
        missing = [k for k in cur if k not in sd]
        cur.update({k: sd[k] for k in loaded})
        m.load_state_dict(cur)
        print(f"  [S-CEReBrO] loaded={len(loaded)} shape_mismatch={len(skipped)} unexpected={len(unexpected)} "
              f"missing={len(missing)} total={len(cur)}")
        if skipped or unexpected or missing:
            print("    mismatch:", skipped[:5], "| unexpected:", unexpected[:5], "| missing:", missing[:5])
        if len(loaded) < 0.9 * len(cur):
            raise RuntimeError("사전학습 가중치가 대부분 로드되지 않았습니다 → arch 설정·가중치 파일 확인")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = m.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.coords = torch.from_numpy(referential_coordinates(repo, cfg["channels"], reference)).to(self.device)
        self.patch = int(arch.get("patch_size", 200))
        self.pool = cfg.get("pool", "channel")
        self.bs = int(cfg.get("batch_size", 64))

    def embed(self, segs):
        torch = self.torch
        N, C, T = segs.shape
        if T % self.patch:
            raise ValueError(f"입력 길이 {T} 가 패치 {self.patch} 의 배수가 아닙니다 (fs=200, seg_sec 정수)")
        x = minmax(segs.astype(np.float32)).reshape(N, C, T // self.patch, self.patch).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, N, self.bs):
                xb = torch.from_numpy(x[i:i + self.bs]).to(self.device)
                pos = self.coords.unsqueeze(0).expand(xb.shape[0], -1, -1, -1)
                z = self.model(xb, channel_positions=pos, directly_input_tokens=False, attn_mask=None)  # [B, C*P, D]
                if self.pool == "mean":
                    z = z.mean(dim=1)
                else:
                    z = z.reshape(z.shape[0], C, -1, z.shape[-1]).mean(dim=2).reshape(z.shape[0], -1)
                out.append(z.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)
