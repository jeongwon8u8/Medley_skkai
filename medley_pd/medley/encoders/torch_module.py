"""일반 PyTorch 모듈 어댑터 — CEReBrO 등 코드·가중치를 직접 받아 쓰는 인코더용

configs/encoders/cerebro.yaml 처럼 module_file / class_name / init_kwargs / checkpoint / input_layout 을 지정한다.
출력이 [B,C,P,D] 면 P 평균 후 펼침(CBraMod 와 같은 방식), [B,L,D] 면 L 평균, [B,D] 면 그대로.
"""
import importlib.util
import os
import sys

import numpy as np

from . import Encoder, register


@register("torch_module")
class TorchModule(Encoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        need = ["module_file", "class_name", "checkpoint"]
        miss = [k for k in need if not cfg.get(k)]
        if miss:
            raise ValueError(f"인코더 '{cfg['name']}' 설정이 비어 있습니다: {miss} → configs/encoders/{cfg['name']}.yaml")
        import torch
        self.torch = torch
        path = cfg["module_file"]
        sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
        spec = importlib.util.spec_from_file_location("ext_encoder_module", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = getattr(mod, cfg["class_name"])(**(cfg.get("init_kwargs") or {}))
        sd = torch.load(cfg["checkpoint"], map_location="cpu")
        if cfg.get("state_dict_key"):
            sd = sd[cfg["state_dict_key"]]
        if cfg.get("strip_prefix"):
            p = cfg["strip_prefix"]
            sd = {k[len(p):] if k.startswith(p) else k: v for k, v in sd.items()}
        res = m.load_state_dict(sd, strict=False)
        print(f"  [{cfg['name']}] 가중치 로드: missing {len(res.missing_keys)}, unexpected {len(res.unexpected_keys)}")
        if res.missing_keys:
            print("    missing 예시:", res.missing_keys[:5], "→ strip_prefix/state_dict_key 확인")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = m.to(self.device).eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.fwd = getattr(self.model, cfg["forward_method"]) if cfg.get("forward_method") else self.model
        self.bs = int(cfg.get("batch_size", 64))

    def _layout(self, x):
        if self.cfg.get("input_layout", "BCT") == "BCPT":
            pl = int(self.cfg["patch_len"])
            N, C, T = x.shape
            if T % pl:
                raise ValueError(f"입력 길이 {T} 가 patch_len {pl} 의 배수가 아닙니다")
            return x.reshape(N, C, T // pl, pl)
        return x

    @staticmethod
    def _pool(z):
        if isinstance(z, (tuple, list)):
            z = z[0]
        if z.dim() == 4:
            return z.mean(dim=2).reshape(z.shape[0], -1)
        if z.dim() == 3:
            return z.mean(dim=1)
        if z.dim() == 2:
            return z
        raise ValueError(f"지원하지 않는 출력 모양: {tuple(z.shape)}")

    def embed(self, segs):
        torch = self.torch
        x = self._layout(segs)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.bs):
                z = self.fwd(torch.from_numpy(np.ascontiguousarray(x[i:i + self.bs])).to(self.device))
                out.append(self._pool(z).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)
