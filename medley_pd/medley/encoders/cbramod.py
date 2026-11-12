"""frozen CBraMod (기존 노트북 build_cbramod / extract_features 와 동일)"""
import sys

import numpy as np

from . import Encoder, register


class _BraindecodeWrap:
    def __init__(self, m):
        self.m = m

    def parameters(self):
        return self.m.parameters()

    def __call__(self, x):
        return self.m(x.reshape(x.shape[0], x.shape[1], -1))


@register("cbramod")
class CBraMod(Encoder):
    PATCH = 200                                                 # 1초 패치 (200 Hz)

    def __init__(self, cfg):
        super().__init__(cfg)
        import torch
        self.torch = torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        repo, weights = cfg.get("repo"), cfg.get("weights")
        if repo is None:
            from braindecode.models import CBraMod as BD
            m = BD.from_pretrained("braindecode/cbramod-pretrained", return_encoder_output=True).eval().to(dev)
            for p in m.parameters():
                p.requires_grad_(False)
            self.model = _BraindecodeWrap(m)
        else:
            sys.path.insert(0, repo)
            from models.cbramod import CBraMod as Orig
            m = Orig(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
            m.load_state_dict(torch.load(weights, map_location="cpu"))
            m.proj_out = torch.nn.Identity()
            m = m.to(dev).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.model = m
        self.device = dev
        self.bs = int(cfg.get("batch_size", 64))

    def embed(self, segs):
        torch = self.torch
        N, C, T = segs.shape
        if T % self.PATCH:
            raise ValueError(f"CBraMod 입력 길이 {T} 가 {self.PATCH} 의 배수가 아닙니다 (fs=200, seg_sec 정수)")
        x = segs.reshape(N, C, T // self.PATCH, self.PATCH)
        out = []
        with torch.no_grad():
            for i in range(0, N, self.bs):
                z = self.model(torch.from_numpy(x[i:i + self.bs]).to(self.device))   # [B, C, P, 200]
                out.append(z.mean(dim=2).reshape(z.shape[0], -1).cpu().numpy())    # 패치 평균 → C*200
        return np.concatenate(out).astype(np.float32)
