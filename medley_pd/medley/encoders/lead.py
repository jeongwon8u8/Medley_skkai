"""frozen LEAD (P-Base) 어댑터 — DL4mHealth/LEAD models/LEAD.py 를 그대로 불러 사전학습 가중치로 특징 추출

- 입력: segs [N, C, T] (μV, 평균 기준, 0.5–45 Hz, 200 Hz, 2초) → 구간·채널별 z-score → LEAD 입력 [N, T, C]
- 출력 (pool):
    channel: 채널별 마지막 패치 토큰에 GELU → [N, C*128]  (채널 순서 = channels. 채널 평균하면 encode() 와 같음)
    mean:    model.encode() → [N, 128]
- 가중치는 사전학습 단계(pretrain)만 사용. projection_head 는 불러오지만 쓰지 않음
"""
import sys
from types import SimpleNamespace

import numpy as np

from . import Encoder, register


def _zscore(segs):
    """LEAD data_provider/uea.py normalize_batch_ts 와 같은 구간·채널별 표준화 (std 0 이면 1로)"""
    m = segs.mean(axis=2, keepdims=True)
    s = segs.std(axis=2, keepdims=True)
    s[s == 0] = 1.0
    return (segs - m) / s


@register("lead")
class LEAD(Encoder):
    DATASET_ID = 0

    def __init__(self, cfg):
        super().__init__(cfg)
        import torch
        self.torch = torch
        repo, weights = cfg.get("repo"), cfg.get("weights")
        if not repo or not weights:
            raise ValueError("LEAD: configs/local.yaml 에 encoders.lead.repo(LEAD 클론 경로)와 weights(checkpoint.pth)를 지정하세요")
        sys.path.insert(0, repo)
        from models.LEAD import Model                            # LEAD 저장소의 원본 모델

        a = cfg.get("lead_arch") or {}
        self.fs = int(cfg["fs"])
        rates = [int(r) for r in a.get("sampling_rate_list", [200, 100, 50])]
        if self.fs not in rates:
            raise ValueError(f"LEAD 사전학습 샘플링 {rates} 에 fs={self.fs} 가 없습니다")
        mc = SimpleNamespace(
            task_name="pretrain",                                # 사전학습과 같은 구조 → 가중치를 빠짐없이 불러옴
            output_attention=False,
            patch_len=a.get("patch_len", 50), stride=a.get("stride", 50),
            enc_in=len(cfg["channels"]), d_model=a.get("d_model", 128), n_heads=a.get("n_heads", 8),
            d_ff=a.get("d_ff", 256), e_layers=a.get("e_layers", 12), dropout=a.get("dropout", 0.1),
            activation=a.get("activation", "gelu"), num_class=2,
            contrastive_token_ratio=1.0,
            augmentations="none",
            channel_names_by_id={self.DATASET_ID: list(cfg["channels"])},
            montage_by_id={self.DATASET_ID: cfg.get("montage", "standard_1005")},
            use_sampling_embedding=bool(a.get("use_sampling_embedding", True)),
            sampling_rate_to_id={r: i for i, r in enumerate(rates)},
            temporal_pos_type=a.get("temporal_pos_type", "learnable"),
            channel_pos_type=a.get("channel_pos_type", "3D"),
            max_patch_positions=a.get("max_patch_positions", 1024),
            max_channel_positions=a.get("max_channel_positions", 368),
        )
        m = Model(mc)

        sd = torch.load(weights, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        own = m.state_dict()
        bad = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
        if bad:
            raise RuntimeError(f"LEAD 가중치 모양 불일치 (lead_arch 확인): {bad[:5]}")
        res = m.load_state_dict(sd, strict=False)
        # 공개 P-Base 에는 대조학습용 projection_head 가 빠져 있음 → 특징 추출(encode·채널 토큰)에는 쓰지 않으므로 허용
        head = ("projection_head.", "classifier.")
        missing = [k for k in res.missing_keys if not k.startswith(head)]
        n_enc = sum(1 for k in sd if not k.startswith(head))
        print(f"  [lead] 가중치 로드: 인코더 텐서 {n_enc}개, missing {len(missing)} (헤드 {len(res.missing_keys) - len(missing)}개 제외), "
              f"unexpected {len(res.unexpected_keys)}")
        if missing or res.unexpected_keys:
            print("    missing:", missing[:10], "unexpected:", res.unexpected_keys[:10])
            raise RuntimeError("LEAD 가중치가 모델과 맞지 않습니다. lead_arch 설정과 checkpoint 경로를 확인하세요")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = m.to(self.device).eval()                    # eval: dropout 끔, 증강 없음
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.pool = cfg.get("pool", "channel")
        if self.pool not in ("channel", "mean"):
            raise ValueError(f"LEAD pool 은 channel | mean: {self.pool}")
        self.bs = int(cfg.get("batch_size", 256))

    def embed(self, segs):
        torch = self.torch
        x = _zscore(segs.astype(np.float32)).transpose(0, 2, 1)   # [N, C, T] → [N, T, C]
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.bs):
                xb = torch.from_numpy(np.ascontiguousarray(x[i:i + self.bs])).to(self.device)
                n = xb.shape[0]
                # label_id 열: [disease_id, subject_id, sampling_frequency, dataset_id] — 질환·피험자 열은 쓰이지 않음
                lid = torch.zeros(n, 4, device=self.device)
                lid[:, 2] = self.fs
                lid[:, 3] = self.DATASET_ID
                if self.pool == "mean":
                    z = self.model.encode(xb, lid)                                   # [n, D]
                else:
                    enc, C, P, _, _ = self.model._encode_full(xb, lid, apply_augmentation=False)
                    z = self.model.act(self.model._last_token_per_channel(enc, C, P))   # [n, C, D]
                    z = z.reshape(n, -1)                                             # [n, C*D]
                out.append(z.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)
