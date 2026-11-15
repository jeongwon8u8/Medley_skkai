"""EEG 인코더 레지스트리

새 인코더 추가 방법
  1) medley/encoders/<이름>.py 에 Encoder 를 상속한 클래스를 만들고 @register("<type>") 를 붙인다
  2) embed(segs[N, C, T]) -> np.ndarray[N, D] 만 구현하면 된다 (전처리·캐시·평가는 공통 코드가 처리)
  3) configs/encoders/<이름>.yaml 에 type 과 입력 사양(fs, seg_sec, amp_div, channels)을 적는다
"""
import importlib

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


class Encoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg["name"]

    def embed(self, segs):
        """segs: float32 [N, C, T] (이미 μV / amp_div) → float32 [N, D]"""
        raise NotImplementedError


def build_encoder(cfg):
    for mod in ("cbramod", "scerebro", "torch_module"):
        importlib.import_module(f"{__name__}.{mod}")
    t = cfg.get("type")
    if t not in REGISTRY:
        raise KeyError(f"인코더 type '{t}' 이 없습니다. 등록된 type: {sorted(REGISTRY)}")
    return REGISTRY[t](cfg)
