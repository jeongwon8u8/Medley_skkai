"""EEG 파일 찾기·읽기·전처리 (기존 노트북 load_raw / preprocess 를 데이터셋·인코더 설정으로 일반화)

공통: reference 유지 → 채널 이름으로 선택(10-20 옛 이름 T3/T4/T5/T6 → 10-10) → 없는/평평한 채널 보간
     → l_freq–h_freq 대역통과 → line_freq 노치 → 인코더 fs로 리샘플링 → seg_sec 비중첩 구간 → μV / amp_div
"""
import fnmatch
import os
import re

import numpy as np

ALIAS = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8"}


# ----------------------------------------------------------------------------- 파일 찾기
class FileIndex:
    """데이터셋 폴더를 한 번만 훑어서 피험자별 파일을 찾음 (대소문자 무시)"""

    def __init__(self, root, ds_cfg):
        self.pattern = ds_cfg.get("file_pattern", "{sid}_*task-rest*_eeg.*").lower()
        self.exts = tuple("." + e.lower() for e in ds_cfg.get("file_ext", ["set", "edf", "bdf", "vhdr", "fif"]))
        self.files = []
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(self.exts):
                    self.files.append(os.path.join(dp, f))

    def find(self, sid):
        pat = self.pattern.format(sid=sid.lower())
        hits = [p for p in self.files if fnmatch.fnmatch(os.path.basename(p).lower(), pat)]
        hits.sort(key=lambda h: ("derivatives" in h.lower(), h))
        return hits[0] if hits else None


# ----------------------------------------------------------------------------- 읽기·채널
def read_raw(path):
    import mne
    if path.lower().endswith(".set"):
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    return mne.io.read_raw(path, preload=True, verbose="ERROR")


def _norm(ch):
    c = ch.lower().replace("eeg", "").strip(" -_")
    c = re.split(r"[-_ ]", c)[0]
    return ALIAS.get(c, c)


def load_raw(path, channels, ref_aliases=(), min_good=20):
    """channels 를 이름으로 골라 그 순서로 정렬. 없거나 신호가 0인 채널은 보간. (raw, 보간 채널) 반환"""
    import mne
    raw = read_raw(path)
    mont = mne.channels.make_standard_montage("standard_1005")
    canon = {n.lower(): n for n in mont.ch_names}
    rename, used = {}, set()
    for c in raw.ch_names:
        k = _norm(c)
        if k in canon and canon[k] not in used:
            rename[c] = canon[k]
            used.add(canon[k])
    refs = {r.upper() for r in ref_aliases or ()}
    if "Cz" not in used and refs:                              # 기준전극 채널이 Cz 위치인 장비 (EGI VREF)
        for c in raw.ch_names:
            if c.upper() in refs and c not in rename:
                rename[c] = "Cz"
                used.add("Cz")
                break
    raw = raw.pick(list(rename)).rename_channels(rename)
    raw.set_channel_types({c: "eeg" for c in raw.ch_names}, verbose="ERROR")
    missing = [t for t in channels if t not in raw.ch_names]
    if missing:
        info = mne.create_info(missing, raw.info["sfreq"], "eeg")
        raw.add_channels([mne.io.RawArray(np.zeros((len(missing), raw.n_times)), info, verbose="ERROR")],
                         force_update_info=True)
    raw.set_montage(mont, match_case=False, on_missing="ignore", verbose="ERROR")
    x = raw.get_data(picks=channels)
    flat = [t for t, s in zip(channels, x.std(1)) if s < 1e-9 and t not in missing]
    bads = missing + flat
    if bads:
        if len(raw.ch_names) - len(bads) < min_good:
            raise RuntimeError(f"보간에 쓸 채널이 부족합니다 ({len(raw.ch_names) - len(bads)}개): {raw.ch_names}")
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True, origin="auto", verbose="ERROR")
    raw.pick(channels)
    return raw, bads


# ----------------------------------------------------------------------------- 전처리
def check_encoder_spec(enc):
    need = ["fs", "seg_sec", "amp_div", "channels"]
    miss = [k for k in need if enc.get(k) in (None, "", [])]
    if miss:
        raise ValueError(f"인코더 '{enc['name']}' 입력 사양이 비어 있습니다: {miss} → configs/encoders/{enc['name']}.yaml")


def preprocess(path, cfg):
    """반환: segs [N, C, T] (μV / amp_div, float32), flag [N] (100 μV 초과 구간), 정보 dict"""
    pp, ds, enc = cfg["preprocess"], cfg["dataset"], cfg["encoder"]
    check_encoder_spec(enc)
    raw, interp = load_raw(path, enc["channels"], ds.get("ref_aliases", []), pp.get("min_good_channels", 20))
    fs_orig = raw.info["sfreq"]
    h = min(pp["h_freq"], fs_orig / 2 - 1)
    raw.filter(pp["l_freq"], h, verbose="ERROR")
    notch = pp.get("notch") or ds.get("line_freq")
    if notch and notch < fs_orig / 2:
        raw.notch_filter(notch, verbose="ERROR")
    fs = int(enc["fs"])
    if abs(fs_orig - fs) > 1e-6:
        raw.resample(fs, verbose="ERROR")
    x = raw.get_data() * 1e6                                   # V → μV
    L = int(round(fs * float(enc["seg_sec"])))
    n = x.shape[1] // L
    C = len(enc["channels"])
    segs = x[:, : n * L].reshape(C, n, L).transpose(1, 0, 2)  # [N, C, T]
    flag = np.abs(segs).reshape(n, -1).max(1) > pp["flag_uv"] if n else np.zeros(0, bool)
    info = {"fs_orig": fs_orig, "dur_s": round(x.shape[1] / fs, 1), "n_seg": int(n), "interp": "+".join(interp)}
    return (segs / float(enc["amp_div"])).astype(np.float32), flag, info
