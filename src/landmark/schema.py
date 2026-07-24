"""
LANmark CONFIGuration — defines the project facial landmarks and symmetry relationships.

Point naming convention:
  - Bilateral points (left/right): {name}_L, {name}_R
  - Midline points: {name}

Landmark index order (0-indexed, used throughout the project):
  [0]  ebe_L    左眉尾点
  [1]  ebe_R    右眉尾点
  [2]  ebs_L    左眉头点
  [3]  ebs_R    右眉头点
  [4]  os_L     左眉下缘最高点
  [5]  os_R     右眉下缘最高点
  [6]  sci_L    左眉上缘最高点
  [7]  sci_R    右眉上缘最高点
  [8]  en_L     左内眦点
  [9]  en_R     右内眦点
  [10] p_L      左瞳孔中心
  [11] p_R      右瞳孔中心
  [12] ex_L     左外眦点
  [13] ex_R     右外眦点
  [14] Na       鼻根点 (nasion)
  [15] prn      鼻尖点 (pronasale)
  [16] Sn       鼻下点 (subnasale)
  [17] al_L     左鼻翼点
  [18] al_R     右鼻翼点
  [19] cph_L    左侧人中嵴点
  [20] cph_R    右侧人中嵴点
  [21] ch_L     左口裂点 (cheilion)
  [22] ch_R     右口裂点 (cheilion)
  [23] ls       上唇中点 (labrale superius)
  [24] sto      口裂中点 (stomion)
  [25] li       下唇中点 (labrale inferius)
  [26] Go_L     左下颌角点 (gonion)
  [27] Go_R     右下颌角点 (gonion)
  [28] Me       颏下点 (menton)

Symmetry pairs:
  Each pair (L, R) indexes into the list above.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# 29个关键点的标准顺序定义
# 双侧点：名称_L（左侧）/ 名称_R（右侧），按x坐标判断左右
# 正中线点：单一名称
# ============================================================
# --- canonical order ---
# fmt: off
POINT_NAMES_31 = [
    "ebe_L", "ebe_R",       # 0, 1
    "ebs_L", "ebs_R",       # 2, 3
    "os_L", "os_R",         # 4, 5
    "sci_L", "sci_R",       # 6, 7
    "en_L", "en_R",         # 8, 9
    "p_L", "p_R",           # 10, 11
    "ex_L", "ex_R",         # 12, 13
    "Na",                   # 14
    "prn",                  # 15
    "Sn",                   # 16
    "al_L", "al_R",         # 17, 18
    "cph_L", "cph_R",       # 19, 20
    "ch_L", "ch_R",         # 21, 22
    "ls",                   # 23
    "sto",                  # 24
    "li",                   # 25
    "Go_L", "Go_R",         # 26, 27
    "Me",                   # 28
]
# fmt: on

NUM_POINTS = len(POINT_NAMES_31)  # 29

# Name → index mapping
NAME_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(POINT_NAMES_31)}

# labelme 标注标签 → 本系统点名的映射
# 注意：双侧标签在JSON中出现两次，通过x坐标区分左右（x小为左，x大为右）
LABELME_TAG_MAP: Dict[str, str] = {
    "ebe": "ebe",
    "ebs": "ebs",
    "os": "os",
    "sci": "sci",
    "en": "en",
    "p": "p",
    "ex": "ex",
    "al": "al",
    "cph": "cph",
    "ch": "ch",
    "Go": "Go",
    "Na": "Na",
    "prn": "prn",
    "Sn": "Sn",
    "ls": "ls",
    "sto": "sto",
    "li": "li",
    "Me": "Me",
}

# 正中线点标签（在JSON中只出现一次）
MIDLINE_TAGS = {"Na", "prn", "Sn", "ls", "sto", "li", "Me"}

# 双侧对称点标签（在JSON中出现两次，左右各一）
BILATERAL_TAGS = {"ebe", "ebs", "os", "sci", "en", "p", "ex", "al", "cph", "ch", "Go"}


# --- Symmetry pairs (left_idx, right_idx) ---
# Each tuple is (left_index, right_index) in the POINT_NAMES_31 list.
SYMMETRY_PAIRS: List[Tuple[int, int]] = [
    (0, 1),    # ebe
    (2, 3),    # ebs
    (4, 5),    # os
    (6, 7),    # sci
    (8, 9),    # en
    (10, 11),  # p
    (12, 13),  # ex
    (17, 18),  # al
    (19, 20),  # cph
    (21, 22),  # ch
    (26, 27),  # Go
]

# Midline point indices (self-symmetric)
MIDLINE_INDICES = [14, 15, 16, 23, 24, 25, 28]  # Na, prn, Sn, ls, sto, li, Me

# All midline + bilateral pairs combined for axis fitting
ALL_SYMMETRY_GROUPS = list(SYMMETRY_PAIRS) + [(i, i) for i in MIDLINE_INDICES]

# 按解剖学区域分组，用于后续分区对称性评分。
# 根据用户最新确认，os/sci 属于眉部，不属于眼部。
REGIONS: Dict[str, List[int]] = {
    "eyebrow": [0, 1, 2, 3, 4, 5, 6, 7],
    "eye": [8, 9, 10, 11, 12, 13],
    "nose": [14, 15, 16, 17, 18],
    "mouth": [19, 20, 21, 22, 23, 24, 25],
    "jaw": [26, 27, 28],
}


def get_point_name(idx: int) -> str:
    return POINT_NAMES_31[idx]


def get_point_idx(name: str) -> int:
    return NAME_TO_IDX[name]


def is_bilateral(idx: int) -> bool:
    name = POINT_NAMES_31[idx]
    return "_L" in name or "_R" in name


def is_midline(idx: int) -> bool:
    return idx in MIDLINE_INDICES


def get_symmetric_partner(idx: int) -> Optional[int]:
    for l, r in SYMMETRY_PAIRS:
        if idx == l:
            return r
        if idx == r:
            return l
    if idx in MIDLINE_INDICES:
        return idx
    return None


@dataclass
class LandmarkSchema:
    num_points: int = NUM_POINTS
    point_names: List[str] = field(default_factory=lambda: POINT_NAMES_31.copy())
    symmetry_pairs: List[Tuple[int, int]] = field(default_factory=lambda: SYMMETRY_PAIRS.copy())
    midline_indices: List[int] = field(default_factory=lambda: MIDLINE_INDICES.copy())
    regions: Dict[str, List[int]] = field(default_factory=lambda: {k: v.copy() for k, v in REGIONS.items()})
    labelme_tag_map: Dict[str, str] = field(default_factory=lambda: LABELME_TAG_MAP.copy())
    bilateral_tags: set = field(default_factory=lambda: BILATERAL_TAGS.copy())
    midline_tags: set = field(default_factory=lambda: MIDLINE_TAGS.copy())
