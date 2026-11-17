"""CrisisDynNet 统一配置（Nature Human Behaviour 投稿）。

所有路径、随机种子、设计参数都集中在这里，其余脚本一律通过 ``import config as C`` 读取，
不再各自拼接路径。改目录只需要改本文件（或用下面列出的环境变量）。

目录约定
--------
    <PROJECT>/                     "Nature Human Behaviour"
    ├─ data/                       前瞻队列数据（11 张表，格式见 docs/数据约定.md）
    ├─ manuscript/                 稿件模板与生成稿（步骤 13 写入）
    ├─ review_rounds/              自动合理性审查记录（步骤 10 写入）
    └─ daima/                      本代码目录
       ├─ config.py ← 你在这里
       ├─ run_all.py               一键运行
       ├─ pipeline/                00–14 各步骤脚本
       ├─ common/                  共享模块（生成模型、真实性审查、框架、基线）
       ├─ tools/                   数据体检、表头导出等辅助工具
       ├─ preregistration/         预注册冻结快照（只读归档）
       └─ outputs/                 全部运行产物
          ├─ results/  figures/  logs/  cache/

环境变量（可选，优先级最高）
----------------------------
    NHB_DATA_DIR      前瞻队列数据目录，默认 <PROJECT>/data
    NHB_PUBLIC_DIR    公开数据集根目录，默认 E:\\data
    NHB_AFFECT_DIR    情感数据集根目录，默认 E:\\生理情感数据库
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- 目录
HERE = Path(__file__).resolve().parent          # daima/
PROJECT = HERE.parent                           # "Nature Human Behaviour"
STEP_DIR = HERE / "pipeline"
COMMON_DIR = HERE / "common"
TOOLS_DIR = HERE / "tools"
PREREG = HERE / "preregistration"

OUT = HERE / "outputs"
RES = OUT / "results"                           # 所有结果表 / json
FIG = OUT / "figures"                           # 图 1–5（pdf + png）
CACHE = OUT / "cache"                           # 生成器真值、真实特征缓存（不进 data/）
LOGS = OUT / "logs"                             # 运行日志与流水线状态

MANUSCRIPT = PROJECT / "manuscript"             # 模板与生成稿
REVIEW = PROJECT / "review_rounds"              # 自动审查记录

DATA = Path(os.environ.get("NHB_DATA_DIR") or (PROJECT / "data")).resolve()
COHORT = DATA                                    # 兼容旧脚本中的 C.COHORT 写法

DATA_ROOT = Path(os.environ.get("NHB_PUBLIC_DIR") or r"E:\data")          # 公开数据（只读）
AFFECT_ROOT = Path(os.environ.get("NHB_AFFECT_DIR") or r"E:\生理情感数据库")  # DEAP/DREAMER（只读）

for _p in (RES, FIG, CACHE, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20261116

# ---------------------------------------------------------------- 前瞻队列数据表
# 11 张表的文件名与列名固定（格式见 docs/数据约定.md）；tools/check_data.py 按同一份约定体检。
COHORT_TABLES = (
    "academic_calendar", "counselor_assessments", "crisis_events", "daily_ema", "daily_wearable",
    "lab_epochs", "lab_sessions", "participants", "screening_baseline", "wave_questionnaires",
    "weekly_phq4",
)

# 生成器的真值缓存（亚型、起搏日、潜变量轨迹、类间耦合矩阵），只用于设计检验，不进 data/
TRUTH_NPZ = CACHE / "cohort_truth.npz"
TRUTH_GRAPHS = CACHE / "cohort_truth_graphs.json"


def has_truth():
    """真值缓存是否可用；缺失时与真值有关的检验自动跳过。"""
    return TRUTH_NPZ.exists() and TRUTH_GRAPHS.exists()


# ---------------------------------------------------------------- 公开数据集（只读）
PUBLIC = {
    "xinxiang": DATA_ROOT / "01_Xinxiang_PHQ9_GAD7_ISI_PSS",
    "freshmen22": DATA_ROOT / "47_Chinese_freshmen_3level_screening",
    "ces": DATA_ROOT / "26_College_Experience_Study" / "extracted",
    "nethealth": DATA_ROOT / "15_NetHealth_NotreDame_longitudinal",
    "lifesnaps": DATA_ROOT / "17_LifeSnaps_Fitbit_EMA" / "extracted" / "rais_anonymized" / "csv_rais_anonymized",
    "psyched": DATA_ROOT / "60_PSYCHE-D_Fitbit_PHQ9_monthly" / "anon_processed_df_parquet",
    "ssaqs": DATA_ROOT / "14_SSAQS_student_stress_Fitbit" / "extracted",
    "ecsmp": DATA_ROOT / "08_ECSMP_Chinese_students",
    "wesad": DATA_ROOT / "14_WESAD_lab_TSST" / "WESAD.zip",
    "exam": DATA_ROOT / "03_PhysioNet_wearable_exam_stress" / "extracted",
    "studentlife": DATA_ROOT / "09_StudentLife" / "rdata_zenodo_mirror",
    "dapper": DATA_ROOT / "13_DAPPER_daily_physiology" / "extracted",
    "deap": AFFECT_ROOT / "DEAP" / "data_preprocessed_python",
    "dreamer": AFFECT_ROOT / "Dreamer" / "DREAMER.mat",
}
# 有意不读的数据集：
#  - 78 CFPS：许可禁止把（衍生）数据交给云端 AI，只能作者本机分析。
#  - 73 JCU：数据仅授予作者本人；可选用，只输出汇总值。
#  - 54 完美主义：在 E:\data\00_数据说明.md 中被标记为不可信。

# ---------------------------------------------------------------- 前瞻队列设计
COHORT_DESIGN = dict(
    start_date="2025-09-15",     # 第 0 天 = 2025 秋季学期第一教学周周一
    n_screen=6000,               # 全校筛查池
    n_enrolled=320,              # 富集入组（2/3 筛查阳性、1/3 筛查阴性）
    n_days=182,                  # 6 个月日度生态采样
    waves=(0, 91, 181),          # 实验室波次 T0/T1/T2（第几天）
    exam_windows=((56, 70), (105, 119)),   # 期中、期末
    winter_break=(121, 163),
    golden_week=(17, 24),
    job_window=(7, 77),          # 秋招窗口（仅毕业年级）
    lab_epoch_s=10,              # EEG/ECG/EDA 特征按 10 s 时段
    lab_phases=(("rest", 30), ("stress", 36), ("recovery", 30)),  # 各阶段时段数（5/6/5 分钟）
)
SUBTYPES = ("stable", "acute", "chronic", "latent")     # 内部编码，见 实验设计方案.md §2

# ---------------------------------------------------------------- 校准（仅生成数据时使用）
CAL_ROUNDS = 20
REALISM_AUC_TARGET = 0.70        # 真实/生成判别 AUC 高于此值视为"一眼看出是生成的"

# ---------------------------------------------------------------- 统计
ALPHA = 0.05
N_PERM = 1000
N_BOOT = 200
POWER_NS = (160, 200, 240, 280, 320)
POWER_REPS = 40
