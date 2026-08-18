#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор СИНТЕТИЧЕСКИХ данных контрольных поручений (учебная курсовая ML-Basic).
Задача: бинарная классификация "будет ли просрочено контрольное поручение".

РЕАЛЬНЫЕ ДАННЫЕ ЗАКАЗЧИКА НЕ ИСПОЛЬЗУЮТСЯ — всё ниже сгенерировано случайно,
включая имена пользователей (псевдонимы) и тексты поручений (шаблоны).

────────────────────────────────── v3: фикс историчности ──────────────────────────────────
В v2 у ~45% поручений была ОДНА версия [создание, ∞), которая сразу несла финальный
исход (is_done=true, date_of_execution) — as-of-запрос valid_period @> T для момента T
внутри жизни поручения возвращал уже закрытое состояние → утечка будущего в признаки
"на момент T". В v3 ГЛАВНЫЙ ИНВАРИАНТ: ни одно ИСПОЛНЕННОЕ поручение не несёт свой
финальный исход в версии, покрывающей момент создания. У каждого исполненного
поручения теперь ОБЯЗАТЕЛЬНО есть ранняя версия "в работе" (is_done=false,
date_of_execution=NULL), и лишь ПОЗЖЕ — терминальная версия с исходом. Открытые
(ещё не исполненные) поручения могут оставаться одноверсионными — утечки там нет,
т.к. версия не несёт исхода.

Также в v3: workload_at_t0 (число открытых поручений исполнителя) сделан РЕАЛЬНЫМ
драйвером риска (в v2 был минорным), а карьерность/невнимательность исполнителя
(carelessness) введена как отдельная латентная черта, коррелирующая с tendency
(ρ≈0.6) и управляющая числом возвратов "На доработку".

────────────────────────────── v3.1: стационарность нагрузки ──────────────────────────────
В v3 нагрузка КОПИЛАСЬ от старта системы из пустого состояния: "Просрочено"-открытые и
снятые с контроля поручения считались открытыми ВЕЧНО (is_done=false), их число росло со
временем → workload дрейфовал вверх (1.4→7.6 по годам), а вместе с ним и overdue_rate
(0.15→0.45). Это артефакт конечного окна наблюдения, а не свойство процесса. В v3.1:
(1) поручение выходит из активной нагрузки в момент close_time = date_of_execution
(исполнено) | время снятия (deleted) | +∞ (открыто на срез) — снятые больше не висят вечно;
(2) просроченное поручение исполняется поздно через Exponential(mean=LATE_EXEC_MEAN_DAYS)
дней после дедлайна (давно просроченные закрываются, открытыми остаются только свежие).
Итог: нагрузка СТАЦИОНАРНА (спред по годам ~1.2× вместо 5.4×), overdue_rate почти не
дрейфует, а предсказательная сила нагрузки идёт от РЕАЛЬНОЙ перегрузки "здесь и сейчас"
(усилен COEF_WORK), а не от паразитной корреляции со временем.

Запуск: python generate_data.py (интерпретатор окружения проекта с pandas/sklearn/psycopg)
"""
import calendar
import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.range import Range
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ────────────────────────────── конфигурация ──────────────────────────────
SEED = 42
random.seed(SEED)
RNG = np.random.default_rng(SEED)

DSN = "host=127.0.0.1 port=5432 dbname=coursework user=ml password=ml_pw"

N_USERS = 150  # меньше исполнителей -> больше поручений на каждого -> устойчивый target-encoding
N_ORDERS = 10_000
BATCH_ID = "synthetic_gen_v3"

DATE_MIN = datetime(2023, 1, 1, tzinfo=timezone.utc)
DATE_MAX = datetime(2026, 6, 30, tzinfo=timezone.utc)
SPAN_DAYS = (DATE_MAX - DATE_MIN).days

# Дата среза витрины ("сегодня" витрины) — ФИКСИРОВАННАЯ и воспроизводимая.
# Всё, что "просрочено"/"на исполнении"/days_expired, считается ОТНОСИТЕЛЬНО неё,
# а не относительно реального времени запуска скрипта.
AS_OF = datetime(2026, 8, 2, tzinfo=timezone.utc)
REAL_RUN_TS = datetime.now(timezone.utc)  # только для служебного поля generated_at в meta

TYPE_NAMES = [
    "Поручение Мэра",
    "Протокольное поручение",
    "Поручение Заместителя Мэра",
    "Обращение граждан",
    "Депутатский запрос",
    "Поручение по итогам совещания",
    "Контроль реализации проекта",
    "Прочее",
]
# базовые скрытые сдвиги риска по типам (до применения общего SCALE)
TYPE_EFFECTS_BASE = [-0.35, 0.25, -0.15, 0.15, 0.05, 0.30, 0.40, 0.10]

STATUS_ON_TIME_DONE = "Исполнено в срок"
STATUS_LATE_DONE = "Исполнено с нарушением срока"
STATUS_IN_PROGRESS = "На исполнении"
STATUS_OVERDUE_OPEN = "Просрочено"
STATUS_CANCELLED = "Снято с контроля"
STATUS_REWORK = "На доработку"  # новый статус v3 (схема не меняется — status свободный TEXT)

# ── калиброванные коэффициенты логита ──
# Калибровка v3: workload_at_t0 теперь СЧИТАЕТСЯ ЧЕСТНО одним хронологическим
# проходом по РЕАЛЬНОМУ (а не проксированному через deadline) времени исполнения
# более ранних поручений того же исполнителя (t_exec > T ИЛИ поручение не исполнено),
# и участвует в логите как ЗАМЕТНЫЙ (не минорный) компонент. Коэффициенты подобраны
# сеткой отдельным калибровочным прогоном (LogisticRegression, time-split 70/30 по order_created_at,
# признаки УРОВНЯ СТУДЕНТА: структурные + target-encoding истории исполнителя ± workload):
#   без workload: honest AUC ~ 0.79      с workload: honest AUC ~ 0.83 (Δ ~ +0.04)
#   overdue_rate ~ 0.28
SCALE = 1.3
TEND_MULT = 1.15  # усиление tendency, чтобы она была восстановима через target-encoding
RHO_CARELESS = 0.6  # корреляция carelessness с tendency

COEF_SLACK = 1.4 * SCALE
SLACK_REF = 45.0
SLACK_SCALE = 35.0
COEF_TEND = 0.55 * SCALE * TEND_MULT
COEF_CARELESS = 0.25 * SCALE
COEF_CO = 0.12 * SCALE
COEF_DESC = 0.45 * SCALE
COEF_WORK = 0.60 * SCALE  # v3.1: усилен — после устранения паразитной корреляции нагрузки со
# временем (снятые/просроченные больше не копятся вечно) её предсказательная сила должна идти
# от РЕАЛЬНОЙ перегрузки исполнителя «здесь и сейчас», а не от дрейфа; входит в проверку AUC
COEF_FRIDAY = 0.15 * SCALE
COEF_EOM = 0.15 * SCALE
TYPE_EFFECTS = [e * SCALE for e in TYPE_EFFECTS_BASE]
INTERCEPT = -4.15
NOISE_SIGMA = 0.5

# число возвратов "На доработку" ~ Poisson(exp(RETURNS_A + RETURNS_B*carelessness)), cap 0..3
RETURNS_A = -0.7
RETURNS_B = 0.6
MIN_SEGMENT_SECONDS = 3600.0  # версии в цепочке возвратов разнесены на >= 1 час

# v3.1 (фикс временно́го дрейфа нагрузки): задержка позднего исполнения просроченного
# поручения ~ Exponential(mean=LATE_EXEC_MEAN_DAYS) дней ПОСЛЕ дедлайна. Если срок
# позднего исполнения укладывается до даты среза — поручение получает date_of_execution
# ("Исполнено с нарушением срока") и ВЫХОДИТ из активной нагрузки исполнителя; иначе к
# дате среза оно всё ещё "Просрочено" (открыто). Так давно просроченные поручения со
# временем закрываются, а не копятся ВЕЧНОЙ нагрузкой — нагрузка становится
# СТАЦИОНАРНОЙ, и overdue_rate перестаёт дрейфовать вверх по годам (артефакт старта
# системы из пустого состояния). Метка (overdue=1) при этом не меняется: и позднее
# исполнение, и остающаяся открытость — обе просрочка.
LATE_EXEC_MEAN_DAYS = 30.0

THEME_TOPICS = [
    "развитию инвестиционной инфраструктуры",
    "реализации мероприятий промышленной политики",
    "подготовке проекта нормативного акта",
    "рассмотрению обращения граждан",
    "исполнению протокольных решений совещания",
    "организации выездного совещания",
    "подготовке отчёта о реализации проекта",
    "актуализации плана мероприятий",
    "согласованию проектной документации",
    "подготовке ответа депутату",
    "контролю сроков реализации инвестпроекта",
    "подготовке материалов к заседанию комиссии",
    "устранению замечаний по итогам проверки",
    "подготовке предложений по мерам поддержки",
    "формированию сводной информации",
]

DESC_SENTENCES = [
    "Необходимо проработать вопрос с профильным департаментом.",
    "Требуется подготовить сводную справку по итогам исполнения.",
    "Обеспечить контроль сроков реализации мероприятий.",
    "Провести согласование с заинтересованными сторонами.",
    "Подготовить и направить ответ заявителю в установленном порядке.",
    "Организовать рабочую встречу с участием ответственных лиц.",
    "Актуализировать план-график по проекту с учётом текущего статуса.",
    "Учесть замечания, поступившие от согласующих подразделений.",
    "Представить промежуточные результаты на очередном совещании.",
    "Уточнить у исполнителя причины отклонения от плановых сроков.",
]
DESC_SHORT = ["См. поручение.", "Без описания.", "Уточнить позже.", "См. вложение.", "Экспресс-контроль."]

PROGRESS_SENTENCES = [
    "Направлен запрос в профильное подразделение.",
    "Получены материалы от соисполнителей, идёт обработка.",
    "Подготовлен проект ответа, на согласовании.",
    "Проведено совещание по вопросу исполнения.",
    "Уточняются сроки готовности документов.",
    "Материалы направлены на подпись руководителю.",
    "Исполнение продлено по объективным причинам.",
]
REWORK_SENTENCES = [
    "Возвращено на доработку: замечания по содержанию ответа.",
    "Возвращено на доработку: требуется уточнить сроки и ответственных.",
    "Возвращено на доработку: не учтены замечания согласующих подразделений.",
    "Возвращено на доработку куратором.",
]

# ─────────────────────────────── утилиты ───────────────────────────────
def rand_uuid() -> uuid.UUID:
    """Детерминированный UUID (использует seeded random, НЕ os.urandom)."""
    return uuid.UUID(int=random.getrandbits(128), version=4)


def random_dt(day_offset_max: int) -> datetime:
    d = int(RNG.integers(0, day_offset_max))
    h = int(RNG.integers(0, 24))
    m = int(RNG.integers(0, 60))
    return DATE_MIN + timedelta(days=d, hours=h, minutes=m)


def is_eom(d: datetime) -> bool:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return d.day >= last_day - 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fmt_range(start: datetime, end: datetime | None) -> Range:
    return Range(lower=start, upper=end, bounds="[)")


def build_prechain_starts(t0: datetime, terminal_start: datetime, n_segments: int) -> list[datetime]:
    """n_segments >= 1 стартов версий на [t0, terminal_start), каждая шириной >= 1 час
    (кроме тривиального случая n_segments==1, где нет ограничения на ширину — просто
    один сегмент [t0, terminal_start)). Гарантированно строго возрастают."""
    if n_segments <= 1:
        return [t0]
    span = (terminal_start - t0).total_seconds()
    total_base = MIN_SEGMENT_SECONDS * n_segments
    extra = max(0.0, span - total_base)
    weights = RNG.dirichlet(np.ones(n_segments))
    widths = MIN_SEGMENT_SECONDS + weights * extra
    starts = [t0]
    cum = 0.0
    for w in widths[:-1]:
        cum += w
        starts.append(t0 + timedelta(seconds=cum))
    return starts


def max_returns_for_span(span_seconds: float) -> int:
    """Наибольшее n, для которого (2n+1) сегментов по >= 1 часу помещаются в span."""
    if span_seconds < MIN_SEGMENT_SECONDS:
        return 0
    n = int((span_seconds / MIN_SEGMENT_SECONDS - 1) // 2)
    return max(0, n)


# ═══════════════════════════════ 1. Подключение и очистка ═══════════════════════════════
conn = psycopg.connect(DSN, autocommit=False)
cur = conn.cursor()
cur.execute(
    "TRUNCATE control_order_progress_v, control_orders_user_roles_v, "
    "control_orders_v, control_order_types, users RESTART IDENTITY CASCADE;"
)
conn.commit()
print("Таблицы очищены (TRUNCATE ... RESTART IDENTITY CASCADE).")

# ═══════════════════════════════ 2. users ═══════════════════════════════
user_ids = [rand_uuid() for _ in range(N_USERS)]
user_names = [f"user_{i+1:04d}" for i in range(N_USERS)]
tendency = RNG.normal(0, 1, size=N_USERS)  # латентная склонность к просрочке
careless_eps = RNG.normal(0, 1, size=N_USERS)
carelessness = RHO_CARELESS * tendency + np.sqrt(1 - RHO_CARELESS**2) * careless_eps  # латентная невнимательность

with cur.copy("COPY users (elma_id, name) FROM STDIN") as cp:
    for uid, name in zip(user_ids, user_names):
        cp.write_row((uid, name))
conn.commit()
print(f"users: вставлено {N_USERS}")

# ═══════════════════════════════ 3. control_order_types ═══════════════════════════════
type_ids = [rand_uuid() for _ in range(len(TYPE_NAMES))]
with cur.copy("COPY control_order_types (elma_id, name) FROM STDIN") as cp:
    for tid, name in zip(type_ids, TYPE_NAMES):
        cp.write_row((tid, name))
conn.commit()
print(f"control_order_types: вставлено {len(TYPE_NAMES)}")

# ═══════════════════════════════ 4. Генерация признаков объектов (порядок не важен) ═══════════════════════════════
order_bks = [rand_uuid() for _ in range(N_ORDERS)]
order_created_at = [random_dt(SPAN_DAYS) for _ in range(N_ORDERS)]

type_idx = RNG.integers(0, len(TYPE_NAMES), size=N_ORDERS)

primary_user_idx = RNG.integers(0, N_USERS, size=N_ORDERS)
has_second_exec = RNG.random(N_ORDERS) < 0.1

n_co = np.clip(RNG.poisson(1.2, size=N_ORDERS), 0, 6)
n_participants = np.clip(RNG.poisson(0.8, size=N_ORDERS), 0, 4)

deadline_null_mask = RNG.random(N_ORDERS) < 0.03

SLACK_CHOICES = np.array([3, 5, 7, 10, 14, 15, 20, 30, 45, 60, 90, 120])
SLACK_WEIGHTS = np.array([6, 8, 10, 10, 14, 10, 8, 12, 8, 6, 5, 3], dtype=float)
SLACK_WEIGHTS /= SLACK_WEIGHTS.sum()
slack_days_arr = RNG.choice(SLACK_CHOICES, size=N_ORDERS, p=SLACK_WEIGHTS).astype(float)
slack_days_arr = np.clip(slack_days_arr + RNG.integers(-2, 3, size=N_ORDERS), 1, 120)

deadline_dt_arr = [
    None if deadline_null_mask[i] else order_created_at[i] + timedelta(days=float(slack_days_arr[i]))
    for i in range(N_ORDERS)
]

deleted_mask = RNG.random(N_ORDERS) < 0.02

desc_roll = RNG.random(N_ORDERS)  # <0.07 null, <0.22 short, else long

dow = np.array([d.weekday() for d in order_created_at])
is_friday = (dow == 4).astype(float)
is_eom_arr = np.array([1.0 if is_eom(d) else 0.0 for d in order_created_at])

# число возвратов "На доработку" (латентная carelessness исполнителя), кап позже по времени
returns_lambda = np.exp(RETURNS_A + RETURNS_B * carelessness[primary_user_idx])
n_returns_sampled = np.clip(RNG.poisson(returns_lambda), 0, 3)

# ── роли/участники по объекту ──
responsible_lists = []
co_performer_lists = []
participant_lists = []
author_list = []
created_by_list = []

for i in range(N_ORDERS):
    excluded = {int(primary_user_idx[i])}
    resp = [int(primary_user_idx[i])]
    if has_second_exec[i]:
        cand = int(RNG.integers(0, N_USERS))
        tries = 0
        while cand in excluded and tries < 10:
            cand = int(RNG.integers(0, N_USERS))
            tries += 1
        resp.append(cand)
        excluded.add(cand)
    responsible_lists.append(resp)

    k = int(n_co[i])
    co = []
    if k > 0:
        pool = [u for u in range(N_USERS) if u not in excluded]
        chosen = RNG.choice(pool, size=min(k, len(pool)), replace=False)
        co = [int(x) for x in chosen]
        excluded.update(co)
    co_performer_lists.append(co)

    p = int(n_participants[i])
    part = []
    if p > 0:
        pool = [u for u in range(N_USERS) if u not in excluded]
        chosen = RNG.choice(pool, size=min(p, len(pool)), replace=False)
        part = [int(x) for x in chosen]
    participant_lists.append(part)

    author_idx = int(RNG.integers(0, N_USERS))
    author_list.append(author_idx)
    created_by_list.append(author_idx)

# ═══════════════════════════════ 5. Хронологический проход: workload → логит → overdue → терминальный исход ═══════════════════════════════
# workload_at_t0[i] = число ДРУГИХ поручений того же исполнителя, созданных РАНЕЕ и ещё
# не исполненных к t0 (t_exec_j > t0, или j вообще не исполнено). Это ЧЕСТНЫЙ, детерминированный
# по датам признак, но чтобы вычислить его по РЕАЛЬНОМУ t_exec (а не по deadline-заглушке),
# нужно решать исход каждого поручения В ТОМ ЖЕ хронологическом проходе, последовательно —
# иначе более раннее поручение ещё не имеет решённого исхода к моменту, когда он нужен более
# позднему поручению того же исполнителя.
order_index_sorted = sorted(range(N_ORDERS), key=lambda i: order_created_at[i])
# user_history[u] = список close_time более ранних поручений исполнителя u, где
# close_time — момент ВЫХОДА поручения из активной нагрузки: date_of_execution (если
# исполнено), время снятия с контроля (если удалено/снято), либо None (всё ещё открыто
# на дату среза → нагружает вечно, но таких по построению немного и они «свежие»).
user_history: dict[int, list] = {u: [] for u in range(N_USERS)}

workload_arr = np.zeros(N_ORDERS)
overdue_true = np.zeros(N_ORDERS, dtype=int)
final_status_arr = [None] * N_ORDERS
final_is_done_arr = [False] * N_ORDERS
final_date_exec_arr: list[datetime | None] = [None] * N_ORDERS
final_state_time_arr: list[datetime] = [None] * N_ORDERS  # terminal_start (начало терминальной версии)

for i in order_index_sorted:
    t0 = order_created_at[i]
    u_ = int(primary_user_idx[i])
    has_deadline = not deadline_null_mask[i]
    deadline_dt = deadline_dt_arr[i]

    # ── workload_at_t0: честный проход по УЖЕ РЕШЁННЫМ более ранним поручениям исполнителя ──
    # поручение «висит» на исполнителе в момент t0, если оно ещё не вышло из активной
    # нагрузки к t0: close_time is None (открыто на срез) ИЛИ close_time > t0.
    lst = user_history[u_]
    cnt = 0
    for close_f in lst:
        if close_f is None or close_f > t0:
            cnt += 1
    workload_arr[i] = cnt
    workload_capped = min(cnt, 8)

    # ── логит и сэмплирование overdue (латентная целевая переменная) ──
    slack_term = 0.0 if not has_deadline else COEF_SLACK * (SLACK_REF - slack_days_arr[i]) / SLACK_SCALE
    tendency_term = COEF_TEND * tendency[u_]
    careless_term = COEF_CARELESS * carelessness[u_]
    co_term = COEF_CO * n_co[i]
    type_term = TYPE_EFFECTS[type_idx[i]]
    desc_term = COEF_DESC * (1.0 if desc_roll[i] < 0.22 else 0.0)
    work_term = COEF_WORK * workload_capped
    season_term = COEF_FRIDAY * is_friday[i] + COEF_EOM * is_eom_arr[i]
    logit_signal = (
        slack_term + tendency_term + careless_term + co_term + type_term + desc_term + work_term + season_term
    )
    noise_i = float(RNG.normal(0, NOISE_SIGMA))
    p_i = float(sigmoid(INTERCEPT + logit_signal + noise_i))
    overdue_i = 1 if RNG.random() < p_i else 0
    if not has_deadline:
        overdue_i = 0  # без дедлайна просрочки не бывает
    overdue_true[i] = overdue_i

    # ── терминальная категория (как в v2, но переиспользуется здесь пошагово) ──
    is_deleted = bool(deleted_mask[i])
    if is_deleted:
        final_status = STATUS_CANCELLED
        final_is_done = False
        final_date_of_exec = None
    elif not has_deadline:
        r = RNG.random()
        if r < 0.55:
            final_status = STATUS_ON_TIME_DONE
            final_is_done = True
            final_date_of_exec = min(t0 + timedelta(days=int(RNG.integers(1, 60))), AS_OF)
        else:
            final_status = STATUS_IN_PROGRESS
            final_is_done = False
            final_date_of_exec = None
    elif deadline_dt < AS_OF:
        # срок уже прошёл к дате среза -> исход РЕШЁН
        if overdue_i == 1:
            # v3.1: время до фактического позднего исполнения ~ Exponential(mean=
            # LATE_EXEC_MEAN_DAYS) дней после дедлайна. Уложилось до среза -> "Исполнено
            # с нарушением срока" (выходит из активной нагрузки); не уложилось -> к дате
            # среза всё ещё "Просрочено" (открыто). Раньше это был фиксированный сплит
            # 65/35 без учёта времени -> просроченные копились вечной нагрузкой -> дрейф.
            late_days = float(RNG.exponential(LATE_EXEC_MEAN_DAYS)) + 0.5
            exec_candidate = deadline_dt + timedelta(days=late_days)
            if exec_candidate <= AS_OF - timedelta(hours=1):
                final_status = STATUS_LATE_DONE
                final_is_done = True
                final_date_of_exec = exec_candidate
            else:
                final_status = STATUS_OVERDUE_OPEN
                final_is_done = False
                final_date_of_exec = None
        else:
            final_status = STATUS_ON_TIME_DONE
            final_is_done = True
            span = max(1, int(slack_days_arr[i]) - 1)
            final_date_of_exec = t0 + timedelta(
                days=int(RNG.integers(0, span + 1)), hours=int(RNG.integers(0, 24))
            )
            if final_date_of_exec > deadline_dt:
                final_date_of_exec = deadline_dt
    else:
        # deadline >= AS_OF: срок ещё не наступил -> либо уже досрочно исполнено, либо "На исполнении"
        r = RNG.random()
        if r < 0.4:
            final_status = STATUS_ON_TIME_DONE
            final_is_done = True
            final_date_of_exec = t0 + timedelta(
                days=int(RNG.integers(0, max(1, (AS_OF - t0).days))), hours=int(RNG.integers(0, 24))
            )
            if final_date_of_exec > AS_OF:
                final_date_of_exec = AS_OF
        else:
            final_status = STATUS_IN_PROGRESS
            final_is_done = False
            final_date_of_exec = None

    if final_date_of_exec is not None and final_date_of_exec <= t0:
        final_date_of_exec = t0 + timedelta(hours=1)

    # ── момент наступления финального состояния (начало терминальной версии) ──
    if is_deleted:
        max_off = max(5, min(250, int((AS_OF - t0).days) - 1 or 5))
        final_state_time = t0 + timedelta(days=int(RNG.integers(5, max(6, max_off + 1))), hours=int(RNG.integers(0, 24)))
        if final_state_time >= AS_OF:
            final_state_time = AS_OF - timedelta(hours=1)
    elif final_is_done:
        final_state_time = final_date_of_exec
    elif final_status == STATUS_OVERDUE_OPEN:
        final_state_time = deadline_dt + timedelta(days=int(RNG.integers(1, 15)))
        cap = AS_OF - timedelta(hours=1)
        if final_state_time > cap:
            final_state_time = max(cap, deadline_dt + timedelta(minutes=30))
    else:  # На исполнении, ещё не готово
        upper_bound = deadline_dt if has_deadline else AS_OF
        upper_bound = min(upper_bound, AS_OF)
        span_days_avail = max(1, (upper_bound - t0).days)
        final_state_time = t0 + timedelta(days=int(RNG.integers(0, span_days_avail)), hours=int(RNG.integers(0, 24)))
    if final_state_time <= t0:
        final_state_time = t0 + timedelta(hours=1)
    if final_state_time > AS_OF and not is_deleted:
        final_state_time = AS_OF - timedelta(minutes=30)

    final_status_arr[i] = final_status
    final_is_done_arr[i] = final_is_done
    final_date_exec_arr[i] = final_date_of_exec
    final_state_time_arr[i] = final_state_time

    # close_time: когда поручение вышло из активной нагрузки исполнителя
    if final_is_done:
        close_time = final_date_of_exec              # исполнено (в срок / с нарушением)
    elif is_deleted:
        close_time = final_state_time                # снято с контроля (terminal_start)
    else:
        close_time = None                            # всё ещё открыто на дату среза
    user_history[u_].append(close_time)

workload_capped_arr = np.minimum(workload_arr, 8)
print(f"Хронологический проход (workload/логит/исход) завершён для {N_ORDERS} поручений.")

# ═══════════════════════════════ 6. Построение SCD2-цепочек версий ═══════════════════════════════
rows_v = []  # для COPY control_orders_v
progress_rows = []  # для COPY control_order_progress_v

for i in range(N_ORDERS):
    t0 = order_created_at[i]
    order_bk = order_bks[i]
    type_bk = type_ids[type_idx[i]]
    theme = f"О {THEME_TOPICS[int(RNG.integers(0, len(THEME_TOPICS)))]}"

    dr = desc_roll[i]
    if dr < 0.07:
        description = None
    elif dr < 0.22:
        description = DESC_SHORT[int(RNG.integers(0, len(DESC_SHORT)))]
    else:
        n_sent = int(RNG.integers(2, 5))
        idxs = RNG.choice(len(DESC_SENTENCES), size=n_sent, replace=False)
        description = " ".join(DESC_SENTENCES[j] for j in idxs)

    has_deadline = not deadline_null_mask[i]
    deadline_dt = deadline_dt_arr[i]

    resp_idx = responsible_lists[i]
    co_idx = co_performer_lists[i]
    part_idx = participant_lists[i]
    author_idx = author_list[i]

    resp_bks = [user_ids[j] for j in resp_idx]
    co_bks = [user_ids[j] for j in co_idx] if co_idx else (None if RNG.random() < 0.5 else [])
    part_bks = [user_ids[j] for j in part_idx] if part_idx else (None if RNG.random() < 0.5 else [])
    author_bks = [user_ids[author_idx]]
    created_by_bk = user_ids[created_by_list[i]]

    is_deleted = bool(deleted_mask[i])
    final_status = final_status_arr[i]
    final_is_done = final_is_done_arr[i]
    final_date_of_exec = final_date_exec_arr[i]
    terminal_start = final_state_time_arr[i]

    # days_expired: посчитано от ФИКСИРОВАННОЙ даты среза AS_OF, а НЕ от момента
    # версии — намеренная рассинхронизация с историей, воспроизводимо (leakage-поле).
    if not has_deadline:
        days_expired_v = None
    elif AS_OF > deadline_dt:
        days_expired_v = str((AS_OF - deadline_dt).days)
    else:
        days_expired_v = random.choice(["0", None])

    # ── ГЛАВНЫЙ ИНВАРИАНТ v3: исполненные и удалённые поручения ВСЕГДА получают
    # отдельную раннюю версию "в работе" (v1) до терминальной; открытые поручения
    # (ещё не исполненные к AS_OF) могут остаться одноверсионными — это разрешено
    # явно, т.к. такая версия не несёт исхода (is_done=false, date_of_execution=NULL).
    mandatory_v1 = final_is_done or is_deleted

    span_seconds = (terminal_start - t0).total_seconds()
    n_returns_max = max_returns_for_span(span_seconds)
    n_returns_final = min(int(n_returns_sampled[i]), n_returns_max)

    has_prechain = mandatory_v1 or n_returns_final > 0
    if has_prechain:
        n_pre_segments = max(1, 2 * n_returns_final + 1)
        pre_starts = build_prechain_starts(t0, terminal_start, n_pre_segments)
        version_starts = pre_starts + [terminal_start]
    else:
        # открытое поручение без возвратов -> допустим одноверсионный [t0, ∞)
        version_starts = [t0]

    order_versions = []
    n_starts = len(version_starts)
    for vi, vstart in enumerate(version_starts):
        is_last = vi == n_starts - 1
        vend = version_starts[vi + 1] if not is_last else None
        if is_last:
            status_v = final_status
            is_done_v = final_is_done
            date_exec_v = final_date_of_exec
            if is_deleted:
                deleted_v = True
                closure_reason_v = "deleted"
                vend = terminal_start + timedelta(hours=int(RNG.integers(1, 48)))
            else:
                deleted_v = False
                closure_reason_v = None  # текущая версия
        else:
            # пре-терминальные версии: чередование "На исполнении" / "На доработку"
            # (первая всегда "На исполнении"; далее пары возврат/повтор)
            pos_in_returns = vi  # 0 = v1, 1 = 1-й возврат, 2 = после 1-го возврата, ...
            if pos_in_returns == 0:
                status_v = STATUS_IN_PROGRESS
            else:
                status_v = STATUS_REWORK if pos_in_returns % 2 == 1 else STATUS_IN_PROGRESS
            is_done_v = False
            date_exec_v = None
            deleted_v = False
            closure_reason_v = "changed"

        updated_at_v = vend if (not is_last or is_deleted) else vstart

        order_versions.append(
            dict(
                order_bk=order_bk,
                control_order_type_bk=type_bk,
                control_order_theme=theme,
                description=description,
                deadline=deadline_dt,
                order_created_at=t0,
                status=status_v,
                is_done=is_done_v,
                date_of_execution=date_exec_v,
                days_expired=days_expired_v,
                responsible_executor_bks=resp_bks,
                co_performers_bks=co_bks,
                participants_bks=part_bks,
                task_author_bks=author_bks,
                created_at=t0,
                updated_at=updated_at_v,
                created_by_bk=created_by_bk,
                deleted=deleted_v,
                valid_period=fmt_range(vstart, vend),
                closure_reason=closure_reason_v,
                _batch_id=BATCH_ID,
                _start=vstart,  # служебное, для последующего join по version_id
            )
        )
    rows_v.extend(order_versions)

    # ── progress (ход исполнения): привязан к возвратам + немного случайных заметок ──
    prog_authors = resp_idx + co_idx + [author_idx]
    ordn = 0
    # заметка на каждый возврат "На доработку" (в момент его начала)
    for vi in range(1, n_starts - 1):  # исключая v1 (vi=0) и терминальную версию
        pos_in_returns = vi
        if pos_in_returns % 2 == 1:  # это старт версии "На доработку"
            ordn += 1
            progress_rows.append(
                (
                    order_bk,
                    ordn,
                    REWORK_SENTENCES[int(RNG.integers(0, len(REWORK_SENTENCES)))],
                    user_ids[int(prog_authors[int(RNG.integers(0, len(prog_authors)))])],
                    version_starts[vi],
                )
            )
    if RNG.random() < 0.6:
        n_prog = int(RNG.integers(0, 4))
        span_end = terminal_start
        for _ in range(n_prog):
            frac = RNG.uniform(0.05, 0.95)
            pdt = t0 + (span_end - t0) * frac
            ordn += 1
            progress_rows.append(
                (
                    order_bk,
                    ordn,
                    PROGRESS_SENTENCES[int(RNG.integers(0, len(PROGRESS_SENTENCES)))],
                    user_ids[int(prog_authors[int(RNG.integers(0, len(prog_authors)))])],
                    pdt,
                )
            )

print(f"Сформировано версий-строк: {len(rows_v)} (объектов: {N_ORDERS})")

# ═══════════════════════════════ 7. COPY control_orders_v ═══════════════════════════════
COLS_V = [
    "order_bk", "control_order_type_bk", "control_order_theme", "description",
    "deadline", "order_created_at", "status", "is_done", "date_of_execution",
    "days_expired", "responsible_executor_bks", "co_performers_bks",
    "participants_bks", "task_author_bks", "created_at", "updated_at",
    "created_by_bk", "deleted", "valid_period", "closure_reason", "_batch_id",
]
with cur.copy(f"COPY control_orders_v ({', '.join(COLS_V)}) FROM STDIN") as cp:
    for r in rows_v:
        cp.write_row(tuple(r[c] for c in COLS_V))
conn.commit()
print(f"control_orders_v: вставлено {len(rows_v)} строк-версий")

# ═══════════════════════════════ 8. Сопоставление version_id ═══════════════════════════════
cur.execute("SELECT version_id, order_bk, lower(valid_period) FROM control_orders_v")
vid_map = {}
for version_id, order_bk, start in cur.fetchall():
    vid_map[(order_bk, start)] = version_id
print(f"version_id сопоставлено: {len(vid_map)}")

# ═══════════════════════════════ 9. control_orders_user_roles_v ═══════════════════════════════
role_rows = []
for r in rows_v:
    vid = vid_map[(r["order_bk"], r["_start"])]
    for pos, ub in enumerate(r["responsible_executor_bks"] or []):
        role_rows.append((vid, "responsible_executor", ub, pos))
    for pos, ub in enumerate(r["co_performers_bks"] or []):
        role_rows.append((vid, "co_performer", ub, pos))
    for pos, ub in enumerate(r["participants_bks"] or []):
        role_rows.append((vid, "participant", ub, pos))
    for pos, ub in enumerate(r["task_author_bks"] or []):
        role_rows.append((vid, "author", ub, pos))

with cur.copy(
    "COPY control_orders_user_roles_v (parent_version_id, role, user_bk, _pos) FROM STDIN"
) as cp:
    for row in role_rows:
        cp.write_row(row)
conn.commit()
print(f"control_orders_user_roles_v: вставлено {len(role_rows)} строк")

# ═══════════════════════════════ 10. control_order_progress_v ═══════════════════════════════
with cur.copy(
    "COPY control_order_progress_v (order_bk, ord, text, author_bk, created_at) FROM STDIN"
) as cp:
    for row in progress_rows:
        cp.write_row(row)
conn.commit()
print(f"control_order_progress_v: вставлено {len(progress_rows)} строк")

# ═══════════════════════════════ 11. Сводка ═══════════════════════════════
cur.execute("SELECT count(*) FROM users")
n_users_db = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM control_order_types")
n_types_db = cur.fetchone()[0]
cur.execute("SELECT count(DISTINCT order_bk) FROM control_orders_v")
n_objects_db = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM control_orders_v")
n_versions_db = cur.fetchone()[0]
cur.execute(
    "SELECT status, count(*) FROM control_orders_v WHERE is_current AND NOT deleted GROUP BY status ORDER BY 2 DESC"
)
status_dist = cur.fetchall()
cur.execute(
    "SELECT count(*) FILTER (WHERE status IN (%s,%s))::float / count(*) "
    "FROM control_orders_v WHERE is_current AND NOT deleted",
    (STATUS_OVERDUE_OPEN, STATUS_LATE_DONE),
)
overdue_share = cur.fetchone()[0]
cur.execute(
    "SELECT count(*) FILTER (WHERE deadline IS NULL)::float / count(*) FROM control_orders_v"
)
deadline_null_share = cur.fetchone()[0]
cur.execute(
    "SELECT count(*) FILTER (WHERE description IS NULL)::float / count(*) FROM control_orders_v"
)
desc_null_share = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM control_orders_v WHERE deleted")
n_deleted_versions = cur.fetchone()[0]
cur.execute("SELECT count(DISTINCT order_bk) FROM control_orders_v WHERE deleted")
n_deleted_objects = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM control_orders_user_roles_v")
n_roles_db = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM control_order_progress_v")
n_progress_db = cur.fetchone()[0]
cur.execute("SELECT count(DISTINCT order_bk) FROM control_orders_v WHERE is_current")
n_current_objects = cur.fetchone()[0]

cur.execute(
    """
    SELECT status,
           (
             (date_of_execution IS NOT NULL AND deadline IS NOT NULL AND date_of_execution > deadline)
             OR (NOT is_done AND deadline IS NOT NULL AND deadline < %s)
           ) AS y_from_dates,
           count(*)
    FROM control_orders_v
    WHERE is_current AND NOT deleted
    GROUP BY 1, 2
    ORDER BY 1, 2
    """,
    (AS_OF,),
)
status_y_crosstab = cur.fetchall()
cur.execute(
    """
    SELECT count(*) FILTER (
        WHERE (date_of_execution IS NOT NULL AND deadline IS NOT NULL AND date_of_execution > deadline)
           OR (NOT is_done AND deadline IS NOT NULL AND deadline < %s)
    )::float / count(*)
    FROM control_orders_v WHERE is_current AND NOT deleted
    """,
    (AS_OF,),
)
overdue_share_from_dates = cur.fetchone()[0]

cur.execute(
    "SELECT count(DISTINCT order_bk) FROM control_orders_v GROUP BY order_bk HAVING count(*) > 1"
)
cur.execute("SELECT order_bk, count(*) c FROM control_orders_v GROUP BY order_bk")
version_counts = [c for _, c in cur.fetchall()]
version_count_dist = pd.Series(version_counts).value_counts().sort_index()

print("\n════════════════════════ СВОДКА ════════════════════════")
print(f"users:                       {n_users_db}")
print(f"control_order_types:         {n_types_db}")
print(f"control_orders_v объектов:   {n_objects_db}")
print(f"control_orders_v строк-версий: {n_versions_db}")
print(f"объектов с текущей версией (не deleted): {n_current_objects}")
print(f"объектов помечены deleted:   {n_deleted_objects} ({n_deleted_objects/N_ORDERS:.1%})")
print(f"строк с deleted=true:        {n_deleted_versions}")
print(f"доля overdue среди ТЕКУЩИХ версий неудалённых объектов (по status): {overdue_share:.1%}")
print(f"доля overdue среди ТЕКУЩИХ версий неудалённых объектов (по датам, AS_OF={AS_OF.date()}): {overdue_share_from_dates:.1%}")
print("распределение status (текущие, не deleted):")
for s, c in status_dist:
    print(f"    {s:35s} {c:6d}")
print("\nstatus × y_from_dates (должно быть 0/1 БЕЗ размазывания на статус):")
print(f"    {'status':35s} {'y':>5s} {'count':>8s}")
for s, y, c in status_y_crosstab:
    print(f"    {s:35s} {str(y):>5s} {c:8d}")
print(f"доля NULL deadline:          {deadline_null_share:.1%}")
print(f"доля NULL description:       {desc_null_share:.1%}")
print("распределение числа версий на поручение:")
for n_v, cnt in version_count_dist.items():
    print(f"    {n_v} версий: {cnt} поручений ({cnt/N_ORDERS:.1%})")
print(f"control_orders_user_roles_v: {n_roles_db} строк")
print(f"control_order_progress_v:    {n_progress_db} строк")
print("══════════════════════════════════════════════════════════")

# ═══════════════════════════════ 12. ОБЯЗАТЕЛЬНЫЕ САМОПРОВЕРКИ ═══════════════════════════════
print("\n████████████████████████ САМОПРОВЕРКИ v3 ████████████████████████")

# ── проверка 1: главный анти-утечка инвариант ──
# версия, покрывающая (order_created_at + 1 секунда), должна иметь is_done=false
# и date_of_execution IS NULL — для КАЖДОГО поручения, без исключений.
cur.execute(
    """
    SELECT count(*)
    FROM control_orders_v v
    WHERE v.valid_period @> (v.order_created_at + interval '1 second')
      AND (v.is_done IS TRUE OR v.date_of_execution IS NOT NULL)
    """
)
n_violations_invariant = cur.fetchone()[0]
cur.execute("SELECT count(DISTINCT order_bk) FROM control_orders_v")
n_orders_checked = cur.fetchone()[0]
print(f"[1] Анти-утечка инвариант (версия на t0+1с): нарушений = {n_violations_invariant} "
      f"из {n_orders_checked} поручений — {'ОК, 0 нарушений' if n_violations_invariant == 0 else 'FAIL!'}")

# ── проверка 2: as-of через valid_period @> T совпадает с расчётом из дат ──
# T берём как AS_OF (текущий срез) — сравниваем состояние из is_current версии
# с прямым расчётом "created<=T и (date_of_execution NULL или >T)" для ВСЕХ поручений
# (включая deleted — у них тоже должно быть is_done=false на любой T).
cur.execute(
    """
    WITH asof AS (
        SELECT v.order_bk,
               v.is_done AS asof_is_done,
               v.date_of_execution AS asof_date_exec
        FROM control_orders_v v
        WHERE v.valid_period @> %(t)s::timestamptz
    ),
    dates_calc AS (
        SELECT DISTINCT ON (order_bk) order_bk, order_created_at, deleted
        FROM control_orders_v
        ORDER BY order_bk, valid_period
    )
    SELECT count(*)
    FROM asof a
    JOIN dates_calc d ON d.order_bk = a.order_bk
    WHERE d.order_created_at <= %(t)s::timestamptz
      AND a.asof_is_done IS DISTINCT FROM (a.asof_date_exec IS NOT NULL AND a.asof_date_exec <= %(t)s::timestamptz)
    """,
    {"t": AS_OF},
)
n_mismatches_asof = cur.fetchone()[0]
cur.execute(
    "SELECT count(*) FROM control_orders_v v WHERE v.valid_period @> %(t)s::timestamptz",
    {"t": AS_OF},
)
n_asof_rows = cur.fetchone()[0]
print(f"[2] as-of(valid_period @> AS_OF) согласован с is_done: расхождений = {n_mismatches_asof} "
      f"из {n_asof_rows} версий, покрывающих AS_OF — {'ОК, 0 расхождений' if n_mismatches_asof == 0 else 'FAIL!'}")

# ── проверка 3: нетекущие версии — все is_done=false и date_of_execution IS NULL ──
cur.execute(
    "SELECT count(*) FROM control_orders_v WHERE NOT is_current AND (is_done IS TRUE OR date_of_execution IS NOT NULL)"
)
n_bad_noncurrent = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM control_orders_v WHERE NOT is_current")
n_noncurrent = cur.fetchone()[0]
print(f"[3] Нетекущие версии без исхода: нарушений = {n_bad_noncurrent} из {n_noncurrent} нетекущих версий — "
      f"{'ОК' if n_bad_noncurrent == 0 else 'FAIL!'}")

# ── проверка 4: честный AUC (структурные + история исполнителя ± workload_at_T) ──
cur.execute(
    """
    SELECT v.order_bk, v.order_created_at, v.deadline, v.control_order_type_bk,
           v.description, v.control_order_theme,
           coalesce(array_length(v.co_performers_bks, 1), 0) AS n_co,
           coalesce(array_length(v.participants_bks, 1), 0) AS n_part,
           v.responsible_executor_bks[1] AS executor_bk,
           v.is_done, v.date_of_execution, v.status
    FROM control_orders_v v
    WHERE v.is_current AND NOT v.deleted
    """
)
cols = [d.name for d in cur.description]
df = pd.DataFrame(cur.fetchall(), columns=cols)

# workload_at_T пересчитываем ЧЕСТНО из НАБЛЮДАЕМЫХ дат для ВСЕХ поручений. close_time —
# момент выхода из активной нагрузки: date_of_execution (исполнено), время снятия с
# контроля (deleted → начало deleted-версии), либо None (открыто на срез). Это ТО ЖЕ
# правило, что в генеративном проходе, и оно ВОСПРОИЗВОДИМО в ноутбуке из наблюдаемых
# полей (без латентных величин).
cur.execute(
    """
    SELECT DISTINCT ON (order_bk) order_bk, order_created_at, responsible_executor_bks[1] AS executor_bk
    FROM control_orders_v
    ORDER BY order_bk, valid_period
    """
)
all_orders_cols = [d.name for d in cur.description]
df_all = pd.DataFrame(cur.fetchall(), columns=all_orders_cols)
df_outcome = df[["order_bk", "is_done", "date_of_execution"]].copy()
df_all = df_all.merge(df_outcome, on="order_bk", how="left")
df_all["is_done"] = df_all["is_done"].fillna(False)  # не в текущем срезе (снято) -> не исполнено
# время снятия с контроля для удалённых поручений = начало их deleted-версии
cur.execute("SELECT order_bk, lower(valid_period) FROM control_orders_v WHERE deleted")
removal_time = dict(cur.fetchall())
# close_time по каждому поручению
df_all["close_time"] = [
    dexec if bool(done) else removal_time.get(obk)  # done -> dexec; snyato -> removal; иначе None
    for obk, done, dexec in zip(df_all["order_bk"], df_all["is_done"], df_all["date_of_execution"])
]

df_all_sorted = df_all.sort_values("order_created_at").reset_index(drop=True)
workload_by_bk = {}
hist_by_executor: dict = {}
for row in df_all_sorted.itertuples():
    u = row.executor_bk
    T = row.order_created_at
    lst = hist_by_executor.setdefault(u, [])
    cnt = sum(1 for close_f in lst if close_f is None or close_f > T)
    workload_by_bk[row.order_bk] = cnt
    lst.append(row.close_time)

df["workload_at_t"] = df["order_bk"].map(workload_by_bk).clip(upper=8)

# целевая метка y — канонический date-based расчёт относительно AS_OF
df["y"] = (
    ((df["date_of_execution"].notna()) & (df["deadline"].notna()) & (df["date_of_execution"] > df["deadline"]))
    | ((~df["is_done"]) & (df["deadline"].notna()) & (df["deadline"] < AS_OF))
).astype(int)

df["has_deadline"] = df["deadline"].notna().astype(float)
df["slack_days"] = (df["deadline"] - df["order_created_at"]).dt.total_seconds() / 86400.0
df["slack_days"] = df["slack_days"].fillna(0.0)
df["desc_len"] = df["description"].fillna("").str.len()
df["desc_flag"] = (df["desc_len"] < 30).astype(float)
df["theme_len"] = df["control_order_theme"].fillna("").str.len()
df["dow"] = df["order_created_at"].dt.weekday
df["is_friday"] = (df["dow"] == 4).astype(float)
df["is_eom"] = df["order_created_at"].apply(lambda d: 1.0 if is_eom(d) else 0.0)
for t_i, t_bk in enumerate(type_ids):
    df[f"type_{t_i}"] = (df["control_order_type_bk"] == t_bk).astype(float)

# time-split 70/30 по order_created_at
df = df.sort_values("order_created_at").reset_index(drop=True)
n_train = int(len(df) * 0.7)
train_mask = np.zeros(len(df), dtype=bool)
train_mask[:n_train] = True
df["train"] = train_mask
split_ts = df.loc[n_train - 1, "order_created_at"]

# target-encoding истории исполнителя: train -> expanding prior-only среднее; test -> статичное train-среднее
global_train_mean = df.loc[df["train"], "y"].mean()
exec_hist_enc = np.zeros(len(df))
sums: dict = {}
counts: dict = {}
train_sums: dict = {}
train_counts: dict = {}
for idx, row in df.iterrows():
    u = row["executor_bk"]
    if row["train"]:
        if train_counts.get(u, 0) > 0:
            exec_hist_enc[idx] = train_sums[u] / train_counts[u]
        else:
            exec_hist_enc[idx] = global_train_mean
        train_sums[u] = train_sums.get(u, 0.0) + row["y"]
        train_counts[u] = train_counts.get(u, 0) + 1
    else:
        if train_counts.get(u, 0) > 0:
            exec_hist_enc[idx] = train_sums[u] / train_counts[u]
        else:
            exec_hist_enc[idx] = global_train_mean
df["exec_hist_enc"] = exec_hist_enc

feat_struct = ["slack_days", "has_deadline", "desc_flag", "desc_len", "theme_len", "n_co", "n_part",
               "is_friday", "is_eom", "exec_hist_enc"] + [f"type_{t}" for t in range(len(type_ids))]
feat_with_work = feat_struct + ["workload_at_t"]


def fit_auc(feats, frame):
    Xtr = frame.loc[frame.train, feats].values
    Xte = frame.loc[~frame.train, feats].values
    ytr = frame.loc[frame.train, "y"].values
    yte = frame.loc[~frame.train, "y"].values
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict_proba(scaler.transform(Xte))[:, 1]
    return roc_auc_score(yte, pred)


auc_no_work = fit_auc(feat_struct, df)
auc_with_work = fit_auc(feat_with_work, df)
delta_work = auc_with_work - auc_no_work

# leakage-признаки: status/is_done/date_of_execution должны давать AUC ~= 1.0
df["is_done_f"] = df["is_done"].astype(float)
df["has_date_exec"] = df["date_of_execution"].notna().astype(float)
status_dummies = pd.get_dummies(df["status"], prefix="status").astype(float)
df_leak = pd.concat([df[["train", "y", "is_done_f", "has_date_exec"]], status_dummies], axis=1)
leak_feats = ["is_done_f", "has_date_exec"] + list(status_dummies.columns)
auc_leakage = fit_auc(leak_feats, df_leak)

print(f"[4] Честный AUC (структурные + история исполнителя, БЕЗ workload):  {auc_no_work:.4f}")
print(f"    Честный AUC (структурные + история исполнителя + workload_at_T): {auc_with_work:.4f}")
print(f"    Прирост от workload_at_T: {delta_work:+.4f} (порог >= +0.03: {'ОК' if delta_work >= 0.03 else 'FAIL!'})")
print(f"    Честный AUC в целевом окне [0.82, 0.88]: {'ОК' if 0.82 <= auc_with_work <= 0.88 else 'FAIL!'}")
print(f"    Leakage-признаки (status/is_done/date_of_execution) AUC: {auc_leakage:.4f} "
      f"({'ОК, ~1.0' if auc_leakage >= 0.995 else 'подозрительно низко!'})")
print(f"    (train/test split по order_created_at: train={train_mask.sum()}, test={(~train_mask).sum()}, "
      f"граница = {split_ts})")

# ── проверка 5: доля overdue / версии / статусы / NULL — уже выведены выше в сводке ──
print(f"[5] Доля overdue (по датам): {overdue_share_from_dates:.1%} "
      f"(целевое окно 0.25-0.30: {'ОК' if 0.25 <= overdue_share_from_dates <= 0.30 else 'вне окна!'})")
print("    Распределение числа версий и статусов — см. СВОДКУ выше; доли NULL — см. выше.")
print("████████████████████████████████████████████████████████████████")

cur.close()
conn.close()

# ═══════════════════════════════ 13. _signal_meta.json ═══════════════════════════════
meta = {
    "seed": SEED,
    "as_of": AS_OF.isoformat(),
    "batch_id": BATCH_ID,
    "note": (
        "v3.1: фикс историчности (v3) + стационарность нагрузки (v3.1). Латентные параметры "
        "генеративной модели overdue. Явной колонки "
        "overdue в БД нет. КАНОНИЧЕСКАЯ деривация таргета — ИЗ ДАТ (не из status!), относительно "
        "фиксированной даты среза as_of: overdue = 1, если (date_of_execution IS NOT NULL AND "
        "deadline IS NOT NULL AND date_of_execution > deadline) OR (NOT is_done AND deadline IS NOT "
        "NULL AND deadline < as_of); иначе overdue = 0 (в т.ч. всегда 0, если deadline IS NULL). "
        "ГЛАВНЫЙ ИНВАРИАНТ v3: ни одна версия, покрывающая (order_created_at + 1 секунда), не несёт "
        "финальный исход — is_done=false и date_of_execution IS NULL для версии на этот момент для "
        "ЛЮБОГО поручения. Исполненные и удалённые поручения ВСЕГДА имеют >= 2 версии (ранняя "
        "'в работе' + терминальная); открытые (ещё не исполненные к as_of) поручения могут быть "
        "одноверсионными [t0, ∞) — это не утечка, т.к. версия не несёт исхода."
    ),
    "target_overdue_rate": [0.25, 0.30],
    "target_honest_auc_no_workload": 0.80,
    "target_honest_auc_with_workload": [0.82, 0.88],
    "target_workload_auc_gain_min": 0.03,
    "honest_auc_recipe": (
        "AUC проверяется на признаках УРОВНЯ СТУДЕНТА (латентные tendency/carelessness/type_effect "
        "из этого файла НЕ подаются модели напрямую!): slack_days=(deadline-order_created_at), "
        "has_deadline, len(description)/is_null, len(control_order_theme), n_co_performers/"
        "n_participants (NULL и {} трактовать как 0), control_order_type_bk one-hot, day-of-week/"
        "is_friday/is_end_of_month, ЦЕЛЕВОЕ КОДИРОВАНИЕ responsible_executor_bks[0] (train: "
        "expanding по времени, только более ранние TRAIN-заказы того же исполнителя, unseen -> "
        "train-среднее; test: статичное train-среднее) и, отдельно, workload_at_T (число других "
        "поручений того же исполнителя, открытых на момент T = order_created_at; открытым считается "
        "поручение, для которого T < t_exec ИЛИ оно вообще не исполнено, посчитано честным "
        "хронологическим проходом по датам). Time-split: train = первые 70% по order_created_at, "
        "test = последние 30%. LogisticRegression на стандартизованных признаках. "
        "Ожидаемый результат: без workload ~0.80, с workload ~0.82-0.88 (прирост >= +0.03). "
        "Leakage-признаки (status/is_done/date_of_execution) дают AUC ~1.0 — сохранено намеренно."
    ),
    "n_users": N_USERS,
    "tendency_by_user_bk": {str(user_ids[i]): float(tendency[i]) for i in range(N_USERS)},
    "carelessness_by_user_bk": {str(user_ids[i]): float(carelessness[i]) for i in range(N_USERS)},
    "type_effect_by_type_bk": {str(type_ids[i]): float(TYPE_EFFECTS[i]) for i in range(len(TYPE_NAMES))},
    "type_name_by_type_bk": {str(type_ids[i]): TYPE_NAMES[i] for i in range(len(TYPE_NAMES))},
    "logit_coefficients": {
        "intercept": INTERCEPT,
        "noise_sigma": NOISE_SIGMA,
        "scale": SCALE,
        "tendency_multiplier": TEND_MULT,
        "rho_carelessness_vs_tendency": RHO_CARELESS,
        "coef_slack": COEF_SLACK,
        "slack_ref_days": SLACK_REF,
        "slack_scale_days": SLACK_SCALE,
        "coef_tendency": COEF_TEND,
        "coef_carelessness": COEF_CARELESS,
        "coef_n_co_performers": COEF_CO,
        "coef_short_or_missing_description": COEF_DESC,
        "coef_workload_capped_at_8": COEF_WORK,
        "coef_is_friday": COEF_FRIDAY,
        "coef_is_end_of_month": COEF_EOM,
        "returns_poisson_a": RETURNS_A,
        "returns_poisson_b": RETURNS_B,
        "late_exec_mean_days": LATE_EXEC_MEAN_DAYS,
    },
    "logit_formula": (
        "logit = intercept "
        "+ coef_slack * (slack_ref_days - slack_days) / slack_scale_days   [0, если deadline IS NULL] "
        "+ coef_tendency * tendency[responsible_executor_bks[0]] "
        "+ coef_carelessness * carelessness[responsible_executor_bks[0]] "
        "+ coef_n_co_performers * len(co_performers_bks) "
        "+ type_effect[control_order_type_bk] "
        "+ coef_short_or_missing_description * (description IS NULL OR len(description) < 30) "
        "+ coef_workload_capped_at_8 * min(open_orders_of_executor_at_T_by_real_t_exec, 8)   "
        "[v3: РЕАЛЬНЫЙ (не минорный) компонент, входит в проверочный набор honest_auc_recipe] "
        "+ coef_is_friday * (order_created_at.weekday()==Friday) "
        "+ coef_is_end_of_month * (order_created_at в последних 3 днях месяца) "
        "+ Normal(0, noise_sigma); "
        "p = sigmoid(logit); overdue_sampled ~ Bernoulli(p); "
        "carelessness = rho_carelessness_vs_tendency * tendency + sqrt(1-rho^2) * N(0,1) [по пользователю]; "
        "n_returns ~ Poisson(exp(returns_poisson_a + returns_poisson_b * carelessness[executor])), cap 0..3 "
        "-> число пар версий 'На доработку'/'На исполнении' перед терминальной версией "
        "(число версий урезается, если не помещается в доступный интервал [t0, terminal_start) "
        "с шагом >= 1 час на сегмент). "
        "Далее исход (done/overdue-open/in-progress/deleted) выбирается как в v2, но ПОСЛЕДОВАТЕЛЬНО "
        "в хронологическом порядке по order_created_at, т.к. workload_at_T следующего поручения "
        "зависит от УЖЕ решённого исхода (в т.ч. t_exec) более ранних поручений того же исполнителя."
    ),
    "workload_definition": (
        "workload_at_T = число ДРУГИХ поручений того же responsible_executor_bks[0], созданных "
        "РАНЕЕ (order_created_at < T) и ещё 'висящих' на исполнителе в момент T, т.е. не вышедших "
        "из активной нагрузки к T. Момент выхода close_time: date_of_execution (если поручение "
        "исполнено — в срок или с нарушением), время СНЯТИЯ с контроля (если поручение удалено/снято "
        "— начало deleted-версии, lower(valid_period)), либо +∞ (если поручение всё ещё открыто на "
        "дату среза). Поручение висит в момент T, если close_time IS NULL (открыто на срез) ИЛИ "
        "close_time > T. v3.1: раньше 'снятые' и 'просроченные-открытые' считались открытыми ВЕЧНО "
        "(is_done=false), из-за чего нагрузка неограниченно копилась от старта системы из пустого "
        "состояния и коррелировала со временем → overdue_rate дрейфовал вверх по годам. Теперь "
        "снятые выходят из нагрузки в момент снятия, а просроченные — при позднем исполнении "
        "(см. LATE_EXEC_MEAN_DAYS), поэтому нагрузка СТАЦИОНАРНА. Правило close_time воспроизводимо "
        "в ноутбуке из наблюдаемых полей (date_of_execution, is_done, флаг/время снятия) без "
        "латентных величин. Это ЗАМЕТНЫЙ драйвер риска: входит в honest_auc_recipe и добавляет "
        ">= +0.03 AUC поверх структурных признаков + истории исполнителя."
    ),
    "late_execution_model": (
        "Просроченное (overdue=1) поручение с уже прошедшим к дате среза дедлайном исполняется "
        "поздно через late_days ~ Exponential(mean=LATE_EXEC_MEAN_DAYS) дней после дедлайна. Если "
        "deadline + late_days <= as_of - 1час → статус 'Исполнено с нарушением срока', "
        "date_of_execution = deadline + late_days (поручение выходит из активной нагрузки). Иначе к "
        "дате среза поручение всё ещё 'Просрочено' (открыто, date_of_execution IS NULL). Так давно "
        "просроченные поручения со временем закрываются (нагрузка стационарна), а открытыми к срезу "
        "остаются в основном поручения со свежими дедлайнами. Метка overdue=1 в обоих случаях."
    ),
    "anti_leakage_invariant": (
        "Для КАЖДОГО поручения версия, покрывающая (order_created_at + 1 секунда), имеет "
        "is_done=false И date_of_execution IS NULL. Исполненные/удалённые поручения имеют >= 2 "
        "версии (ранняя 'в работе' [t0, t_exec) + терминальная [t_exec, ...) с исходом); открытые "
        "(ещё не исполненные к as_of) поручения МОГУТ быть одноверсионными [t0, ∞) без исхода — "
        "это не нарушение инварианта."
    ),
    "generated_at": REAL_RUN_TS.isoformat(),
}

meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_signal_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"\n_signal_meta.json сохранён: {meta_path}")
