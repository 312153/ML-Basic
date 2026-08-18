-- Учебная БД контрольных поручений (подмножество для задачи «просрочка поручения»).
-- Схема повторяет семантику витрины-источника: SCD2-историчность, EXCLUDE на
-- непересекающиеся версии, справочники без FK, bridge ролей, вьюха «текущий срез».
-- Синтетические данные (см. generate_data.py) — реальные данные заказчика не используются.

CREATE EXTENSION IF NOT EXISTS btree_gist;   -- нужен для EXCLUDE с равенством по UUID

DROP VIEW  IF EXISTS control_orders CASCADE;
DROP TABLE IF EXISTS control_order_progress_v   CASCADE;
DROP TABLE IF EXISTS control_orders_user_roles_v CASCADE;
DROP TABLE IF EXISTS control_orders_v            CASCADE;
DROP TABLE IF EXISTS control_order_types         CASCADE;
DROP TABLE IF EXISTS users                       CASCADE;

-- ── Справочники (плоские, без истории, PK = elma_id) ───────────────────────────
CREATE TABLE users (
    elma_id UUID PRIMARY KEY,
    name    TEXT NOT NULL            -- ФИО (в синтетике — обезличенный псевдоним)
);

CREATE TABLE control_order_types (
    elma_id UUID PRIMARY KEY,
    name    TEXT NOT NULL
);

-- ── Факт с историей (SCD2) ─────────────────────────────────────────────────────
CREATE TABLE control_orders_v (
    version_id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_bk               UUID NOT NULL,               -- бизнес-ключ объекта
    -- данные карточки
    control_order_type_bk  UUID,                        -- → control_order_types (без FK)
    control_order_theme    TEXT,
    description            TEXT,
    deadline               TIMESTAMPTZ,                 -- контрольный срок
    order_created_at       TIMESTAMPTZ,                 -- дата постановки
    status                 TEXT,                        -- ⚠ исход-подобное (утечка)
    is_done                BOOLEAN,                     -- ⚠ утечка
    date_of_execution      TIMESTAMPTZ,                 -- ⚠ утечка
    days_expired           TEXT,                        -- ⚠ производное «от сегодня», утечка
    responsible_executor_bks UUID[],                    -- исполнители
    co_performers_bks      UUID[],                      -- соисполнители
    participants_bks       UUID[],                      -- участники
    task_author_bks        UUID[],                      -- постановщики
    -- системные
    created_at             TIMESTAMPTZ,
    updated_at             TIMESTAMPTZ,
    created_by_bk          UUID,                        -- → users
    deleted                BOOLEAN DEFAULT FALSE,
    -- SCD2-хвост
    valid_period           TSTZRANGE NOT NULL,
    is_current             BOOLEAN GENERATED ALWAYS AS (upper_inf(valid_period)) STORED,
    closure_reason         TEXT,                        -- 'changed' | 'deleted' | NULL
    _batch_id              TEXT,
    _loaded_at             TIMESTAMPTZ DEFAULT now(),
    -- версии одного бизнес-ключа не пересекаются во времени
    EXCLUDE USING gist (order_bk WITH =, valid_period WITH &&)
);
CREATE INDEX ix_co_v_bk       ON control_orders_v (order_bk);
CREATE INDEX ix_co_v_current  ON control_orders_v (order_bk) WHERE upper_inf(valid_period);
CREATE INDEX ix_co_v_period   ON control_orders_v USING gist (valid_period);

-- ── Bridge: участники по ролям (снимок на версию родителя) ─────────────────────
CREATE TABLE control_orders_user_roles_v (
    parent_version_id BIGINT NOT NULL REFERENCES control_orders_v(version_id) ON DELETE CASCADE,
    role              TEXT NOT NULL,        -- 'responsible_executor'|'co_performer'|'participant'|'author'
    user_bk           UUID NOT NULL,
    _pos              INT  NOT NULL
);
CREATE INDEX ix_co_roles_parent ON control_orders_user_roles_v (parent_version_id);

-- ── Leakage-ребёнок: ход исполнения (в признаки НЕ брать) ───────────────────────
CREATE TABLE control_order_progress_v (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_bk   UUID NOT NULL,
    ord        INT,
    text       TEXT,
    author_bk  UUID,
    created_at TIMESTAMPTZ
);
CREATE INDEX ix_co_progress_bk ON control_order_progress_v (order_bk);

-- ── Вьюха «текущий срез»: только актуальные версии, имена вместо UUID ──────────
CREATE VIEW control_orders AS
SELECT v.order_bk                       AS id,
       t.name                           AS type,          -- расшифровка type_bk
       v.control_order_theme,
       v.description,
       v.deadline,
       v.order_created_at,
       v.status,
       v.is_done,
       v.date_of_execution,
       v.days_expired,
       v.responsible_executor_bks,
       v.co_performers_bks,
       v.participants_bks,
       v.task_author_bks,
       u.name                           AS created_by,     -- расшифровка created_by_bk
       v.created_at,
       v.updated_at,
       v.deleted
FROM   control_orders_v v
LEFT   JOIN control_order_types t ON t.elma_id = v.control_order_type_bk
LEFT   JOIN users u               ON u.elma_id = v.created_by_bk
WHERE  upper_inf(v.valid_period);          -- NB: мягко удалённые (deleted) НЕ отсекаются
