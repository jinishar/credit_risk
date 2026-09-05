-- DuckDB schema for the Home Credit application table.
-- Loaded fresh from data/application_train.csv (real or synthetic) on every
-- container start by src/data/loader.py. This file exists primarily as
-- documentation and as grounding context for the NL->SQL prompt templates
-- (src/talk_to_data/prompt_templates.py), which quote it verbatim.
--
-- Only a curated subset of the 121 application columns is listed here (the
-- ones EDA, the model, and the chatbot actually reason about) to keep the
-- prompt token-efficient; the full 121-column dictionary lives alongside
-- this file in HomeCredit_columns_description.csv and the loader keeps
-- every column from the source CSV regardless of what's listed below.

CREATE TABLE IF NOT EXISTS applications (
    SK_ID_CURR                  BIGINT PRIMARY KEY,   -- unique applicant/loan id
    TARGET                      TINYINT,              -- 1 = defaulted, 0 = repaid (NULL for application_test rows)
    NAME_CONTRACT_TYPE          VARCHAR,              -- 'Cash loans' or 'Revolving loans'
    CODE_GENDER                 VARCHAR,              -- 'M' / 'F'
    FLAG_OWN_CAR                VARCHAR,              -- 'Y' / 'N'
    FLAG_OWN_REALTY             VARCHAR,              -- 'Y' / 'N'
    CNT_CHILDREN                INTEGER,
    AMT_INCOME_TOTAL            DOUBLE,               -- annual income
    AMT_CREDIT                  DOUBLE,               -- requested credit amount
    AMT_ANNUITY                 DOUBLE,               -- loan annuity (monthly payment)
    AMT_GOODS_PRICE             DOUBLE,               -- price of the goods the loan is for
    NAME_INCOME_TYPE            VARCHAR,
    NAME_EDUCATION_TYPE         VARCHAR,
    NAME_FAMILY_STATUS          VARCHAR,
    NAME_HOUSING_TYPE           VARCHAR,
    DAYS_BIRTH                  INTEGER,              -- negative days before application; age = -DAYS_BIRTH/365.25
    DAYS_EMPLOYED                INTEGER,             -- negative days employed; 365243 = anomalous "not employed" placeholder
    OCCUPATION_TYPE              VARCHAR,
    CNT_FAM_MEMBERS              DOUBLE,
    REGION_RATING_CLIENT         INTEGER,              -- 1 (best) - 3 (worst)
    ORGANIZATION_TYPE            VARCHAR,
    EXT_SOURCE_1                 DOUBLE,               -- normalized external credit score (0-1)
    EXT_SOURCE_2                 DOUBLE,
    EXT_SOURCE_3                 DOUBLE
    -- ... plus ~100 further columns loaded as-is from the source CSV (building info,
    -- document flags, credit bureau enquiry counts, etc.) - see HomeCredit_columns_description.csv
);
