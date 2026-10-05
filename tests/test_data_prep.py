import numpy as np
import pandas as pd

from conftest import load_script


def test_bioactivity_hitcalls_are_clipped_and_inactive_potency_is_masked():
    normalize = load_script("01_normalize_epa_data")
    raw = pd.DataFrame({"dtxsid": ["DTXSID1", "DTXSID2", "DTXSID3"], "aenm": ["a", "a", "a"],
                        "hitc": [1.0, 0.2, 5.0], "ac50": [10.0, 20.0, 30.0]})
    out = normalize.normalize_bioactivity(raw)
    assert set(out["DTXSID"]) == {"DTXSID1", "DTXSID2"}             # hit call 5.0 is outside [-1, 1] and dropped
    assert out.loc[out.DTXSID == "DTXSID1", "ac50_uM"].iloc[0] == 10.0
    assert np.isnan(out.loc[out.DTXSID == "DTXSID2", "ac50_uM"].iloc[0])   # inactive: potency masked


def test_alias_columns_are_recognised():
    normalize = load_script("01_normalize_epa_data")
    assert normalize.alias_column(["DSSTox_Substance_Id", "x"], "DTXSID") == "DSSTox_Substance_Id"
    assert normalize.alias_column(["HITC"], "hitcall") == "HITC"


def test_hazard_uses_only_a_single_study_design():
    cohort = load_script("02_build_training_cohort")
    common = {"DTXSID": "D1", "metric": "NOAEL", "units": "mg/kg-day", "route": "oral", "species": "rat"}
    same = pd.DataFrame([{**common, "value": 10.0, "study_type": "chronic", "duration": "90"},
                         {**common, "value": 10.0, "study_type": "chronic", "duration": "90"}])
    mixed = pd.DataFrame([{**common, "value": 10.0, "study_type": "chronic", "duration": "90"},
                          {**common, "value": 10.0, "study_type": "acute", "duration": "1"}])
    assert cohort.prepare_hazard(same)["hazard_count"].iloc[0] == 2
    assert cohort.prepare_hazard(mixed)["noael_log"].isna().all()


def test_exposure_is_summarised_per_chemical():
    cohort = load_script("02_build_training_cohort")
    exposure = pd.DataFrame({"DTXSID": ["D1", "D1", "D2"], "product_id": ["p1", "p2", "p1"],
                             "use_category": ["x", "x", "y"]})
    out = cohort.prepare_exposure(exposure).set_index("DTXSID")
    assert out.loc["D1", "product_count"] == 2 and out.loc["D1", "use_category_count"] == 1


def test_split_terms_handles_separators():
    mea = load_script("06_prepare_mea_data")
    assert mea.split_terms("Na channel; K channel | nan + GABA") == ["Na channel", "K channel", "GABA"]
    assert mea.split_terms(float("nan")) == []
