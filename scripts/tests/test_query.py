from core.query import load_tables,profile_from_tables,HIT_DIRECTION


def test_hit_direction_mapping():
    assert HIT_DIRECTION[1]=="increase"; assert HIT_DIRECTION[2]=="decrease"


def test_real_mea_query_permethrin_runs():
    profile=profile_from_tables("Permethrin",load_tables(),use_pubchem=False)
    assert profile["status"]=="ok"; assert profile["preferred_name"]=="Permethrin"
    assert any(x["direction"]=="increase" for x in profile.get("mea_active_metrics",[]))
    assert any(x["direction"]=="decrease" for x in profile.get("mea_active_metrics",[]))
