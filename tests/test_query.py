import networkx as nx
import pytest

from conftest import ROOT, load_script

query = load_script("07_query_chemical")


def test_hit_direction_labels():
    assert query.HIT_DIRECTION[1] == "increase" and query.HIT_DIRECTION[2] == "decrease"


def test_name_normalization_ignores_case_wells_and_punctuation():
    assert query.normalize_name("(+/-)-cis-Permethrin_Y2P19") == query.normalize_name("cis PERMETHRIN")


def test_graphml_edge_ids_are_unique(tmp_path):
    G = nx.MultiGraph()
    G.add_edge("a", "b", relation="x")
    G.add_edge("a", "c", relation="y")   # a networkx multigraph would give both edges key 0
    path = tmp_path / "g.graphml"
    query.write_graphml(G, path)
    ids = [line.split('id="')[1].split('"')[0] for line in path.read_text().splitlines() if "<edge " in line]
    assert len(ids) == len(set(ids)) == 2


@pytest.mark.skipif(not (ROOT / "data" / "mea_processed" / "mea_compound_summary.csv").exists(),
                    reason="run step 06 first")
def test_permethrin_profile_has_mea_phenotypes():
    profile = query.build_profile("Permethrin", query.load_tables(), use_pubchem=False)
    assert profile["status"] == "ok" and profile["preferred_name"] == "Permethrin"
    directions = {m["direction"] for m in profile.get("mea_active_metrics", [])}
    assert {"increase", "decrease"} <= directions
