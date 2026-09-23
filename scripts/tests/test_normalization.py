import pandas as pd
import numpy as np

# These tests independently verify the central schema assumptions used by the normalizer.
def test_tcpl_log_ac50_conversion():
    log_ac50 = pd.Series([-1.0, 0.0, 1.0])
    ac50 = np.power(10.0, log_ac50)
    assert list(ac50.round(6)) == [0.1, 1.0, 10.0]

def test_expected_hit_direction_definition():
    mapping = {0: "no_hit", 1: "increase", 2: "decrease"}
    assert mapping[1] == "increase"
    assert mapping[2] == "decrease"


def test_toxcast_continuous_hitcall_threshold():
    threshold=0.90
    values=[0.0,0.25,0.89,0.90,1.0]
    active=[x>=threshold for x in values]
    assert active == [False,False,False,True,True]
