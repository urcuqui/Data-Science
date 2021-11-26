from utils_pos import get_word_tag, preprocess
import pandas as pd
from collections import defaultdict
import math
import numpy as np
import pickle

def test_create_dictionaries(target, training_corpus, vocab):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_case",
            "input": {
                "training_corpus": training_corpus,
                "vocab": vocab,
                "verbose": False,
            },
            "expected": {
                "len_emission_counts": 31140,
                "len_transition_counts": 1421,
                "len_tag_counts": 46,
                "emission_counts": {
                    ("DT", "the"): 41098,
                    ("NNP", "--unk_upper--"): 4635,
                    ("NNS", "Arts"): 2,
                },
                "transition_counts": {
                    ("VBN", "TO"): 2142,
                    ("CC", "IN"): 1227,
                    ("VBN", "JJR"): 66,
                },
                "tag_counts": {"PRP": 17436, "UH": 97, ")": 1376,},
            },
        },
        {
            "name": "small_case",
            "input": {
                "training_corpus": training_corpus[:1000],
                "vocab": vocab,
                "verbose": False,
            },
            "expected": {
                "len_emission_counts": 442,
                "len_transition_counts": 272,
                "len_tag_counts": 38,
                "emission_counts": {
                    ("DT", "the"): 48,
                    ("NNP", "--unk_upper--"): 9,
                    ("NNS", "Arts"): 1,
                },
                "transition_counts": {
                    ("VBN", "TO"): 3,
                    ("CC", "IN"): 2,
                    ("VBN", "JJR"): 1,
                },
                "tag_counts": {"PRP": 11, "UH": 0, ")": 2,},
            },
        },
    ]

    for test_case in test_cases:
        result_emission, result_transition, result_tag = target(**test_case["input"])

        # emission dictionary
        try:
            assert isinstance(result_emission, defaultdict)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": defaultdict,
                    "got": type(result_emission),
                }
            )
            print(
                f"Wrong output type for emission_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result_emission) == test_case["expected"]["len_emission_counts"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["len_emission_counts"],
                    "got": len(result_emission),
                }
            )
            print(
                f"Wrong output length for emission_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            for k, v in test_case["expected"]["emission_counts"].items():
                assert np.isclose(result_emission[k], v)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["emission_counts"],
                    "got": result_emission,
                }
            )
            print(
                f"Wrong output values for emission_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')}."
            )

        # transition dictionary
        try:
            assert isinstance(result_transition, defaultdict)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": defaultdict,
                    "got": type(result_transition),
                }
            )
            print(
                f"Wrong output type for transition_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert (
                len(result_transition) == test_case["expected"]["len_transition_counts"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["len_transition_counts"],
                    "got": len(result_transition),
                }
            )
            print(
                f"Wrong output length for transition_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            for k, v in test_case["expected"]["transition_counts"].items():
                assert np.isclose(result_transition[k], v)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["transition_counts"],
                    "got": result_transition,
                }
            )
            print(
                f"Wrong output values for transition_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')}."
            )

        # tags count
        try:
            assert isinstance(result_tag, defaultdict)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": defaultdict,
                    "got": type(result_transition),
                }
            )
            print(
                f"Wrong output type for tag_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result_tag) == test_case["expected"]["len_tag_counts"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["len_tag_counts"],
                    "got": len(result_tag),
                }
            )
            print(
                f"Wrong output length for tag_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            for k, v in test_case["expected"]["tag_counts"].items():
                assert np.isclose(result_tag[k], v)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["tag_counts"],
                    "got": result_tag,
                }
            )
            print(
                f"Wrong output values for tag_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases
    
    
    
def test_predict_pos(target, prep, y, emission_counts, vocab, states):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "prep": prep,
                "y": y,
                "emission_counts": emission_counts,
                "vocab": vocab,
                "states": states,
            },
            "expected": 0.8888563993099213,
        },
        {
            "name": "small_check",
            "input": {
                "prep": prep[:1000],
                "y": y[:1000],
                "emission_counts": emission_counts,
                "vocab": vocab,
                "states": states,
            },
            "expected": 0.876,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])
        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output values for tag_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases
    
    
def test_initialize(target, states, tag_counts, A, B, corpus, vocab):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "states": states,
                "tag_counts": tag_counts,
                "A": A,
                "B": B,
                "corpus": corpus,
                "vocab": vocab,
            },
            "expected": {
                "best_probs_shape": (46, 34199),
                "best_paths_shape": (46, 34199),
                "best_probs_col0": np.array(
                    [
                        -22.60982633,
                        -23.07660654,
                        -23.57298822,
                        -19.76726066,
                        -24.74325104,
                        -35.20241402,
                        -35.00096024,
                        -34.99203854,
                        -21.35069072,
                        -19.85767814,
                        -21.92098414,
                        -4.01623741,
                        -19.16380593,
                        -21.1062242,
                        -20.47163973,
                        -21.10157273,
                        -21.49584851,
                        -20.4811853,
                        -18.25856307,
                        -23.39717471,
                        -21.92146798,
                        -9.41377777,
                        -21.03053445,
                        -21.08029591,
                        -20.10863677,
                        -33.48185979,
                        -19.47301382,
                        -20.77150242,
                        -20.11727696,
                        -20.56031676,
                        -20.57193964,
                        -32.30366295,
                        -18.07551522,
                        -22.58887909,
                        -19.1585905,
                        -16.02994331,
                        -24.30968545,
                        -20.92932218,
                        -21.96797222,
                        -24.29571895,
                        -23.45968569,
                        -22.43665883,
                        -20.46568904,
                        -22.75551606,
                        -19.6637215,
                        -18.36288463,
                    ]
                ),
            },
        }
    ]

    for test_case in test_cases:
        result_best_probs, result_best_paths = target(**test_case["input"])

        try:
            assert isinstance(result_best_probs, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]) + "index 0",
                    "expected": np.ndarray,
                    "got": type(result_best_probs),
                }
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result_best_paths, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]) + "index 1",
                    "expected": np.ndarray,
                    "got": type(result_best_paths),
                }
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_best_probs.shape == test_case["expected"]["best_probs_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_probs_shape"],
                    "got": result_best_probs.shape,
                }
            )
            print(
                f"Wrong output shape for best_probs.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_best_paths.shape == test_case["expected"]["best_paths_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_paths_shape"],
                    "got": result_best_paths.shape,
                }
            )
            print(
                f"Wrong output shape for best_paths.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_probs[:, 0], test_case["expected"]["best_probs_col0"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_probs_col0"],
                    "got": result_best_probs[:, 0],
                }
            )
            print(
                f"Wrong non-zero values for best_probs.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.all((result_best_paths == 0))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": "Array of zeros with shape (46, 34199)",
                }
            )
            print(
                f"Wrong values for best_paths.\n\t Expected: {failed_cases[-1].get('expected')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases
    
    
   
def test_viterbi_forward(target, A, B, test_corpus, vocab):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "A": A,
                "B": B,
                "test_corpus": test_corpus,
                "best_probs": pickle.load(
                    open("./support_files/best_probs_initilized.pkl", "rb")
                ),
                "best_paths": pickle.load(
                    open("./support_files/best_paths_initilized.pkl", "rb")
                ),
                "vocab": vocab,
                "verbose": False,
            },
            "expected": {
                "best_probs0:5": np.array(
                    [
                        [
                            -22.60982633,
                            -24.78215633,
                            -34.08246498,
                            -34.34107105,
                            -49.56012613,
                        ],
                        [
                            -23.07660654,
                            -24.51583896,
                            -35.04774303,
                            -35.28281026,
                            -50.52540418,
                        ],
                        [
                            -23.57298822,
                            -29.98305064,
                            -31.98004656,
                            -38.99187549,
                            -47.45770771,
                        ],
                        [
                            -19.76726066,
                            -25.7122143,
                            -31.54577612,
                            -37.38331695,
                            -47.02343727,
                        ],
                        [
                            -24.74325104,
                            -28.78696025,
                            -31.458494,
                            -36.00456711,
                            -46.93615515,
                        ],
                    ]
                ),
                "best_probs30:35": np.array(
                    [
                        [
                            -202.75618827,
                            -208.38838519,
                            -210.46938402,
                            -210.15943098,
                            -223.79223672,
                        ],
                        [
                            -202.58297597,
                            -217.72266765,
                            -207.23725672,
                            -215.529735,
                            -224.13957203,
                        ],
                        [
                            -202.00878092,
                            -214.23093833,
                            -217.41021623,
                            -220.73768708,
                            -222.03338753,
                        ],
                        [
                            -200.44016117,
                            -209.46937757,
                            -209.06951664,
                            -216.22297765,
                            -221.09669653,
                        ],
                        [
                            -208.74189499,
                            -214.62088817,
                            -209.79346523,
                            -213.52623459,
                            -228.70417526,
                        ],
                    ]
                ),
                "best_paths0:5": np.array(
                    [
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                    ]
                ),
                "best_paths30:35": np.array(
                    [
                        [20, 19, 35, 11, 21],
                        [20, 19, 35, 11, 21],
                        [20, 19, 35, 11, 21],
                        [20, 19, 35, 11, 21],
                        [35, 19, 35, 11, 34],
                    ]
                ),
            },
        }
    ]

    for test_case in test_cases:
        result_best_probs, result_best_paths = target(**test_case["input"])

        try:
            assert isinstance(result_best_probs, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]) + "index 0",
                    "expected": np.ndarray,
                    "got": type(result_best_probs),
                }
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result_best_paths, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]) + "index 1",
                    "expected": np.ndarray,
                    "got": type(result_best_paths),
                }
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_probs[0:5, 0:5], test_case["expected"]["best_probs0:5"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_probs0:5"],
                    "got": result_best_probs[0:5, 0:5],
                }
            )
            print(
                f"Wrong values for best_probs.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_probs[30:35, 30:35],
                test_case["expected"]["best_probs30:35"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_probs30:35"],
                    "got": result_best_probs[:, 0],
                }
            )
            print(
                f"Wrong values for best_probs.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_paths[0:5, 0:5], test_case["expected"]["best_paths0:5"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_paths0:5"],
                    "got": result_best_paths[0:5, 0:5],
                }
            )
            print(
                f"Wrong values for best_paths.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_paths[30:35, 30:35],
                test_case["expected"]["best_paths30:35"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_paths30:35"],
                    "got": result_best_paths[30:35, 30:35],
                }
            )
            print(
                f"Wrong values for best_paths.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases
    
    
def test_viterbi_backward(target, corpus, states):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "corpus": corpus,
                "best_probs": pickle.load(
                    open("./support_files/best_probs_trained.pkl", "rb")
                ),
                "best_paths": pickle.load(
                    open("./support_files/best_paths_trained.pkl", "rb")
                ),
                "states": states,
            },
            "expected": {
                "pred_len": 34199,
                "pred_head": [
                    "DT",
                    "NN",
                    "POS",
                    "NN",
                    "MD",
                    "VB",
                    "VBN",
                    "IN",
                    "JJ",
                    "NN",
                ],
                "pred_tail": [
                    "PRP",
                    "MD",
                    "RB",
                    "VB",
                    "PRP",
                    "RB",
                    "IN",
                    "PRP",
                    ".",
                    "--s--",
                ],
            },
        }
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": str(test_case["name"]), "expected": list, "got": type(result)}
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result) == test_case["expected"]["pred_len"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["pred_len"],
                    "got": len(result),
                }
            )
            print(
                f"Wrong output lenght.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert result[:10] == test_case["expected"]["pred_head"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["pred_head"],
                    "got": result[:10],
                }
            )
            print(
                f"Wrong values for pred list.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert result[-10:] == test_case["expected"]["pred_tail"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["pred_tail"],
                    "got": result[-10:],
                }
            )
            print(
                f"Wrong values for pred list.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases
    
    
def test_compute_accuracy(target, pred, y):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"pred": pred, "y": y},
            "expected": 0.953063647155511,
        },
        {
            "name": "small_check",
            "input": {"pred": pred[:100], "y": y[:100]},
            "expected": 0.979381443298969,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, float)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": float,
                    "got": type(result),
                }
            )
            print(
                f"Wrong output type.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": float,
                    "got": type(result),
                }
            )
            print(
                f"Wrong output type.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases