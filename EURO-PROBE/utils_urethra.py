import os
from typing import List


def get_available_delineators(EuroProbePath: str, verbose: bool = True) -> List[str]:
    """
    Arg:
        EuroProbePath (str): Path to the EURO-PROBE directory
        verbose (bool, optional): If True, print available delineators (default: True)
    Returns:
        avaliable_delineators (List[str]): Sorted list of available delineator identifiers
    """
    euroProbePath = os.path.join(EuroProbePath)
    available_delineators = set()

    if os.path.isdir(euroProbePath):
        for fname in os.listdir(euroProbePath):
            if fname.startswith("mask_prostate_") and fname.endswith(".nii.gz"):
                delineator = fname.replace("mask_prostate_", "").replace(".nii.gz", "")
                available_delineators.add(delineator)

    available_delineators = sorted(available_delineators)

    if verbose:
        print("Available delineators for this case:")
        if available_delineators:
            for d in available_delineators:
                print(f"  - {d}")
        else:
            print("  (none found)")

    return available_delineators


def validate_delineator(delineator: str, available_delineators: List[str]) -> None:
    """
    Arg:
        delineator (str): Selected delineator identifier
        available_delineators (List[str]): List of detected delineators
    Raises:
        ValueError: If the delineator is not available
    """
    if delineator not in available_delineators:
        raise ValueError(
            f"Delineator '{delineator}' not available for this case. "
            f"Available options: {available_delineators}"
        )
