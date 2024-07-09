import numpy as np
import re
import sys
import os
import glob
import logging


def check_channelized_file_count(
    logger, output_dir, target_id, reference_ids, params, verbose=False
):
    # From here we will work in the output directory
    os.chdir(output_dir)

    logger.info("Checking that there are enough output files.")
    if verbose:
        print("Checking that there are enough output files.")
    for ref in reference_ids:
        for beam in params["ref_beams"]:
            all_files = glob.glob(f"{output_dir}{ref}/{beam}/*spw*.h5")
            if len(all_files) != params["spws"]:
                logger.error(
                    f"The number of channelized files is different than expected for reference: {ref}, beam: {beam}"
                )
                logger.error("Will not continue.")
                logger.error(f"{len(all_files)} != {params['spws']}")
                sys.exit(1)

    all_files = glob.glob(f"{output_dir}{target_id}/*spw*.h5")
    if len(all_files) != params["target_beams"] * params["spws"]:
        logger.error(
            "The number of channelized files is different than expected for reference: {0}".format(
                ref
            )
        )
        logger.error("Will not continue.")
        sys.exit(1)

    logger.info("The number of channelized files is as expected. Continuing...")
    return True


def check_correlated_file_count(
    logger,
    output_dir,
    target_id,
    reference_ids,
    xcorr_dt,
    params,
    verbose=False,
    return_missing=False,
):
    logger.info("Checking that there are enough output files in subfolder {}")
    if verbose:
        print("Checking that there are enough output files.")

    missing = []
    _missing = False

    for ref in reference_ids:
        base = f"{output_dir}{target_id}_xcorr/{ref}/*"
        # for ibm in params['xcorr_beams']:
        for spw in range(params["spws"]):
            fn = f"{base}/SAP*_B*_*spw{spw}_avg{xcorr_dt}.h5"
            all_files = glob.glob(fn)
            if len(all_files) != len(reference_ids) * len(params["ref_beams"]) * len(
                params["xcorr_beams"]
            ):
                _missing = True

                for ref_beam in params["ref_beams"]:
                    for _ibm in params["xcorr_beams"]:
                        _base = f"{output_dir}{target_id}_xcorr/{ref}/{ref_beam}"
                        fn = f"{_base}/SAP*_B{str(_ibm).zfill(3)}_*spw{spw}_avg{xcorr_dt}.h5"
                        all_files = glob.glob(fn)
                        if len(all_files) == 0:
                            missing.append((ref, ref_beam, spw, _ibm))

    if _missing:
        logger.info(f"Missing files: {missing}")
        return False, np.array(missing) if return_missing else None

    logger.info("The number of correlated files is as expected. Continuing...")
    return True, missing if return_missing else None
