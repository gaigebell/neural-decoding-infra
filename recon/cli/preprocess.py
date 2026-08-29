"""CLI: preprocessing pipeline entry point.

Usage:
    python -m recon.cli.preprocess stage=meg_zresp subject=1 stories=[1]
    python -m recon.cli.preprocess stage=semantic_delay subject=1 stories=1-5

Stages produce artifacts with sidecars; re-runs skip valid artifacts
(``resume=true``). See ``configs/preprocess.yaml`` for all parameters.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..preprocessing import fmri, meg, semantic
from ..preprocessing.common import sidecar_valid
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _stories(cfg: DictConfig) -> list[int]:
    stories = cfg.get("stories")
    if stories is None:
        return list(range(1, 61))
    if isinstance(stories, int):
        return [stories]
    out: list[int] = []
    for item in stories:
        if isinstance(item, str) and "-" in item:  # "1-5" range syntax
            a, b = item.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(item))
    return out


def _eliminate(cfg: DictConfig, story: int) -> list[int] | None:
    ed = cfg.get("eliminate_data") or {}
    ed = dict(ed)
    return list(ed[story]) if story in ed else None


def _run_stage(cfg: DictConfig, sub_id: int, story: int) -> None:
    stage = cfg.stage
    layer = int(cfg.layer)
    paths = cfg.paths

    if stage == "gpt_char_features":
        mat = Path(paths.data_root) / cfg.char_time_pattern.format(sub=sub_id, story=story)
        save = Path(cfg.wordvector_dir)
        layers = [int(x) for x in cfg.layers]
        # Skip only if EVERY layer's artifact has a valid sidecar
        all_valid = all(
            sidecar_valid(
                save / f"story{story}_{L}.npy",
                {"mat": mat},
                {
                    "stage": stage, "story_id": story, "layers": [L],
                    "context_len": int(cfg.context_len), "gpt_path": paths.gpt_path,
                },
            )
            for L in layers
        )
        if cfg.resume and all_valid:
            logger.info("skip (sidecar valid): %s layers", layers)
            return
        semantic.extract_gpt_char_features(
            mat, paths.gpt_path, save, story,
            layers=layers,
            eliminate=_eliminate(cfg, story),
            context_len=int(cfg.context_len),
            device=cfg.gpt_device,
        )
        return

    if stage == "semantic_downsample":
        wv = Path(cfg.wordvector_dir) / f"story{story}_{layer}.npy"
        mat = Path(paths.data_root) / cfg.char_time_pattern.format(sub=sub_id, story=story)
        artifact = Path(cfg.ds_dir) / f"ds_{story}_{layer}.npy"
        params = {
            "stage": stage, "story_id": story, "layer": layer,
            "first_char_onset_time": float(cfg.first_char_onset_time),
            "interval": float(cfg.interval), "eliminate": _eliminate(cfg, story),
        }
        if cfg.resume and sidecar_valid(artifact, {"wordvectors": wv, "mat": mat}, params):
            logger.info("skip (sidecar valid): %s", artifact)
            return
        semantic.downsample_story(
            wv, mat, Path(cfg.ds_dir), story, layer,
            first_char_onset_time=float(cfg.first_char_onset_time),
            interval=float(cfg.interval),
            eliminate=_eliminate(cfg, story),
        )
        return

    if stage == "semantic_delay":
        ds = Path(cfg.ds_dir) / f"ds_{story}_{layer}.npy"
        artifact = Path(cfg.zstim_dir) / f"sub{sub_id}_zstim_{layer}_story{story}.npy"
        params = {
            "stage": stage, "subject_id": sub_id, "story_id": story, "layer": layer,
            "delays": [int(d) for d in cfg.delays],
        }
        if cfg.resume and sidecar_valid(artifact, {"ds": ds}, params):
            logger.info("skip (sidecar valid): %s", artifact)
            return
        semantic.delay_story(ds, Path(cfg.zstim_dir), sub_id, story, layer, delays=[int(d) for d in cfg.delays])
        return

    if stage == "meg_zresp":
        fif = Path(paths.data_root) / cfg.meg_fif_pattern.format(sub=sub_id, story=story)
        ds = Path(cfg.ds_dir) / f"ds_{story}_{layer}.npy"
        artifact = Path(cfg.zresp_dir) / f"zresp{sub_id}_{story}.npy"
        params = {
            "stage": stage, "subject_id": sub_id, "story_id": story, "layer": layer,
            "interval": float(cfg.interval),
            "first_char_onset_time": float(cfg.first_char_onset_time),
            "channel_start": int(cfg.channel_start), "channel_end": int(cfg.channel_end),
        }
        if cfg.resume and sidecar_valid(artifact, {"fif": fif, "ds": ds}, params):
            logger.info("skip (sidecar valid): %s", artifact)
            return
        meg.process_story(
            fif, ds, Path(cfg.zresp_dir), sub_id, story, layer,
            interval=float(cfg.interval),
            first_char_onset_time=float(cfg.first_char_onset_time),
            channel_start=int(cfg.channel_start), channel_end=int(cfg.channel_end),
        )
        return

    if stage == "meg_context":
        zresp = Path(cfg.zresp_dir) / f"zresp{sub_id}_{story}.npy"
        artifact = Path(cfg.zresp_dir) / f"zresp{sub_id}_{story}_context_{int(cfg.n_context)}.npy"
        params = {"stage": stage, "subject_id": sub_id, "story_id": story, "n_context": int(cfg.n_context)}
        if cfg.resume and sidecar_valid(artifact, {"zresp": zresp}, params):
            logger.info("skip (sidecar valid): %s", artifact)
            return
        meg.chunk_context(zresp, Path(cfg.zresp_dir), sub_id, story, n_context=int(cfg.n_context))
        return

    if stage == "fmri_cube":
        nii = Path(paths.data_root) / cfg.fmri_nii_pattern.format(sub=sub_id, story=story)
        ds = Path(cfg.ds_dir) / f"ds_{story}_{layer}.npy"
        artifact = Path(cfg.fmri_zresp_dir) / f"zresp{sub_id}_{story}.npy"
        params = {
            "stage": stage, "subject_id": sub_id, "story_id": story, "layer": layer,
            "start_row": int(cfg.fmri_start_row),
        }
        if cfg.resume and sidecar_valid(artifact, {"nii": nii, "ds": ds}, params):
            logger.info("skip (sidecar valid): %s", artifact)
            return
        fmri.process_story(nii, ds, Path(cfg.fmri_zresp_dir), sub_id, story, layer, start_row=int(cfg.fmri_start_row))
        return

    raise ValueError(
        f"Unknown stage: {stage} (expected gpt_char_features | semantic_downsample | "
        "semantic_delay | meg_zresp | meg_context | fmri_cube)"
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    """Preprocessing entry point."""
    logger.info("Preprocessing: stage=%s subject=%s", cfg.stage, cfg.subject)
    sub_id = int(cfg.subject)
    for story in _stories(cfg):
        try:
            _run_stage(cfg, sub_id, story)
        except FileNotFoundError as e:
            logger.warning("story %d skipped: %s", story, e)


if __name__ == "__main__":
    main()
