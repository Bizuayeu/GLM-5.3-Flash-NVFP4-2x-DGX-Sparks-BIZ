import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup.runtime import patch_sampler_nonfinite as sampler

# The lines of vLLM 385dce36 v1/worker/gpu/sample/gumbel.py around the one site.
GUMBEL = """# SPDX-License-Identifier: Apache-2.0
import torch


@triton.jit
def _gumbel_sample_kernel(
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    value, idx = gumbel_block_argmax(
        logits,
        vocab_size,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)
"""

# The lines of vLLM 385dce36 v1/worker/gpu/spec_decode/rejection_sampler_utils.py around
# its two sites, at their own indents: the greedy branch (8) and the resample kernel (4).
REJECTION = """# SPDX-License-Identifier: Apache-2.0
import torch


@triton.jit
def _compute_local_logits_stats_kernel(
    vocab_size,
):
    if temp == 0.0:
        # Greedy sampling. Only the target max/argmax are needed.
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            target_local_argmax_ptr
            + logit_idx * target_local_argmax_stride
            + block_idx,
            token_id,
        )
    else:
        num_blocks = tl.minimum(vocab_num_blocks, PADDED_VOCAB_NUM_BLOCKS)


@triton.jit
def _resample_kernel(
    vocab_size,
):
    value, idx = gumbel_block_argmax(
        residual_logits,
        vocab_size,
        IS_DRAFTING=False,
        APPLY_TEMPERATURE=False,
        USE_FP64=USE_FP64,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + block_idx,
        token_id,
    )
"""

BOUNDED = "token_id = tl.minimum(block_idx * BLOCK_SIZE + idx, vocab_size - 1)\n"
UNBOUNDED = "token_id = block_idx * BLOCK_SIZE + idx\n"


class PatchTests(unittest.TestCase):
    def test_each_of_the_three_sites_is_bounded_once(self):
        gumbel = sampler.patch_gumbel(GUMBEL)
        self.assertEqual(gumbel.count("    " + BOUNDED), 1)
        rejection = sampler.patch_rejection(REJECTION)
        self.assertEqual(rejection.count("        " + BOUNDED), 1)  # greedy branch
        self.assertEqual(rejection.count("\n    " + BOUNDED), 1)  # resample kernel
        for patched in (gumbel, rejection):
            self.assertNotIn(UNBOUNDED, patched.replace(BOUNDED, ""))
            self.assertTrue(patched.startswith("# Modified by GLM setup"))
        # The rest of each file is the pinned text: only the sites and their comments change.
        for original, patched in ((GUMBEL, gumbel), (REJECTION, rejection)):
            kept = [
                line
                for line in patched.removeprefix(sampler.HEADER).splitlines()
                if not line.lstrip().startswith("#")
                and "tl.minimum(block_idx" not in line
            ]
            pinned = [
                line
                for line in original.splitlines()
                if not line.lstrip().startswith("#") and UNBOUNDED.strip() not in line
            ]
            self.assertEqual(kept, pinned)

    def test_the_edits_are_the_upstream_pull_request_lines(self):
        # vllm-project/vllm #50843 at 3737f51f: one comment block per site, then the clamp.
        gumbel = sampler.patch_gumbel(GUMBEL)
        self.assertIn(
            "    # Clamp to provide an addressable in-vocabulary fallback. This does not\n"
            "    # define a semantically correct token or fix the source of non-finite logits.\n"
            "    " + BOUNDED,
            gumbel,
        )
        rejection = sampler.patch_rejection(REJECTION)
        self.assertIn(
            "        # the source of non-finite logits.\n        " + BOUNDED, rejection
        )
        self.assertIn(
            "    # _compute_local_logits_stats_kernel for the resampling path.\n    "
            + BOUNDED,
            rejection,
        )

    def test_refuses_drifted_or_already_patched_source(self):
        with self.assertRaises(ValueError):
            sampler.patch_gumbel(GUMBEL.replace(UNBOUNDED, "token_id = idx\n"))
        # One rejection site gone: the other alone is not enough.
        drifted = REJECTION.replace(
            "    )\n    " + UNBOUNDED, "    )\n    token_id = idx\n"
        )
        with self.assertRaises(ValueError):
            sampler.patch_rejection(drifted)
        with self.assertRaisesRegex(ValueError, "already applied"):
            sampler.patch_gumbel(sampler.patch_gumbel(GUMBEL))
        with self.assertRaisesRegex(ValueError, "already applied"):
            sampler.patch_rejection(sampler.patch_rejection(REJECTION))


class PrepareTests(unittest.TestCase):
    def package(self, directory):
        package = Path(directory) / "vllm"
        for name, text in ((sampler.GUMBEL, GUMBEL), (sampler.REJECTION, REJECTION)):
            (package / name).parent.mkdir(parents=True, exist_ok=True)
            # Bytes, not write_text: on Windows that writes CRLF and the anchors miss.
            (package / name).write_bytes(text.encode("utf-8"))
        return package

    def test_both_files_are_checked_before_either_is_patched(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.package(directory)
            hashes = {
                name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                for name in sampler.SOURCES
            }
            original = dict(sampler.SOURCES)
            try:
                sampler.SOURCES.update(hashes)
                self.assertEqual(set(sampler.prepare(package)), set(hashes))
                sampler.SOURCES[sampler.REJECTION] = "0" * 64
                with self.assertRaisesRegex(
                    ValueError, "sampler vocab bound source hash mismatch"
                ):
                    sampler.prepare(package)
            finally:
                sampler.SOURCES.clear()
                sampler.SOURCES.update(original)

    def test_the_pins_are_the_pinned_vllm_files(self):
        # sha256 of `git show 385dce36:vllm/<target>`, as for the other patches.
        self.assertEqual(
            sampler.SOURCES,
            {
                "v1/worker/gpu/sample/gumbel.py": (
                    "3ec1df510bdad13e8b0a457b5b67c98affdf6e57369e56cb471246cc8f40dd34"
                ),
                "v1/worker/gpu/spec_decode/rejection_sampler_utils.py": (
                    "20ca2e5ac34e9ef93dca388bed72a00d4ff67d569ec6ec2393b21a92d74a1fae"
                ),
            },
        )

    def test_a_second_run_meets_patched_files_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.package(directory)
            hashes = {
                name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                for name in sampler.SOURCES
            }
            with patch.dict(sampler.SOURCES, hashes):
                with contextlib.redirect_stdout(io.StringIO()):
                    sampler.main(["--package", str(package)])
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    sampler.prepare(package)

    def test_the_command_writes_both_files_and_the_image_record(self):
        # The record is baked into images; its text and the printed line are evidence.
        outputs = {sampler.GUMBEL: b"gumbel", sampler.REJECTION: b"rejection"}
        row = {
            name: {
                "source_sha256": sampler.SOURCES[name],
                "patched_sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in outputs.items()
        }
        for check in (False, True):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as tmp:
                package = Path(tmp) / "vllm"
                for name in outputs:
                    (package / name).parent.mkdir(parents=True, exist_ok=True)
                    (package / name).write_bytes(b"pinned")
                printed = io.StringIO()
                argv = ["--package", str(package)] + (["--check"] if check else [])
                with (
                    patch.object(sampler, "prepare", return_value=outputs),
                    contextlib.redirect_stdout(printed),
                ):
                    sampler.main(argv)
                self.assertEqual(
                    printed.getvalue(),
                    json.dumps({"files": row, "check_only": check}) + "\n",
                )
                record = Path(tmp) / "glm53-sampler-nonfinite-patch.json"
                if check:
                    self.assertFalse(record.exists())
                    self.assertEqual((package / sampler.GUMBEL).read_bytes(), b"pinned")
                else:
                    self.assertEqual(record.read_text(), json.dumps(row, indent=2))
                    for name, data in outputs.items():
                        self.assertEqual((package / name).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
