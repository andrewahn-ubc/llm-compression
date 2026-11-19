from __future__ import annotations

import math
import unittest

import torch

from llm_compression.losses import (
    forward_kl,
    lambda_schedule,
    mixed_local_loss,
    relative_error,
)


class RelativeErrorTests(unittest.TestCase):
    def test_exact_ratio_and_gradient(self) -> None:
        source = torch.zeros(1, 2, 2)
        target = torch.ones(1, 2, 2)
        prediction = torch.full((1, 2, 2), 2.0, requires_grad=True)
        loss = relative_error(prediction, target, source)
        self.assertAlmostEqual(loss.item(), 1.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_mask_excludes_padding(self) -> None:
        source = torch.zeros(1, 2, 1)
        target = torch.ones(1, 2, 1)
        prediction = torch.tensor([[[2.0], [101.0]]])
        masked = relative_error(
            prediction,
            target,
            source,
            attention_mask=torch.tensor([[1, 0]]),
        )
        self.assertAlmostEqual(masked.item(), 1.0)

    def test_mixed_loss_uses_shared_teacher_delta(self) -> None:
        source = torch.zeros(1, 1, 2)
        target = torch.ones(1, 1, 2)
        clean_prediction = target.clone()
        student_prediction = target + 2
        loss = mixed_local_loss(
            clean_prediction, student_prediction, target, source, 0.25
        )
        self.assertAlmostEqual(loss.item(), 3.0)


class ScheduleTests(unittest.TestCase):
    def test_linear_and_cosine_endpoints(self) -> None:
        self.assertEqual(lambda_schedule(0, 5), 1.0)
        self.assertEqual(lambda_schedule(4, 5), 0.5)
        self.assertEqual(lambda_schedule(99, 5), 0.5)
        self.assertAlmostEqual(
            lambda_schedule(2, 5, start=1.0, end=0.0, kind="cosine"), 0.5
        )
        self.assertEqual(
            lambda_schedule(3, 10, start=0.5, end=0.5, kind="fixed"), 0.5
        )
        self.assertEqual(lambda_schedule(3, 10, kind="teacher_only"), 1.0)


class ForwardKLTests(unittest.TestCase):
    def test_matches_manual_forward_kl_and_has_student_gradient(self) -> None:
        teacher = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
        student = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]], requires_grad=True)
        loss = forward_kl(teacher, student, temperature=2.0)
        teacher_p = torch.softmax(teacher.detach() / 2.0, dim=-1)
        student_log_p = torch.log_softmax(student / 2.0, dim=-1)
        expected = (
            teacher_p * (teacher_p.log() - student_log_p)
        ).sum(dim=-1).mean() * 4.0
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)

    def test_masked_identity_is_zero(self) -> None:
        logits = torch.randn(2, 3, 5)
        loss = forward_kl(
            logits,
            logits.clone().requires_grad_(),
            attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
        )
        self.assertTrue(math.isclose(loss.item(), 0.0, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
