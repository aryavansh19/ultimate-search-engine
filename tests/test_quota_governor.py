import time
import unittest

from quota.governor import (
    AdaptivePacer,
    QuotaGovernor,
    QuotaState,
    SlidingWindowMeter,
)


class TestSlidingWindowMeter(unittest.TestCase):
    def test_record_and_metrics(self):
        meter = SlidingWindowMeter(short_window_s=60.0, long_window_s=86400.0)
        t0 = 1000.0

        meter.record_request(tokens=100, now_mono=t0)
        meter.record_request(tokens=200, now_mono=t0 + 10)

        rpm, tpm, rpd, tpd = meter.metrics(now_mono=t0 + 20)
        self.assertEqual(rpm, 2)
        self.assertEqual(tpm, 300)
        self.assertEqual(rpd, 2)
        self.assertEqual(tpd, 300)
        self.assertEqual(meter.lifetime_requests, 2)
        self.assertEqual(meter.lifetime_tokens, 300)

    def test_window_expiration(self):
        meter = SlidingWindowMeter(short_window_s=60.0, long_window_s=86400.0)
        t0 = 1000.0

        meter.record_request(tokens=50, now_mono=t0)
        meter.record_request(tokens=50, now_mono=t0 + 30)

        # At t0 + 65, first request expired from 60s window but remains in 24h window
        rpm, tpm, rpd, tpd = meter.metrics(now_mono=t0 + 65)
        self.assertEqual(rpm, 1)
        self.assertEqual(tpm, 50)
        self.assertEqual(rpd, 2)
        self.assertEqual(tpd, 100)

    def test_time_to_free_capacity(self):
        meter = SlidingWindowMeter(short_window_s=60.0)
        t0 = 1000.0

        meter.record_request(tokens=10, now_mono=t0)
        wait = meter.time_to_free_capacity(needed_requests=1, now_mono=t0 + 10)
        self.assertAlmostEqual(wait, 50.0, places=1)


class TestAdaptivePacer(unittest.TestCase):
    def test_healthy_state(self):
        pacer = AdaptivePacer(max_rpm=10.0, warning_threshold=0.75, critical_threshold=0.90)
        meter = SlidingWindowMeter(short_window_s=60.0)
        t0 = 1000.0

        wait, state, _ = pacer.calculate_wait(meter, now_mono=t0)
        self.assertEqual(state, QuotaState.HEALTHY)
        self.assertEqual(wait, 0.0)

    def test_warning_and_critical_pacing(self):
        pacer = AdaptivePacer(max_rpm=10.0, warning_threshold=0.75, critical_threshold=0.90)
        meter = SlidingWindowMeter(short_window_s=60.0)
        t0 = 1000.0

        # Add 7 requests: ratio will be (7+1)/10 = 0.80 (>= 0.75 warning)
        for i in range(7):
            meter.record_request(now_mono=t0 + i)

        pacer.mark_request_dispatched(now_mono=t0 + 6.9)
        wait, state, reason = pacer.calculate_wait(meter, now_mono=t0 + 7.0)
        self.assertEqual(state, QuotaState.WARNING)
        self.assertGreater(wait, 0.0)

    def test_cooldown_on_429(self):
        pacer = AdaptivePacer(max_rpm=10.0)
        meter = SlidingWindowMeter(short_window_s=60.0)

        # Trigger 429 with retry_after=5.0s
        duration = pacer.trigger_backoff(retry_after=5.0)
        self.assertEqual(duration, 5.0)

        wait, state, _ = pacer.calculate_wait(meter)
        self.assertEqual(state, QuotaState.COOLING_DOWN)
        self.assertGreater(wait, 0.0)


class TestQuotaGovernor(unittest.TestCase):
    def test_telemetry_snapshot(self):
        gov = QuotaGovernor(service_name="test_service", max_rpm=10.0, max_tpm=1000, max_rpd=100)
        gov.acquire(estimated_tokens=50)
        gov.record_success(actual_tokens=50)

        telem = gov.telemetry()
        self.assertEqual(telem.service, "test_service")
        self.assertEqual(telem.rpm, 1)
        self.assertEqual(telem.rpm_utilization_pct, 10.0)
        self.assertEqual(telem.status, "healthy")
        self.assertEqual(telem.lifetime_requests, 1)
        self.assertEqual(telem.lifetime_tokens, 100)  # estimated 50 + actual 50

        gov.record_error(status_code=429, retry_after=2.0)
        telem_after = gov.telemetry()
        self.assertEqual(telem_after.status, "cooling_down")
        self.assertGreater(telem_after.cooldown_remaining_s, 0.0)
        self.assertEqual(telem_after.error_counts.get(429), 1)


if __name__ == "__main__":
    unittest.main()
