"""Lock operations unit tests.

Tests for atomic lock extend/release using Lua scripts.
Requires Redis to be running (uses fakeredis for mocking).
"""

import pytest


class TestLockOperations:
    """Test lock acquire/extend/release operations."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis instance using fakeredis."""
        try:
            import fakeredis
            return fakeredis.FakeRedis(decode_responses=True)
        except ImportError:
            pytest.skip("fakeredis not installed")

    @pytest.fixture
    def worker_id(self, monkeypatch):
        """Set a fixed worker ID for testing."""
        monkeypatch.setenv("WORKER_ID", "test-worker-1")
        return "test-worker-1"

    def test_acquire_lock_success(self, mock_redis, worker_id, monkeypatch):
        """Test successful lock acquisition."""
        # Import after monkeypatch
        import sys
        sys.path.insert(0, "/tmp/pdf-yaml-check")

        # Patch WORKER_ID before importing
        monkeypatch.setattr("scripts.file_queue_worker.WORKER_ID", worker_id)

        from scripts.file_queue_worker import acquire_lock, get_lock_key

        file_name = "test.pdf"
        result = mock_redis.set(get_lock_key(file_name), worker_id, nx=True, ex=300)

        assert result is True
        assert mock_redis.get(get_lock_key(file_name)) == worker_id

    def test_acquire_lock_already_held(self, mock_redis, worker_id):
        """Test lock acquisition fails when already held."""
        import sys
        sys.path.insert(0, "/tmp/pdf-yaml-check")
        from scripts.file_queue_worker import get_lock_key

        file_name = "test.pdf"
        lock_key = get_lock_key(file_name)

        # Another worker already holds the lock
        mock_redis.set(lock_key, "other-worker", ex=300)

        # Try to acquire - should fail
        result = mock_redis.set(lock_key, worker_id, nx=True, ex=300)

        assert result is None or result is False
        assert mock_redis.get(lock_key) == "other-worker"

    def test_extend_lock_own_lock(self, mock_redis, worker_id, monkeypatch):
        """Test extending own lock succeeds."""
        import sys
        sys.path.insert(0, "/tmp/pdf-yaml-check")
        monkeypatch.setattr("scripts.file_queue_worker.WORKER_ID", worker_id)

        from scripts.file_queue_worker import get_lock_key, _EXTEND_LOCK_SCRIPT

        file_name = "test.pdf"
        lock_key = get_lock_key(file_name)

        # Acquire lock first
        mock_redis.set(lock_key, worker_id, ex=60)

        # Extend lock using Lua script
        result = mock_redis.eval(_EXTEND_LOCK_SCRIPT, 1, lock_key, worker_id, 300)

        assert result > 0
        assert mock_redis.ttl(lock_key) > 60  # TTL extended

    def test_extend_lock_other_worker_fails(self, mock_redis, worker_id, monkeypatch):
        """Test extending another worker's lock fails."""
        import sys
        sys.path.insert(0, "/tmp/pdf-yaml-check")
        monkeypatch.setattr("scripts.file_queue_worker.WORKER_ID", worker_id)

        from scripts.file_queue_worker import get_lock_key, _EXTEND_LOCK_SCRIPT

        file_name = "test.pdf"
        lock_key = get_lock_key(file_name)

        # Another worker holds the lock
        mock_redis.set(lock_key, "other-worker", ex=300)
        original_ttl = mock_redis.ttl(lock_key)

        # Try to extend - should fail
        result = mock_redis.eval(_EXTEND_LOCK_SCRIPT, 1, lock_key, worker_id, 600)

        assert result == 0
        assert mock_redis.get(lock_key) == "other-worker"  # Owner unchanged

    def test_release_lock_own_lock(self, mock_redis, worker_id, monkeypatch):
        """Test releasing own lock succeeds."""
        import sys
        sys.path.insert(0, "/tmp/pdf-yaml-check")
        monkeypatch.setattr("scripts.file_queue_worker.WORKER_ID", worker_id)

        from scripts.file_queue_worker import get_lock_key, _RELEASE_LOCK_SCRIPT

        file_name = "test.pdf"
        lock_key = get_lock_key(file_name)

        # Acquire lock first
        mock_redis.set(lock_key, worker_id, ex=300)

        # Release lock using Lua script
        result = mock_redis.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, worker_id)

        assert result > 0
        assert mock_redis.get(lock_key) is None  # Lock deleted

    def test_release_lock_other_worker_fails(self, mock_redis, worker_id, monkeypatch):
        """Test releasing another worker's lock fails (critical safety test)."""
        import sys
        sys.path.insert(0, "/tmp/pdf-yaml-check")
        monkeypatch.setattr("scripts.file_queue_worker.WORKER_ID", worker_id)

        from scripts.file_queue_worker import get_lock_key, _RELEASE_LOCK_SCRIPT

        file_name = "test.pdf"
        lock_key = get_lock_key(file_name)

        # Another worker holds the lock
        mock_redis.set(lock_key, "other-worker", ex=300)

        # Try to release - should fail
        result = mock_redis.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, worker_id)

        assert result == 0
        assert mock_redis.get(lock_key) == "other-worker"  # Lock still exists

    def test_release_nonexistent_lock(self, mock_redis, worker_id, monkeypatch):
        """Test releasing nonexistent lock returns 0."""
        import sys
        sys.path.insert(0, "/tmp/pdf-yaml-check")
        monkeypatch.setattr("scripts.file_queue_worker.WORKER_ID", worker_id)

        from scripts.file_queue_worker import get_lock_key, _RELEASE_LOCK_SCRIPT

        file_name = "nonexistent.pdf"
        lock_key = get_lock_key(file_name)

        # No lock exists
        result = mock_redis.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, worker_id)

        assert result == 0

    def test_race_condition_simulation(self, mock_redis, worker_id, monkeypatch):
        """Test that Lua script prevents race condition.

        Scenario: Worker A's lock expires, Worker B acquires it,
        then Worker A tries to extend/release - should fail.
        """
        import sys
        sys.path.insert(0, "/tmp/pdf-yaml-check")

        from scripts.file_queue_worker import get_lock_key, _EXTEND_LOCK_SCRIPT, _RELEASE_LOCK_SCRIPT

        file_name = "race.pdf"
        lock_key = get_lock_key(file_name)

        worker_a = "worker-a"
        worker_b = "worker-b"

        # Worker A's lock expires, Worker B acquires
        mock_redis.set(lock_key, worker_b, ex=300)

        # Worker A (stale reference) tries to extend
        extend_result = mock_redis.eval(_EXTEND_LOCK_SCRIPT, 1, lock_key, worker_a, 600)
        assert extend_result == 0, "Worker A should not extend Worker B's lock"

        # Worker A tries to release
        release_result = mock_redis.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, worker_a)
        assert release_result == 0, "Worker A should not release Worker B's lock"

        # Worker B's lock should be intact
        assert mock_redis.get(lock_key) == worker_b
