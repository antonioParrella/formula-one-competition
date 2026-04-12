import pytest
import sys
sys.path.insert(0, "src")
from tips_parser import TipsParser


class TestDeduplicateSubmissions:
    def test_no_duplicates(self):
        """No changes when all players are unique."""
        parser = TipsParser(None)
        submissions = [
            {"player": "Alice", "submitted_at": "2026-03-26T10:00:00", "main_race": []},
            {"player": "Bob", "submitted_at": "2026-03-26T10:01:00", "main_race": []},
        ]
        result = parser._deduplicate_submissions(submissions)
        assert len(result) == 2

    def test_keeps_latest(self):
        """Keeps the latest submission when player submits twice."""
        parser = TipsParser(None)
        submissions = [
            {"player": "Alice", "submitted_at": "2026-03-26T10:00:00", "main_race": ["VER"]},
            {"player": "Alice", "submitted_at": "2026-03-26T11:00:00", "main_race": ["NOR"]},
        ]
        result = parser._deduplicate_submissions(submissions)
        assert len(result) == 1
        assert result[0]["main_race"] == ["NOR"]

    def test_multiple_players_multiple_duplicates(self):
        """Handles multiple players with multiple duplicates."""
        parser = TipsParser(None)
        submissions = [
            {"player": "Alice", "submitted_at": "2026-03-26T10:00:00"},
            {"player": "Alice", "submitted_at": "2026-03-26T11:00:00"},
            {"player": "Bob", "submitted_at": "2026-03-26T09:00:00"},
            {"player": "Bob", "submitted_at": "2026-03-26T12:00:00"},
            {"player": "Charlie", "submitted_at": "2026-03-26T08:00:00"},
        ]
        result = parser._deduplicate_submissions(submissions)
        assert len(result) == 3
        assert result[0]["player"] == "Alice"
        assert result[0]["submitted_at"] == "2026-03-26T11:00:00"
        assert result[1]["player"] == "Bob"
        assert result[1]["submitted_at"] == "2026-03-26T12:00:00"

    def test_empty_list(self):
        """Handles empty list."""
        parser = TipsParser(None)
        result = parser._deduplicate_submissions([])
        assert result == []

    def test_unknown_player(self):
        """Handles 'Unknown' players separately from named players."""
        parser = TipsParser(None)
        submissions = [
            {"player": "Unknown", "submitted_at": "2026-03-26T10:00:00"},
            {"player": "Unknown", "submitted_at": "2026-03-26T11:00:00"},
        ]
        result = parser._deduplicate_submissions(submissions)
        assert len(result) == 1