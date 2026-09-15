import pytest

from src import data_loader


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])


def test_loader_applies_limit_without_exceeding_dataset(monkeypatch):
    fake = FakeDataset([{"text": str(i), "rating": 5} for i in range(3)])

    def fake_load_dataset(name, config, split):
        assert name == "McAuley-Lab/Amazon-Reviews-2023"
        assert config == "raw_review_Video_Games"
        assert split == "full"
        return fake

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)
    loaded = data_loader.load_video_game_reviews(limit=10)
    assert len(loaded) == 3


@pytest.mark.parametrize("limit", [0, -1, True])
def test_loader_rejects_invalid_limits(limit):
    with pytest.raises(ValueError, match="positive integer"):
        data_loader.load_video_game_reviews(limit=limit)
