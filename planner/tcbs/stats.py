import dataclasses


@dataclasses.dataclass
class StatsContainer:
    n_low_level_expanded: int = 0
    n_high_level_expanded: int = 0

    def inc_low_level_expanded(self):
        self.n_low_level_expanded += 1

    def inc_high_level_expanded(self):
        self.n_high_level_expanded += 1

    def add_low_level_expanded(self, n: int):
        self.n_low_level_expanded += n

    def as_dict(self):
        return dataclasses.asdict(self)
