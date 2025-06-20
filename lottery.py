

""" lottery related class """

__author__ = 'shxrkidd'

from dataclasses import dataclass, field
from typing import List


@dataclass
class LotteryItem:
    """Represents a single lottery item from a company."""
    company: str = ""
    id: int = 0
    w_odds: float = 0.0
    d_odds: float = 0.0
    l_odds: float = 0.0
    cw_odds: float = 0.0
    cd_odds: float = 0.0
    cl_odds: float = 0.0
    back_ratio: float = 0.0
    count: int = 1

    def display(self):
        """Prints the details of the lottery item."""
        print(f"{self.id}\t{self.company}\t{self.cw_odds}\t{self.cd_odds}\t{self.cl_odds}\t")


@dataclass
class LotteryMatch:
    """Represents a single match with its associated lottery items."""
    match_name: str
    match_link: str
    match_time: str
    host_team: str
    guest_team: str
    item_arr: List[LotteryItem]

    def display(self):
        """Prints the details of the match."""
        print(f"match name:\t{self.match_name}\nmatch members:\t{self.host_team} VS {self.guest_team}\nmatch time:\t{self.match_time}")

    def display_items(self):
        """Displays all lottery items for the match."""
        for item in self.item_arr:
            item.display()


@dataclass
class LotteryPortfolio:
    """Represents a portfolio of lottery items for a match."""
    fund_count: float = 0.0
    profit: float = 0.0
    win_item: LotteryItem = field(default_factory=LotteryItem)
    draw_item: LotteryItem = field(default_factory=LotteryItem)
    lose_item: LotteryItem = field(default_factory=LotteryItem)
    win_percentage: float = 0.0
    draw_percentage: float = 0.0
    lose_percentage: float = 0.0

    def display(self):
        """Prints the details of the portfolio."""
        print(f"profit:\t{self.profit}\n"
              f"win:\t{self.win_item.id} {self.win_item.company}\t{self.win_item.cw_odds}\t{self.win_percentage}\n"
              f"draw:\t{self.draw_item.id} {self.draw_item.company}\t{self.draw_item.cd_odds}\t{self.draw_percentage}\n"
              f"lose:\t{self.lose_item.id} {self.lose_item.company}\t{self.lose_item.cl_odds}\t{self.lose_percentage}")