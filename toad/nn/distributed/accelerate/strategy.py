from dataclasses import dataclass


@dataclass
class Strategy:
    method: str = None


@dataclass
class DDPStrategy(Strategy):
    method: str = "ddp"


@dataclass
class FSDPStrategy(DDPStrategy):
    method: str = "fsdp"
    policy: str = None
