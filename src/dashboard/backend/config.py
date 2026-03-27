"""
Configuration for FedPhish Demo Dashboard Backend.
"""

from typing import List
from pydantic import BaseModel


class BankConfig(BaseModel):
    """Configuration for a single bank."""

    bank_id: int
    name: str
    location: str  # City, State
    lat: float
    lon: float
    data_distribution: str  # "iid", "non-iid", "specialized"
    phishing_types: List[str]  # Types of phishing this bank sees
    is_malicious: bool = False


class ScenarioConfig(BaseModel):
    """Configuration for a demo scenario."""

    name: str
    description: str
    num_banks: int = 5
    num_rounds: int = 20
    privacy_level: int = 2  # 1, 2, or 3
    epsilon: float = 1.0
    delta: float = 1e-5
    banks: List[BankConfig]
    attack_config: dict = None


# Demo Scenario Configurations
SCENARIOS = {
    "happy_path": ScenarioConfig(
        name="Happy Path",
        description="Smooth federated learning convergence with all honest banks",
        num_banks=5,
        num_rounds=15,
        privacy_level=2,
        epsilon=1.0,
        banks=[
            BankConfig(
                bank_id=0,
                name="Chase Bank",
                location="New York, NY",
                lat=40.7128,
                lon=-74.0060,
                data_distribution="iid",
                phishing_types=["account_verify", "urgent"],
            ),
            BankConfig(
                bank_id=1,
                name="Bank of America",
                location="Charlotte, NC",
                lat=35.2271,
                lon=-80.8431,
                data_distribution="iid",
                phishing_types=["invoice", "payment"],
            ),
            BankConfig(
                bank_id=2,
                name="Wells Fargo",
                location="San Francisco, CA",
                lat=37.7749,
                lon=-122.4194,
                data_distribution="iid",
                phishing_types=["gift", "lottery"],
            ),
            BankConfig(
                bank_id=3,
                name="Citibank",
                location="Miami, FL",
                lat=25.7617,
                lon=-80.1918,
                data_distribution="iid",
                phishing_types=["tech_support", "irs"],
            ),
            BankConfig(
                bank_id=4,
                name="US Bank",
                location="Minneapolis, MN",
                lat=44.9778,
                lon=-93.2650,
                data_distribution="iid",
                phishing_types=["shipping", "delivery"],
            ),
        ],
    ),
    "non_iid": ScenarioConfig(
        name="Non-IID Challenge",
        description="Heterogeneous data distribution across banks",
        num_banks=5,
        num_rounds=25,
        privacy_level=2,
        epsilon=1.0,
        banks=[
            BankConfig(
                bank_id=0,
                name="Chase Bank",
                location="New York, NY",
                lat=40.7128,
                lon=-74.0060,
                data_distribution="non-iid",
                phishing_types=["account_verify"],  # 80% specialized
            ),
            BankConfig(
                bank_id=1,
                name="Bank of America",
                location="Charlotte, NC",
                lat=35.2271,
                lon=-80.8431,
                data_distribution="non-iid",
                phishing_types=["urgent"],
            ),
            BankConfig(
                bank_id=2,
                name="Wells Fargo",
                location="San Francisco, CA",
                lat=37.7749,
                lon=-122.4194,
                data_distribution="non-iid",
                phishing_types=["gift"],
            ),
            BankConfig(
                bank_id=3,
                name="Citibank",
                location="Miami, FL",
                lat=25.7617,
                lon=-80.1918,
                data_distribution="non-iid",
                phishing_types=["tech_support"],
            ),
            BankConfig(
                bank_id=4,
                name="US Bank",
                location="Minneapolis, MN",
                lat=44.9778,
                lon=-93.2650,
                data_distribution="non-iid",
                phishing_types=["invoice"],
            ),
        ],
    ),
    "attack_scenario": ScenarioConfig(
        name="Attack Scenario",
        description="Byzantine attack with malicious bank",
        num_banks=5,
        num_rounds=20,
        privacy_level=2,
        epsilon=1.0,
        banks=[
            BankConfig(
                bank_id=0,
                name="Chase Bank",
                location="New York, NY",
                lat=40.7128,
                lon=-74.0060,
                data_distribution="iid",
                phishing_types=["account_verify"],
            ),
            BankConfig(
                bank_id=1,
                name="Bank of America",
                location="Charlotte, NC",
                lat=35.2271,
                lon=-80.8431,
                data_distribution="iid",
                phishing_types=["urgent"],
            ),
            BankConfig(
                bank_id=2,
                name="Wells Fargo",
                location="San Francisco, CA",
                lat=37.7749,
                lon=-122.4194,
                data_distribution="iid",
                phishing_types=["gift"],
                is_malicious=True,  # This bank will attack
            ),
            BankConfig(
                bank_id=3,
                name="Citibank",
                location="Miami, FL",
                lat=25.7617,
                lon=-80.1918,
                data_distribution="iid",
                phishing_types=["tech_support"],
            ),
            BankConfig(
                bank_id=4,
                name="US Bank",
                location="Minneapolis, MN",
                lat=44.9778,
                lon=-93.2650,
                data_distribution="iid",
                phishing_types=["invoice"],
            ),
        ],
        attack_config={
            "attack_bank_id": 2,
            "attack_type": "sign_flip",
            "start_round": 5,
        },
    ),
    "privacy_mode": ScenarioConfig(
        name="Privacy Mode",
        description="Demonstrate HT2ML privacy mechanisms",
        num_banks=3,
        num_rounds=15,
        privacy_level=3,
        epsilon=1.0,
        banks=[
            BankConfig(
                bank_id=0,
                name="Chase Bank",
                location="New York, NY",
                lat=40.7128,
                lon=-74.0060,
                data_distribution="iid",
                phishing_types=["account_verify"],
            ),
            BankConfig(
                bank_id=1,
                name="Bank of America",
                location="Charlotte, NC",
                lat=35.2271,
                lon=-80.8431,
                data_distribution="iid",
                phishing_types=["urgent"],
            ),
            BankConfig(
                bank_id=2,
                name="Wells Fargo",
                location="San Francisco, CA",
                lat=37.7749,
                lon=-122.4194,
                data_distribution="iid",
                phishing_types=["gift"],
            ),
        ],
    ),
}


class AppConfig(BaseModel):
    """Application configuration."""

    host: str = "0.0.0.0"
    port: int = 8001
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    rounds_per_second: float = 0.5  # Demo speed
    max_websocket_connections: int = 100
