"""
Unit tests for banks module.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from banks import GlobalBank, RegionalBank, DigitalBank, CreditUnion, InvestmentBank
from data import load_bank_data, get_tokenizer


class TestBanks:
    """Test bank implementations."""

    def test_global_bank_initialization(self):
        """Test Global Bank initialization."""
        bank = GlobalBank(data_path='data/bank_datasets')

        assert bank.profile.name == 'Global Bank'
        assert bank.profile.bank_type == 'international'
        assert bank.profile.n_samples == 100000
        assert bank.profile.data_quality == 0.95

    def test_regional_bank_initialization(self):
        """Test Regional Bank initialization."""
        bank = RegionalBank(data_path='data/bank_datasets')

        assert bank.profile.name == 'Regional Bank'
        assert bank.profile.bank_type == 'local'
        assert bank.profile.n_samples == 30000

    def test_digital_bank_initialization(self):
        """Test Digital Bank initialization."""
        bank = DigitalBank(data_path='data/bank_datasets')

        assert bank.profile.name == 'Digital Bank'
        assert bank.profile.bank_type == 'app_first'
        assert bank.profile.n_samples == 50000

    def test_credit_union_initialization(self):
        """Test Credit Union initialization."""
        bank = CreditUnion(data_path='data/bank_datasets')

        assert bank.profile.name == 'Credit Union'
        assert bank.profile.bank_type == 'member_focused'
        assert bank.profile.n_samples == 15000

    def test_investment_bank_initialization(self):
        """Test Investment Bank initialization."""
        bank = InvestmentBank(data_path='data/bank_datasets')

        assert bank.profile.name == 'Investment Bank'
        assert bank.profile.bank_type == 'high_value'
        assert bank.profile.n_samples == 10000

    def test_bank_load_data(self):
        """Test data loading for banks."""
        bank = GlobalBank(data_path='data/bank_datasets')

        dataset = bank.load_data('train')

        assert dataset is not None
        assert len(dataset) > 0

    def test_bank_get_profile(self):
        """Test getting bank profile."""
        bank = RegionalBank(data_path='data/bank_datasets')

        profile = bank.get_profile()

        assert profile.name == 'Regional Bank'
        assert profile.phishing_distribution['spear_phishing'] == 0.40

    def test_bank_get_attack_distribution(self):
        """Test getting attack distribution."""
        bank = DigitalBank(data_path='data/bank_datasets')

        dist = bank.get_attack_distribution()

        assert isinstance(dist, dict)
        assert 'smishing' in dist
        assert dist['smishing'] == 0.35  # Digital bank has high smishing

    def test_bank_get_data_quality(self):
        """Test getting data quality."""
        bank = CreditUnion(data_path='data/bank_datasets')

        quality = bank.get_data_quality()

        assert quality == 0.75  # Credit union has lower quality

    def test_all_banks_created(self):
        """Test that all 5 banks can be created."""
        banks = [
            GlobalBank(data_path='data/bank_datasets'),
            RegionalBank(data_path='data/bank_datasets'),
            DigitalBank(data_path='data/bank_datasets'),
            CreditUnion(data_path='data/bank_datasets'),
            InvestmentBank(data_path='data/bank_datasets')
        ]

        assert len(banks) == 5

        total_samples = sum(b.profile.n_samples for b in banks)
        assert total_samples == 205000  # 100K + 30K + 50K + 15K + 10K


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
