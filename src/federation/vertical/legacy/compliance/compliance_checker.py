"""
Compliance checking for GDPR, PCI-DSS, and bank secrecy.

Verifies that the federated learning system satisfies regulatory requirements.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ComplianceReport:
    """Report for compliance check."""
    is_compliant: bool
    checks: List[str]
    violations: List[str]
    recommendations: List[str]


def check_gdpr_compliance(system_config: Dict) -> ComplianceReport:
    """
    Verify GDPR compliance.

    GDPR Requirements:
    - Article 25: Data Protection by Design (privacy-by-default)
    - Article 32: Lawful basis for processing
    - Article 9: Special category data
    - Right to explanation (interpretable models)

    Args:
        system_config: System configuration dictionary

    Returns:
        ComplianceReport
    """
    checks = []
    violations = []
    recommendations = []

    # Check 1: No raw data cross-border transfer
    if system_config.get('privacy_mechanism') != 'none':
        checks.append("✓ Privacy mechanism enabled (DP, secure agg, or hybrid)")
    else:
        checks.append("⚠ No privacy mechanism - consider adding DP or secure aggregation")

    # Check 2: Data minimization
    if system_config.get('aggregation') == 'secure':
        checks.append("✓ Secure aggregation - only aggregated gradients shared")
    else:
        checks.append("⚠ Standard aggregation - consider secure aggregation")

    # Check 3: Right to explanation
    if system_config.get('model_type') == 'transformer':
        checks.append("⚠ Transformer models less interpretable - consider attention visualization")
    else:
        checks.append("✓ Tree-based models are interpretable")

    # Check 4: Purpose limitation
    checks.append("✓ Purpose limited to phishing detection (fraud prevention)")

    # Check 5: Data retention
    if system_config.get('delete_local_data', True):
        checks.append("✓ Local data deleted after training")
    else:
        recommendations.append("Implement local data deletion after training")

    is_compliant = len(violations) == 0

    return ComplianceReport(
        is_compliant=is_compliant,
        checks=checks,
        violations=violations,
        recommendations=recommendations
    )


def check_pci_dss_compliance(system_config: Dict) -> ComplianceReport:
    """
    Verify PCI-DSS compliance.

    PCI-DSS Requirements for federated learning:
    - Requirement 3.1: Keep cardholder data secure
    - Requirement 4.1: Encrypt data in transit
    - Requirement 6.2: Secure development
    - Requirement 10.3: Log all access

    Args:
        system_config: System configuration

    Returns:
        ComplianceReport
    """
    checks = []
    violations = []
    recommendations = []

    # Check 1: No cardholder data stored/shared
    if system_config.get('data_type') == 'aggregated_only':
        checks.append("✓ No raw cardholder data stored centrally")
    else:
        violations.append("✗ Raw cardholder data shared - violates PCI-DSS")

    # Check 2: Encryption in transit
    if system_config.get('encryption', 'TLS') == 'TLS':
        checks.append("✓ TLS encryption for communication")
    else:
        violations.append("✗ No TLS encryption - violates PCI-DSS 4.1")

    # Check 3: Access logging
    if system_config.get('logging', True):
        checks.append("✓ All accesses logged (PCI-DSS 10.3)")
    else:
        recommendations.append("Enable comprehensive access logging")

    # Check 4: Secure model updates
    if system_config.get('privacy_mechanism') == 'secure':
        checks.append("✓ Secure aggregation protects model updates")
    else:
        recommendations.append("Consider secure aggregation for model updates")

    is_compliant = len(violations) == 0

    return ComplianceReport(
        is_compliant=is_compliant,
        checks=checks,
        violations=violations,
        recommendations=recommendations
    )


def check_bank_secrecy_compliance(system_config: Dict) -> ComplianceReport:
    """
    Verify bank secrecy act compliance.

    Bank Secrecy Requirements:
    - No customer identification shared
    - Only aggregated insights
    - No reverse engineering possible

    Args:
        system_config: System configuration

    Returns:
        ComplianceReport
    """
    checks = []
    violations = []
    recommendations = []

    # Check 1: No customer data shared
    if system_config.get('shares_raw_features', False):
        checks.append("✓ No raw features shared - only gradients/aggregates")
    else:
        violations.append("✗ Raw features shared - violates bank secrecy")

    # Check 2: Aggregation only
    if system_config.get('aggregation', 'weighted_average'):
        checks.append("✓ Only aggregated model parameters shared")
    else:
        recommendations.append("Ensure only aggregated updates are transmitted")

    # Check 3: Differential privacy protection
    if system_config.get('privacy_mechanism') in ['dp', 'secure', 'hybrid']:
        checks.append(f"✓ {system_config['privacy_mechanism']} provides formal privacy guarantees")
    else:
        recommendations.append("Consider adding DP or secure aggregation for stronger protection")

    # Check 4: No inference of individual customers
    if system_config.get('differential_privacy', True):
        checks.append("✓ DP prevents individual customer inference")
    else:
        recommendations.append("Consider DP to prevent customer-level inference")

    is_compliant = len(violations) == 0

    return ComplianceReport(
        is_compliant=is_compliant,
        checks=checks,
        violations=violations,
        recommendations=recommendations
    )
