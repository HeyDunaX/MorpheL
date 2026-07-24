"""Unit tests for statistical stubs."""

import pytest

from morphel.statistics.bootstrap import paired_bootstrap
from morphel.statistics.correction import holm_correction
from morphel.statistics.mcnemar import mcnemar_test
from morphel.statistics.randomization import randomization_test
from morphel.statistics.reporting import export_significance_report


def test_statistics_stubs() -> None:
    # All are currently stubs and should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        paired_bootstrap([], [], [])

    with pytest.raises(NotImplementedError):
        mcnemar_test([], [], [])

    with pytest.raises(NotImplementedError):
        randomization_test([], [], [], lambda x, y: 0)

    with pytest.raises(NotImplementedError):
        holm_correction([0.01])
