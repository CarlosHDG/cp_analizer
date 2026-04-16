import numpy as np

from data_analizer import ProcessCapabilityAnalizer
from models.result import Results


def test_plot_histogram_creates_axes(sample_data):
    analyzer = ProcessCapabilityAnalizer(
        data_subgroups=sample_data,
        usl=105.0,
        lsl=95.0,
        target_mean=100.0,
    )

    fig, ax = analyzer.plot_histogram("Histograma", np.array([]))

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Histograma"
    assert len(ax.patches) > 0


def test_plot_xbar_chart_creates_xbar(sample_data):
    analyzer = ProcessCapabilityAnalizer(
        data_subgroups=sample_data,
        usl=105.0,
        lsl=95.0,
        target_mean=100.0,
    )

    fig, ax = analyzer.plot_xbar_chart()

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Subgrupo"
    assert ax.get_ylabel() == "Promedio"
    assert any(line.get_label() == "Promedio subgrupo" for line in ax.get_lines())


def test_run_normal_analysis_returns_results_object(analyzer):
    result = analyzer.run_normal_analysis()

    assert isinstance(result, Results)
    assert result.title == "Normal"
    assert result.usl == analyzer.usl
    assert result.lsl == analyzer.lsl
    assert result.cp is not None
    assert result.cpk is not None
    assert result.pp is not None
    assert result.ppk is not None
    assert hasattr(result, "pdf_values")
