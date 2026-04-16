import numpy as np

from data_analizer import ProcessCapabilityAnalizer
from models.result import Results


class TestProcessCapabilityAnalizerReport:
    def test_report_returns_expected_components(self, analyzer):
        fig_hist, fig_xbar, r1, whitin, overall = analyzer.report()

        assert fig_hist is not None
        assert fig_xbar is not None
        assert isinstance(r1, list)
        assert isinstance(whitin, list)
        assert isinstance(overall, list)

    def test_report_returns_two_figures(self, analyzer):
        fig_hist, fig_xbar, *_ = analyzer.report()

        assert hasattr(fig_hist, "axes")
        assert hasattr(fig_xbar, "axes")

    def test_r1_table_structure(self, analyzer):
        _, _, r1, _, _ = analyzer.report()

        assert r1[0] == ["X Peso promedio", "Desviación standar", "Coef. Var.", "Cp", "Cpk"]
        assert len(r1) == 2
        assert len(r1[1]) == 5
        assert all(isinstance(value, (int, float)) for value in r1[1])

    def test_whitin_and_overall_have_four_rows(self, analyzer):
        _, _, _, whitin, overall = analyzer.report()

        assert len(whitin) == 4
        assert len(overall) == 4
        assert all(len(row) == 2 for row in whitin)
        assert all(len(row) == 2 for row in overall)

    def test_metric_values_are_valid_numbers(self, analyzer):
        _, _, r1, whitin, overall = analyzer.report()

        rows = [*r1[1:], *whitin, *overall]
        values = [cell for row in rows for cell in row if isinstance(cell, (int, float))]

        assert all(not np.isnan(value) for value in values)
        assert all(not np.isinf(value) for value in values)

    def test_cp_and_cpk_positive(self, analyzer):
        _, _, r1, _, _ = analyzer.report()

        cp_value = r1[1][3]
        cpk_value = r1[1][4]

        assert cp_value > 0.0
        assert cpk_value > 0.0

    def test_ppm_values_in_range(self, analyzer):
        _, _, _, whitin, overall = analyzer.report()

        ppm_within = whitin[3][1]
        ppm_overall = overall[3][1]

        assert 0 <= ppm_within <= 1_000_000
        assert 0 <= ppm_overall <= 1_000_000


class TestProcessCapabilityAnalizerEdgeCases:
    def test_minimal_dataset_report(self):
        data = np.array([[95.0, 105.0], [98.0, 102.0]])
        analyzer = ProcessCapabilityAnalizer(
            data_subgroups=data,
            usl=110.0,
            lsl=90.0,
            target_mean=100.0,
        )

        fig_hist, fig_xbar, r1, whitin, overall = analyzer.report()

        assert fig_hist is not None
        assert fig_xbar is not None
        assert len(r1) == 2
        assert len(whitin) == 4
        assert len(overall) == 4
        numeric_values = [value for row in [r1[1], *whitin, *overall] for value in row if isinstance(value, (int, float))]
        assert len(numeric_values) > 0
        assert all(not np.isnan(value) and not np.isinf(value) for value in numeric_values)

    def test_large_dataset_report(self):
        np.random.seed(42)
        big_data = np.random.normal(loc=100.0, scale=2.0, size=(100, 10))
        analyzer = ProcessCapabilityAnalizer(
            data_subgroups=big_data,
            usl=110.0,
            lsl=90.0,
            target_mean=100.0,
        )

        fig_hist, fig_xbar, r1, whitin, overall = analyzer.report()

        assert fig_hist is not None
        assert fig_xbar is not None
        assert len(r1) == 2
        assert len(whitin) == 4
        assert len(overall) == 4


class TestProcessCapabilityAnalizerFullAnalysis:
    def test_full_analysis_returns_dictionary(self, analyzer):
        full_results = analyzer.run_full_analysis()

        assert isinstance(full_results, dict)
        assert len(full_results) > 0

    def test_full_analysis_contains_expected_distributions(self, analyzer):
        full_results = analyzer.run_full_analysis()

        expected_distributions = [
            "Normal",
            "Weibull",
            "Lognormal",
            "SEV",
            "LEV",
            "Gamma",
            "Logistic",
            "LogLogistic",
            "Exponential",
        ]
        for dist in expected_distributions:
            assert dist in full_results, f"Distribution {dist} not found in results"

    def test_full_analysis_results_are_results_objects(self, analyzer):
        full_results = analyzer.run_full_analysis()

        for dist_name, result in full_results.items():
            assert isinstance(result, Results), f"{dist_name} result is not a Results object"

    def test_full_analysis_results_have_required_fields(self, analyzer):
        full_results = analyzer.run_full_analysis()

        required_fields = ["cp", "cpk", "pp", "ppk", "title", "usl", "lsl", "pdf_values"]

        for dist_name, result in full_results.items():
            for field in required_fields:
                assert hasattr(result, field), f"{dist_name} missing field {field}"

    def test_full_analysis_normal_distribution_values_valid(self, analyzer):
        full_results = analyzer.run_full_analysis()

        normal_result = full_results["Normal"]

        assert normal_result.title == "Normal"
        assert normal_result.usl == analyzer.usl
        assert normal_result.lsl == analyzer.lsl
        assert normal_result.cp is not None
        assert normal_result.cpk is not None
        assert normal_result.pp is not None
        assert normal_result.ppk is not None
        assert normal_result.cp > 0.0
        assert normal_result.cpk > 0.0
        assert normal_result.pp > 0.0
        assert normal_result.ppk > 0.0

    def test_full_analysis_metric_values_not_nan(self, analyzer):
        full_results = analyzer.run_full_analysis()

        for dist_name, result in full_results.items():
            for field in ["cp", "cpk", "pp", "ppk"]:
                value = getattr(result, field)
                if value is not None:
                    assert not np.isnan(value), f"{dist_name}: {field} is NaN"

    def test_full_analysis_all_distributions_executed_successfully(self, analyzer):
        full_results = analyzer.run_full_analysis()

        for dist_name, result in full_results.items():
            assert result.pdf_values is not None, f"{dist_name} has no pdf_values"
            assert isinstance(result.pdf_values, np.ndarray), f"{dist_name} pdf_values is not numpy array"
