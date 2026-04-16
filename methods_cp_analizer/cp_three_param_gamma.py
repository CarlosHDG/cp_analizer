import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm,gamma
from models.result import Results


def three_parameter_gamma_analysis(self):
    title="3 PARAMETROS GAMMA"
    data_flat=self.data_subgroups.flatten()
    params_three_p_gamma=gamma.fit(data_flat)
    #Anderson Darling and p-value
    percentiles=gamma.cdf(data_flat,*params_three_p_gamma)
    z_scores=norm.ppf(percentiles)
    ad_statistic,p_value=smsd.normal_ad(z_scores)
    #Pp y Ppk with Z-score method
    z_score_usl=norm.ppf(gamma.cdf(self.usl,*params_three_p_gamma))
    z_score_lsl=norm.ppf(gamma.cdf(self.lsl,*params_three_p_gamma))
    #plot
    pdf_values=gamma.pdf(self.x,*params_three_p_gamma)
    # self.plot_histogram(title,pdf_values)
    # plt.show()
    pp=(z_score_usl-z_score_lsl)/6
    ppk=min(-z_score_lsl/3,z_score_usl/3)
    return Results(data=self.data_subgroups,
                    usl=self.usl,
                    lsl=self.lsl,
                    title=title,
                    ad=ad_statistic,
                    p_value=p_value,
                    params=params_three_p_gamma,
                    cp=None,
                    cpk=None,
                    pp=pp,
                    ppk=ppk,
                    pdf_values=pdf_values)