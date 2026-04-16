import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm,weibull_min
from models.result import Results

def three_parameter_weibull_analysis(self):
    title="3-PARAMETROS WEIBULL"
    data_flat=self.data_subgroups.flatten()
    params_three_p_weibull=weibull_min.fit(data_flat)
    #Anderson Darling and p-value
    percentiles=weibull_min.cdf(self.data_flat,*params_three_p_weibull)
    z_scores=norm.ppf(percentiles)
    ad_statistic,p_value=smsd.normal_ad(z_scores)
    #Pp y Ppk with Z-score method
    z_score_usl=norm.ppf(weibull_min.cdf(self.usl,*params_three_p_weibull))
    z_score_lsl=norm.ppf(weibull_min.cdf(self.lsl,*params_three_p_weibull))
    #plot
    pdf_values=weibull_min.pdf(self.x,*params_three_p_weibull)
    pp=(z_score_usl-z_score_lsl)/6
    ppk=min(-z_score_lsl/3,z_score_usl/3)
    return Results(data=self.data_subgroups,
            usl=self.usl,
            lsl=self.lsl,
            title=title,
            ad=ad_statistic,
            p_value=p_value,
            params=params_three_p_weibull,
            cp=None,
            cpk=None,
            pp=pp,
            ppk=ppk,
            pdf_values=pdf_values)