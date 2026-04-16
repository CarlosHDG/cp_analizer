import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm,expon
from models.result import Results

def two_parameter_exponential_analysis(self):
    title="2 PARAMETROS EXPONENCIAL"
    data_flat=self.data_subgroups.flatten()
    two_params_expon=expon.fit(data_flat)
    #Anderson Darling and p-value
    percentiles=expon.cdf(data_flat,*two_params_expon)
    z_scores=norm.ppf(percentiles)
    ad_statistic,p_value=smsd.normal_ad(z_scores)
    #Pp y Ppk with Z-score method
    z_score_usl=norm.ppf(expon.cdf(self.usl,*two_params_expon))
    z_score_lsl=norm.ppf(expon.cdf(self.lsl,*two_params_expon))
    #plot
    pdf_values=expon.pdf(self.x,*two_params_expon)
    pp=(z_score_usl-z_score_lsl)/6
    ppk=min(-z_score_lsl/3,z_score_usl/3)
    return Results(data=self.data_subgroups,
                    usl=self.usl,
                    lsl=self.lsl,
                    title=title,
                    ad=ad_statistic,
                    p_value=p_value,
                    params=two_params_expon,
                    cp=None,
                    cpk=None,
                    pp=pp,
                    ppk=ppk,
                    pdf_values=pdf_values)