import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm,fisk
from models.result import Results

def three_parameter_loglogistic_analysis(self):
    title="3 PARAMETROS LOGLOGISTIC"
    data_flat=self.data_subgroups.flatten()
    params_loglogistic=fisk.fit(data_flat)
    #Anderson Darling and p-value
    percentiles=fisk.cdf(data_flat,*params_loglogistic)
    z_scores=norm.ppf(percentiles)
    ad_statistic,p_value=smsd.normal_ad(z_scores)
    #Pp y Ppk with Z-score method
    z_score_usl=norm.ppf(fisk.cdf(self.usl,*params_loglogistic))
    z_score_lsl=norm.ppf(fisk.cdf(self.lsl,*params_loglogistic))
    #plot
    pdf_values=fisk.pdf(self.x,*params_loglogistic)
    pp=(z_score_usl-z_score_lsl)/6
    ppk=min(-z_score_lsl/3,z_score_usl/3)
    return Results(data=self.data_subgroups,
                    usl=self.usl,
                    lsl=self.lsl,
                    title=title,
                    ad=ad_statistic,
                    p_value=p_value,
                    params=params_loglogistic,
                    cp=None,
                    cpk=None,
                    pp=pp,
                    ppk=ppk,
                    pdf_values=pdf_values)