import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm,logistic
from models.result import Results

def logistic_anaysis(self):
    title="LOGISTIC"
    data_flat=self.data_subgroups.flatten()
    params_logistic=logistic.fit(data_flat)
    #Anderson Darling and p-value
    percentiles=logistic.cdf(data_flat,*params_logistic)
    z_scores=norm.ppf(percentiles)
    ad_statistic,p_value=smsd.normal_ad(z_scores)
    #Pp y Ppk with Z-score method
    z_score_usl=norm.ppf(logistic.cdf(self.usl,*params_logistic))
    z_score_lsl=norm.ppf(logistic.cdf(self.lsl,*params_logistic))
    #plot
    pdf_values=logistic.pdf(self.x,*params_logistic)
    pp=(z_score_usl-z_score_lsl)/6
    ppk=min(-z_score_lsl/3,z_score_usl/3)
    return Results(data=self.data_subgroups,
                    usl=self.usl,
                    lsl=self.lsl,
                    title=title,
                    ad=ad_statistic,
                    p_value=p_value,
                    params=params_logistic,
                    cp=None,
                    cpk=None,
                    pp=pp,
                    ppk=ppk,
                    pdf_values=pdf_values)