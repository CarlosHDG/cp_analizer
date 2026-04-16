import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm,lognorm
import numpy as np
from models.result import Results

def lognormal_analysis(self):
    title="LOGNORMAL"
    data_flat=self.data_subgroups.flatten()
    params_lognormal=lognorm.fit(data_flat,floc=0)
    #Anderson Darling and p-value
    percentiles=lognorm.cdf(data_flat,*params_lognormal)
    z_scores=norm.ppf(percentiles)
    ad_statistic,p_value=smsd.normal_ad(z_scores)
    #Pp y Ppk with Z-score method
    z_score_usl=norm.ppf(lognorm.cdf(self.usl,*params_lognormal))
    z_score_lsl=norm.ppf(lognorm.cdf(self.lsl,*params_lognormal))
    #plot
    pdf_values=lognorm.pdf(np.log(self.x),*params_lognormal)
    pp=(z_score_usl-z_score_lsl)/6
    ppk=min(-z_score_lsl/3,z_score_usl/3)
    return Results(data=self.data_subgroups,
                    usl=self.usl,
                    lsl=self.lsl,
                    title=title,
                    ad=ad_statistic,
                    p_value=p_value,
                    params=params_lognormal,
                    cp=None,
                    cpk=None,
                    pp=pp,
                    ppk=ppk,
                    pdf_values=pdf_values)