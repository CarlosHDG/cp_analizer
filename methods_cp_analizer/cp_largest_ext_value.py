import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm,gumbel_r
from models.result import Results


def largest_extreme_value_analysis(self):
    title="LARGEST EXTREME VALUE"
    data_flat=self.data_subgroups.flatten()
    params_lev=gumbel_r.fit(data_flat)
    #Anderson Darling and p-value
    percentiles=gumbel_r.cdf(data_flat,*params_lev)
    z_scores=norm.ppf(percentiles)
    ad_statistic,p_value=smsd.normal_ad(z_scores)
    #Pp y Ppk with Z-score method
    z_score_usl=norm.ppf(gumbel_r.cdf(self.usl,*params_lev))
    z_score_lsl=norm.ppf(gumbel_r.cdf(self.lsl,*params_lev))
    #plot
    pdf_values=gumbel_r.pdf(self.x,*params_lev)
    pp=(z_score_usl-z_score_lsl)/6
    ppk=min(-z_score_lsl/3,z_score_usl/3)
    return Results(data=self.data_subgroups,
                    usl=self.usl,
                    lsl=self.lsl,
                    title=title,
                    ad=ad_statistic,
                    p_value=p_value,
                    params=params_lev,
                    cp=None,
                    cpk=None,
                    pp=pp,
                    ppk=ppk,
                    pdf_values=pdf_values)