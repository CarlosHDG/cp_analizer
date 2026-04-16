import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm,gumbel_l
from models.result import Results

def smallest_extreme_value_analysis(self):
    title="SMALLEST EXTREME VALUE"
    data_flat=self.data_subgroups.flatten()
    params_sev=gumbel_l.fit(data_flat)
    #Anderson Darling and p-value
    percentiles=gumbel_l.cdf(data_flat,*params_sev)
    z_scores=norm.ppf(percentiles)
    ad_statistic,p_value=smsd.normal_ad(z_scores)
    #Pp y Ppk with Z-score method
    z_score_usl=norm.ppf(gumbel_l.cdf(self.usl,*params_sev))
    z_score_lsl=norm.ppf(gumbel_l.cdf(self.lsl,*params_sev))
    #plot
    pdf_values=gumbel_l.pdf(self.x,*params_sev)
    pp=(z_score_usl-z_score_lsl)/6
    ppk=min(-z_score_lsl/3,z_score_usl/3)
    return Results(data=self.data_subgroups,
            usl=self.usl,
            lsl=self.lsl,
            title=title,
            ad=ad_statistic,
            p_value=p_value,
            params=params_sev,
            cp=None,
            cpk=None,
            pp=pp,
            ppk=ppk,
            pdf_values=pdf_values)