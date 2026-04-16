import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm,johnsonsu
from data_analizer import normal_analysis
from models.result import Results

def jonhson_transformation_analysis(self):
    title="TRASNFORMACIÓN JONHSON"
    data_flat=self.data_subgroups.flatten()
    params_jonhson=johnsonsu.fit(data_flat)
    #Anderson Darling and p-value
    percentiles=johnsonsu.cdf(data_flat,*params_jonhson)
    z_scores=norm.ppf(percentiles)
    usl_johnson=norm.ppf(johnsonsu.cdf(self.usl,*params_jonhson))
    lsl_johnson=norm.ppf(johnsonsu.cdf(self.lsl,*params_jonhson))
    lsl_johnson=johnsonsu.cdf(self.lsl,*params_jonhson)
    ad_statistic,p_value=smsd.normal_ad(z_scores)
    data_johnson=z_scores
    sub_group_size=self.data_subgroups.shape[1]
    data_jonhson_transformed=data_johnson.reshape(-1,sub_group_size)
    normal=normal_analysis(self,data_jonhson_transformed,usl_johnson,lsl_johnson)
    return Results(data=data_jonhson_transformed,
                    usl=usl_johnson,
                    lsl=lsl_johnson,
                    title=title,
                    ad=normal.ad,
                    p_value=normal.p_value,
                    params=params_jonhson,
                    cp=normal.cp,
                    cpk=normal.cpk,
                    pp=normal.pp,
                    ppk=normal.ppk,
                    pdf_values=normal.pdf_values)