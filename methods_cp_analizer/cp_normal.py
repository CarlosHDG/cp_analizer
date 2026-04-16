import statsmodels.stats.diagnostic as smsd
from scipy.stats import norm
import numpy as np
from models.result import Results

def normal_analysis(self,data=None,usl:float|None=None,lsl:float|None=None):
        title:str="Normal"
        if data is None:
            data=self.data_subgroups
        if usl is None and lsl is None:
            usl=self.usl
            lsl=self.lsl
            x=self.x
        else:
            x=np.linspace(np.min(data.flatten())-3*np.std(data.flatten()),np.max(data.flatten())+3*np.std(data.flatten()),200)
        avg_subgroup=data.mean(axis=1)
        avg_within=avg_subgroup.mean()
        rangos=data.max(axis=1)-data.min(axis=1)
        std_subgroup : np.ndarray=data.std(axis=1,ddof=1)
        overall_avg=data.mean()
        overall_std=data.std(ddof=1)
        std_within=((std_subgroup**2).mean())**0.5
        coef_var=(std_within/overall_avg)*100
        cp=(usl-lsl)/(6*std_within)
        cpk=min(((usl-avg_within)/(3*std_within)),((avg_within-lsl)/(3*std_within)))
        pp=(usl-lsl)/(6*overall_std)
        ppk=min(((usl-overall_avg)/(3*overall_std)),((overall_avg-lsl)/(3*overall_std)))
        ad_statistic,p_value=smsd.normal_ad(data.flatten())
        #plot
        pdf_values=norm.pdf(x,overall_avg,overall_std)
        return Results(data=data,
                    usl=usl,
                    lsl=lsl,
                    title=title,
                    ad=ad_statistic,
                    p_value=p_value,
                    params=(),
                    cp=cp,
                    cpk=cpk,
                    pp=pp,
                    ppk=ppk,
                    pdf_values=pdf_values)