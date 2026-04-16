import numpy as np
import matplotlib.pyplot as plt
from models.result import Results
from typing import Dict
from scipy.stats import norm
from methods_cp_analizer.cp_normal import normal_analysis
from methods_cp_analizer.cp_weibull import weibull_analysis
from methods_cp_analizer.cp_lognormal import lognormal_analysis
from methods_cp_analizer.cp_smallest_ext_value import smallest_extreme_value_analysis
from methods_cp_analizer.cp_largest_ext_value import largest_extreme_value_analysis
from methods_cp_analizer.cp_gamma import gamma_analysis
from methods_cp_analizer.cp_logistic import logistic_anaysis
from methods_cp_analizer.cp_loglogistic import loglogistic_anaysis
from methods_cp_analizer.cp_exponential import exponential_analysis
from methods_cp_analizer.cp_three_param_weibull import three_parameter_weibull_analysis
from methods_cp_analizer.cp_three_param_lognormal import three_parameter_lognormal_analysis
from methods_cp_analizer.cp_three_param_gamma import three_parameter_gamma_analysis
from methods_cp_analizer.cp_three_param_loglogistic import three_parameter_loglogistic_analysis
from methods_cp_analizer.cp_two_param_expon import two_parameter_exponential_analysis
from methods_cp_analizer.cp_boxcox import boxcox_transformation_analysis
from methods_cp_analizer.cp_jonhson import jonhson_transformation_analysis
from methods_cp_analizer.cp_nonparametric import non_parametric_anaysis




class ProcessCapabilityAnalizer():
    def __init__(self,data_subgroups : np.ndarray,usl :float,lsl:float,target_mean:float):
        self.data_subgroups=data_subgroups
        self.usl : float =usl
        self.lsl : float =lsl
        self.target_mean=target_mean
        self.data_flat=self.data_subgroups.flatten()
        self.__p_low=0.00135
        self.__p_high=0.99865
        self.x=np.linspace(np.min(self.data_subgroups.flatten())-3*np.std(self.data_subgroups.flatten()),np.max(self.data_subgroups.flatten())+3*np.std(self.data_subgroups.flatten()),200)
        self.method=[
                "Normal",
                "Weibull",
                "Lognormal",
                "SEV",
                "LEV",
                "Gamma",
                "Logistic",
                "LogLogistic",
                "Exponential",
                "Weibull3p",
                "Lognormal3p",
                "Gamma3p",
                "LogLogistic3p",
                "Exponential2p",
                "Box-Cox",
                "Johnson",
                "NonParametric"]
        
    def plot_histogram(self,title:str,pdf_values,data_flat=None):
        if data_flat is None:
            data_flat=self.data_flat
            usl=self.usl
            lsl=self.lsl
            x=self.x
        else:
            usl=np.max(data_flat)
            lsl=np.min(data_flat)
            x=np.linspace(np.min(data_flat)-3*np.std(data_flat),np.max(data_flat)+3*np.std(data_flat),200)

        q1,q3=np.percentile(data_flat,[25,75])
        n=len(data_flat)
        bin_width=2*(q3-q1)/(n**(1/3))
        bins=int(max(5,(np.max(data_flat)-np.min(data_flat))/bin_width))
        fig,ax=plt.subplots(figsize=(10,6))
        hist_values,bins_edges,_=ax.hist(data_flat,bins=bins,density=True,alpha=0.6)
        ax.axvline(lsl,color="red",linestyle="--",label="LSL")
        ax.axvline(usl,color="red",linestyle="--",label="USL")
        if len(pdf_values)!=0:
            pdf_scaled=pdf_values*max(hist_values)/max(pdf_values)
            ax.plot(x,pdf_scaled,linewidth=2)
        ax.set_title(title)
        ax.legend()
        return fig, ax
    def plot_xbar_chart(self):
        data=self.data_subgroups
        avg_subgroup=data.mean(axis=1)
        avg_within=avg_subgroup.mean()
        std_subgroup : np.ndarray=data.std(axis=1,ddof=1)
        std_within=((std_subgroup**2).mean())**0.5
        x=np.arange(1,len(avg_subgroup)+1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x,avg_subgroup,marker="o",color="blue",label="Promedio subgrupo")
        ax.axhline(avg_within+3*(std_within/(data.shape[1])**(1/2)), color='red', linestyle='--', linewidth=2, label='USL')
        ax.axhline(avg_within-3*(std_within/(data.shape[1])**(1/2)), color='red', linestyle='--', linewidth=2, label='LSL')
        ax.axhline(avg_within, color='green', linestyle='--', linewidth=2, label='LSL')
        ax.set_title("Gráfico X")
        ax.set_xlabel("Subgrupo")
        ax.set_ylabel("Promedio")
        ax.legend()
        return fig, ax
        
    def run_normal_analysis(self):
        return normal_analysis(self)
    def run_weibull_analysis(self):
        return weibull_analysis(self)
    def run_lognormal_analysis(self):
        return lognormal_analysis(self)
    def run_smallest_extreme_value_analysis(self):
        return smallest_extreme_value_analysis(self)
    def run_largest_extreme_value_analysis(self):
        return largest_extreme_value_analysis(self)
    def run_gamma_analysis(self):
        return gamma_analysis(self)
    def run_logistic_anaysis(self):
        return logistic_anaysis(self)
    def run_loglogistic_anaysis(self):
        return loglogistic_anaysis(self)
    def run_exponential_analysis(self):
        return exponential_analysis(self)
    def run_three_parameter_weibull_analysis(self):
        return three_parameter_weibull_analysis(self)
    def run_three_parameter_lognormal_analysis(self):
        return three_parameter_lognormal_analysis(self)
    def run_three_parameter_gamma_analysis(self):
        return three_parameter_gamma_analysis(self)
    def run_three_parameter_loglogistic_analysis(self):
        return three_parameter_loglogistic_analysis(self)
    def run_two_parameter_exponential_analysis(self):
        return two_parameter_exponential_analysis(self)
    def run_boxcox_transformation_analysis(self):
        return boxcox_transformation_analysis(self)
    def run_jonhson_transformation_analysis(self):
        return jonhson_transformation_analysis(self)
    def run_non_parametric_anaysis(self):
        return non_parametric_anaysis(self)
    def run_full_analysis(self) -> Dict[str,Results]:
        full_results={}
        analysis=[
                ("Normal", self.run_normal_analysis),
                ("Weibull", self.run_weibull_analysis),
                ("Lognormal", self.run_lognormal_analysis),
                ("SEV", self.run_smallest_extreme_value_analysis),
                ("LEV", self.run_largest_extreme_value_analysis),
                ("Gamma", self.run_gamma_analysis),
                ("Logistic", self.run_logistic_anaysis),
                ("LogLogistic", self.run_loglogistic_anaysis),
                ("Exponential", self.run_exponential_analysis),
                ("Weibull3p", self.run_three_parameter_weibull_analysis),
                ("Lognormal3p", self.run_three_parameter_lognormal_analysis),
                ("Gamma3p", self.run_three_parameter_gamma_analysis),
                ("LogLogistic3p", self.run_three_parameter_loglogistic_analysis),
                ("Exponential2p", self.run_two_parameter_exponential_analysis),
                ("Box-Cox", self.run_boxcox_transformation_analysis),
                ("Johnson", self.run_jonhson_transformation_analysis),
                ("NonParametric", self.run_non_parametric_anaysis)]
        for name,func in analysis:
            try:
                result=func()
                full_results[name]=result
            except Exception as e:
                print(f"Analisis {name}, error {e}") #Implementar Logs
        return full_results
    def report(self):
        full_results=self.run_full_analysis()
        fig_hist, ax=self.plot_histogram(full_results["Normal"].title,full_results["Normal"].pdf_values)
        fig_xbar, ax=self.plot_xbar_chart()
        data=self.data_subgroups
        avg_subgroup=data.mean(axis=1)
        avg_within=round(avg_subgroup.mean(),2)
        std_subgroup : np.ndarray=data.std(axis=1,ddof=1)
        std_within=round(((std_subgroup**2).mean())**0.5,2)
        overall_avg=data.mean()
        coef_var=round((std_within/overall_avg)*100,2)
        cp=round(float(full_results["Normal"].cp),2)
        cpk=round(float(full_results["Normal"].cpk),2)
        overall_std=round(data.std(ddof=1),2)
        ppm_whitin=round(float((norm.cdf(self.lsl,avg_within,std_within)*1000000)+((1-norm.cdf(self.usl,avg_within,std_within))*1000000)),2)
        pp=round(float(full_results["Normal"].pp),2)
        ppk=round(float(full_results["Normal"].ppk),2)
        r1=[
            ["X Peso promedio","Desviación standar","Coef. Var.","Cp","Cpk"],
            [avg_within,std_within,coef_var,cp,cpk]
        ]
        ppm_overall=round(float((norm.cdf(self.lsl,overall_avg,overall_std)*1000000)+((1-norm.cdf(self.usl,overall_avg,overall_std))*1000000)),2)
        whitin=[
            ["Desv. Std",std_within],
            ["Cp",cp],
            ["Cpk",cpk],
            ["PPM",ppm_whitin]
        ]

        overall=[
            ["Desv. Std",overall_std],
            ["Pp",pp],
            ["Ppk",ppk],
            ["PPM",ppm_overall]
        ]

        return fig_hist,fig_xbar,r1,whitin,overall