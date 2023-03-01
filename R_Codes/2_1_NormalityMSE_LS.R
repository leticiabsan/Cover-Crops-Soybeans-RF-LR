rm(list = ls())
Essential=c("agricolae","psych","rlist","rstatix","lme4","car","usefun","tidyverse","PerformanceAnalytics",
            "corrplot","GGally", "ggplot2","dplyr")
Helpful=c("gridExtra","gmodels","lmerTest","multcomp","emmeans","multcompView","nlme","moments","drc")
library(easypackages)
libraries(Essential) # Essential Packages
libraries(Helpful) # Helpful Packages
source("All_Functions.r") # Installs all useful functions

#MSE - Copula Sample
#Linear Regression
df1 <- read.csv("../../Results/Results_paper/RawResultsMSE_006CopSam.csv")
df2=df1 %>%   
  dplyr::select(-X) %>%
  arrange(desc(MSE)) %>%
  dplyr::filter(Method=="Linear Regression") %>% 
    slice_tail(n=0.49*nrow(df1))

hist(df2$MSE)
shapiro.test(df2$MSE)

#Random Forest
df3<- read.csv("../../Results/Results_paper/RawResultsMSE_006CopSam.csv") %>%
  dplyr::select(-X) %>%
  filter(Method=="Random Forest") %>% 
  filter(Size=="80")

hist(df3$MSE)
shapiro.test(df3$MSE)

#MSE - Imputer Sample 
#Linear Regression
df4<- read.csv("../../Results/Results_paper/RawResultsMSE_008ImpSam.csv") %>%
  dplyr::select(-X) %>%
  filter(Method=="Linear Regression")

hist(df4$MSE)
shapiro.test(df3$MSE)

#Random Forest
df5<- read.csv("../../Results/Results_paper/RawResultsMSE_008ImpSam.csv") %>%
  dplyr::select(-X) %>%
  filter(Method=="Random Forest")

hist(df5$MSE)
shapiro.test(df5$MSE)
