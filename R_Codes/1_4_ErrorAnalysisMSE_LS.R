rm(list = ls())
Essential=c("agricolae","psych","rlist","rstatix","lme4","car","usefun","tidyverse","PerformanceAnalytics",
            "corrplot","GGally", "ggplot2")
Helpful=c("gridExtra","gmodels","lmerTest","multcomp","emmeans","multcompView","nlme","moments","drc")
library(easypackages)
libraries(Essential) # Essential Packages
libraries(Helpful) # Helpful Packages
#source("All_Functions.r") # Installs all useful functions

#Calculating the RMSE - Imputer
df1<- read.csv("G:/My Drive/Leticia_Santos/Soybeans/Results/Results_paper/RawResultsMSE_008ImpSam.csv") 
df2=df1%>%dplyr::select(-X)%>%arrange(desc(MSE))%>%slice_tail(n=0.95*nrow(df1))

stats <- df2 %>% 
  group_by(Method, Size) %>% 
  dplyr::summarise(Mean = mean(MSE, na.rm=TRUE),
                   n = n(),
                   Max = max(MSE, na.rm = TRUE),
                   Min = min(MSE, na.rm=TRUE),
                   SE.high = Mean+(sd(MSE, na.rm=TRUE)/sqrt(n)),
                   SE.low = Mean-(sd(MSE, na.rm=TRUE)/sqrt(n)))

p1=ggplot(stats,aes(x=Size,y=Mean,colour=Method))+
  geom_point()+geom_line()+
  scale_colour_manual(values = c("#0072B2","#E69F00"))+theme_classic()+
  geom_errorbar(aes(ymin=SE.low, ymax=SE.high),width=.2,position=position_dodge(0.05))+
  geom_ribbon(aes(ymin=SE.low, ymax=SE.high), linetype=2, alpha=0.1)+
  ylab("Mean Square Error")+xlab("Sample Size")+
  scale_x_continuous(labels=seq(20,120,5),breaks=seq(20,120,5))+
  #scale_color_discrete(name="Model",labels=c("Linear Regression","Random Forest"))+
  #ggtitle("Average MSE with SE Bands By Sample Size For The Two Models")+
  theme(text = element_text(size = 32),
        legend.position = c(0.2, 0.9))
p1

ggsave(p1, width = 12, height = 8, dpi = 300, 
       file="G:/My Drive/Leticia_Santos/Soybeans/Results/Results_paper/ErrorAnalysisMSE_008ImpSam.png")


#Calculating the MSE - Copula
df3<- read.csv("G:/My Drive/Leticia_Santos/Soybeans/Results/Results_paper/RawResultsMSE_006CopSam.csv") 
df4=df3%>%dplyr::select(-X)%>%arrange(desc(MSE))%>%slice_tail(n=0.95*nrow(df3))


stats2 <- df4 %>% 
  group_by(Method, Size) %>% 
  dplyr::summarise(Mean = mean(MSE, na.rm=TRUE),
                   n = n(),
                   Max = max(MSE, na.rm = TRUE),
                   Min = min(MSE, na.rm=TRUE),
                   SE.high = Mean+(sd(MSE, na.rm=TRUE)/sqrt(n)),
                   SE.low = Mean-(sd(MSE, na.rm=TRUE)/sqrt(n)))

p2=ggplot(stats2,aes(x=Size,y=Mean,colour=Method))+
  geom_point()+geom_line()+
  scale_colour_manual(values = c("#0072B2","#E69F00"))+theme_classic()+
  geom_errorbar(aes(ymin=SE.low, ymax=SE.high),width=.2,position=position_dodge(0.05))+
  geom_ribbon(aes(ymin=SE.low, ymax=SE.high), linetype=2, alpha=0.1)+
  ylab("Mean Square Error")+xlab("Sample Size")+
  scale_x_continuous(labels=seq(20,120,5),breaks=seq(20,120,5))+
  #scale_color_discrete(name="Model",labels=c("Linear Regression","Random Forest"))+
  #ggtitle("Average MSE with SE Bands By Sample Size For The Two Models")+
  theme(text = element_text(size = 32),
        legend.position = c(0.2, 0.9))
p2

ggsave(p2, width = 12, height = 8, dpi = 300, 
       file="G:/My Drive/Leticia_Santos/Soybeans/Results/Results_paper/ErrorAnalysisMSE_006CopSam.png")
