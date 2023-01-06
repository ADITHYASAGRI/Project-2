A=read.csv("ICU.csv");head(A)
attach(A)
cancer=ifelse(A$case=='cancer',1,0)                      
kidney_injury=ifelse(A$case=='kidney injury',1,0)
stroke=ifelse(A$case=="stroke",1,0)
heart_failure=ifelse(A$case=="heart failure",1,0)
hepatic_failure=ifelse(A$case=="hepatic failure",1,0)
respiratory=ifelse(A$case=="respiratory",1,0)
brain_injury=ifelse(A$case=="brain injury",1,0)
Vital_Status=ifelse(A$vital.status=='survival',1,0)
Sex=ifelse(A$sex=="M",1,0)
x=cbind(A,Sex,cancer,kidney_injury,stroke,heart_failure,hepatic_failure,respiratory,brain_injury,Vital_Status)
write.csv(x,"Data.csv")

library(xgboost)  # the main algorithm
library(archdata) # for the sample dataset
library(caret)    # for the confusionmatrix() function (also needs e1071 package)
library(dplyr)  
library(adabag)
library(caret)
library(e1071)
library(ipred)
library(randomForest)



X=read.csv("Data.csv");head(X)


library(caTools)
sp=sample.split(X,SplitRatio = 0.75)
tr=subset(X,sp == TRUE);length(tr$age)
trx=tr[,-16]
try=tr[,16]
ts=subset(X,sp==FALSE);head(ts)
length(ts$age)
tsx=ts[,-16];head(tsx)
tsy=ts[,16];head(tsy)

beta=rbind(-0.01,-0.02,0.06,0.06,-0.09,-0.18);beta
aa=cbind(tsx$age,tsx$CO2,tsx$HCO3,tsx$LOC,tsx$cancer,tsx$stroke);aa
q=exp(aa%*%beta)/(1+exp(aa%*%beta));q
qq=ifelse(q>0.67,1,0);qq
st=table(tsy,qq);st
(accuracy=sum(diag(st))/sum(st))
(recall=diag(st)/rowSums(st))
(precision=diag(st)/colSums(st))
(F=(2*precision*recall)/(precision+recall))

beta=rbind(-0.042,-0.015,0.0087,-0.27,-0.25,0.13,2.53);beta
aa=cbind(tsx$pH,tsx$age,tsx$HR,tsx$stroke,tsx$heart_failure,tsx$respiratory,tsx$hepatic_failure);aa
q=exp(aa%*%beta)/(1+exp(aa%*%beta));q
qq=ifelse(q>0.39,1,0);qq
st=table(tsy,qq);st
(accuracy=sum(diag(st))/sum(st))
(recall=diag(st)/rowSums(st))
(precision=diag(st)/colSums(st))
(F=(2*precision*recall)/(precision+recall))




g1=glm(Vital_Status~.,data=tr,family=binomial())
summary(g1)
prob=predict(g1,tsx,type = "response")
prob=as.data.frame(prob);prob
prob=round(prob,2);prob
#write.csv(prob,'prob.csv')
p=ifelse(prob>0.5,1,0)

f=glm(Vital_Status~1,data=X)
f1=glm(Vital_Status~age+HR+pH+LOC+cancer+stroke+heart_failure+hepatic_failure+respiratory,data=X)
library(lmtest)
lrtest(f,f1)




t=table(tsy,p);t
(accuracy=sum(diag(t))/sum(t))
(recall=diag(t)/rowSums(t))
(precision=diag(t)/colSums(t))
(F=(2*precision*recall)/(precision+recall))




#RandomForest
rf=randomForest(Vital_Status~.,data=tr)
p_y=predict(rf,tsx);p_y
py=ifelse(p_y>0.5,1,0)
t1=table(tsy,py);t1
(accuracy=sum(diag(t1))/sum(t1))
(recall=diag(t1)/rowSums(t1))
(precision=diag(t1)/colSums(t1))
(F=(2*precision*recall)/(precision+recall))

#Linear Discriminant Analysis
library("MASS")
ld=lda(Vital_Status~.,data=tr)
ld
summary(ld)
ldp=predict(ld,tsx);ldp
t2=table(tsy,ldp$class);t2
(accuracy=sum(diag(t2))/sum(t2))
(recall=diag(t2)/rowSums(t2))
(precision=diag(t2)/colSums(t2))
(F=(2*precision*recall)/(precision+recall))


#decision tree
library(rpart)
library(rpart.plot)
?rpart()
reg=rpart(formula = Vital_Status~.,data=tr, method="class")
rpart.plot(reg)
pg=predict(reg,tsx,type="class");pg
t3=table(tsy,pg);t3
(accuracy=sum(diag(t3))/sum(t3))
(recall=diag(t3)/rowSums(t3))
(precision=diag(t3)/colSums(t3))
(F=(2*precision*recall)/(precision+recall))


pp=predict(g1,trx,type="response");pp
pp1=ifelse(pp>0.5,1,0)
Y=cbind("LOC"=tr$LOC,"Age"=tr$age,"PH"=tr$pH,"Cancer"=tr$cancer,"Stroke"=tr$stroke,"HF"=tr$heart_failure,"Vital_Status"=pp1);head(Y)
Y1=cbind("LOC"=ts$LOC,"Age"=ts$age,"PH"=ts$pH,"Cancer"=ts$cancer,"Stroke"=ts$stroke,"HF"=ts$heart_failure)
rf1=randomForest(Vital_Status~.,data=Y)
p_y1=predict(rf1,Y1);p_y1
py1=ifelse(p_y1>0.5,1,0)
t11=table(tsy,py1);t11
(accuracy=sum(diag(t11))/sum(t11))
(recall=diag(t11)/rowSums(t11))
(precision=diag(t11)/colSums(t11))
(F=(2*precision*recall)/(precision+recall))


#XGBOOSTING
try1=try=='1'
tsy1=tsy=='1';tsy1
try1=as.matrix(try1)

library(xgboost)
xmatrix1=xgb.DMatrix(data=as.matrix(trx),label=try1)
xmatrix2=xgb.DMatrix(data=as.matrix(tsx),label=tsy1)
xg=xgboost(data=xmatrix1,nrounds = 50,objective="multi:softmax",eta=0.3,num_class=2,max_depth=100)
xgp=predict(xg,xmatrix2)
t7=table(tsy1,xgp);t7
(accuracy=sum(diag(t7))/sum(t7))
(recall=diag(t7)/rowSums(t7))
(precision=diag(t7)/colSums(t7))
(F=(2*precision*recall)/(precision+recall))

Y=cbind("LOC"=tr$LOC,"Age"=tr$age,"PH"=tr$pH,"Vital_Status"=tr$Vital_Status);head(Y)
Y1=cbind("LOC"=ts$LOC,"Age"=ts$age,"PH"=ts$pH)
Y2=cbind("LOC"=tr$LOC,"Age"=tr$age,"PH"=tr$pH);head(Y2)
l=tr$Vital_Status
l1=ts$Vital_Status
xm1=xgb.DMatrix(data=as.matrix(Y2),label=try1)
xm2=xgb.DMatrix(data=as.matrix(Y1),label=tsy1)
xg1=xgboost(data=xm1,nrounds = 50,objective="multi:softmax",eta=0.3,num_class=2,max_depth=100)
xgp1=predict(xg1,xm2)
t8=table(tsy1,xgp1);t8
(accuracy=sum(diag(t8))/sum(t8))
(recall=diag(t8)/rowSums(t8))
(precision=diag(t8)/colSums(t8))
(F=(2*precision*recall)/(precision+recall))



#ANN
library(nnet)
ideal <- class.ind(try)
ideal
seedsANN = nnet(trx,ideal, size=2, softmax=TRUE)
gggg= predict(seedsANN, tsx,type="class")
tann=(table(tsy,gggg))
tann
(accuracy=sum(diag(tann))/sum(tann))
(recall=diag(tann)/rowSums(tann))
(precision=diag(tann)/colSums(tann))
(F=(2*precision*recall)/(precision+recall))


