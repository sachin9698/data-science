bank_data=read.csv('bank-additional-full.csv',sep=';')
summary(bank_data)
library(ggplot2)

ggplot(bank_data, aes(x=age,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=job,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=marital,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=education,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=default,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=housing,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=loan,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=contact,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=month,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=day_of_week,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=duration,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=campaign,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=pdays,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=previous,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=poutcome,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=emp.var.rate,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=cons.conf.idx,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=cons.price.idx,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=euribor3m,fill=y))+
  geom_bar()

ggplot(bank_data, aes(x=nr.employed,fill=y))+
  geom_bar()

rf.1=randomForest(y~age+job+duration+contact+month+day_of_week+campaign,data = bank_data)
table(predict(rf.1),bank_data$y)
