### Math 287B Project Code ###
###         Yang XU        ###
# load packages
require(MASS)
par(mfrow=c(2,2))
#par(mfrow=c(1,1))
n = 1024
p = 2048
C = 0.1
rho = c()
for (i in c(1:2048)) {
  t = i/n
  rho[i] = C*(0.7 * dbeta(t, 1500,3000)+0.5*dbeta(t, 1200,900)+0.5*dbeta(t, 600,160))
  
}
plot(rho, cex = 0.8, pch = 20,main = 'A single gengerated component')
lines(rho,lwd = 2,lty = 1)

v = rnorm(n,0,1)
sigma =diag(1,p,p)
z = mvrnorm(n, mu=rep(0,p), Sigma = sigma)
length(z)
plot(z[1,],cex = 0.8, pch = 20,col='brown',main = 'A sample noise from model',ylab = 'z_1')
x = matrix(NA,nrow = n,ncol=p)
for (i in 1:n) {
  x[i,] = v[i]*rho + z[i,]
}


# apply standard PCA
pca = prcomp(x)
summary(pca)

library(factoextra)
fviz_eig(pca)


var_coord_func <- function(loadings, comp.sdev){
  loadings*comp.sdev
}
loadings <- pca$rotation
sdev <- pca$sdev
var.coord <- t(apply(loadings, 1, var_coord_func, sdev)) 

plot(var.coord[,1],cex = 0.8, pch = 20,col='red',ylab = 'value',main = 'Sample principal component by standard PCA')
lines(var.coord[,1],col='red',lwd = 2,lty = 1)

# apply sparse PCA
library(sparsepca)
spca = spca(x,k=372)
length(spca$loadings)

var.coord <- t(apply(spca$loadings, 1, var_coord_func, spca$sdev)) 
plot(var.coord[,1],cex = 0.8, pch = 20,col='red', ylab = 'value',main = 'Sample principal component by sparse PCA')
lines(var.coord[,1],cex = 0.8, pch = 20,col='red',lwd = 2,lty=1)

a = eigen(cov(pca$x))


####################
library(ggplot2)
dat = data.frame(seq(1,2048),-var.coord[,1])
ggplot(data=dat, aes(x=dat[1], y=dat[2])) +
  geom_line(color="red", size=1,linetype="solid")+
  geom_point(color="red",size=1)+
  ggtitle("Sample principal component by standard PCA")+
  theme(plot.title = element_text(hjust = 0.5))+
  ylab('Value') + xlab('Element in this PC')



