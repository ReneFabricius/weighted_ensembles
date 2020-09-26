library(RcppCNPy)
library("scatterplot3d")
coeffs <- npyLoad("coeffs.npy")
coeffs7 <- apply(coeffs, 1, function(row) sum(row[3:5]))

plot(coeffs[,3]/coeffs7, coeffs[,4]/coeffs7)


scatterplot3d(coeffs[, 3:5])
