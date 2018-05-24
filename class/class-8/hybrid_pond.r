source("linearcbf.r")
predict_linear = predict
source("svdopt.r")
predict_svd = predict

hybrid_pond <- function(train, test, features) {
	cat("Training Linear Model...", "\n")
	lmodel = linearcbf(train, test, features)
	cat("Training SVD Model...", "\n")
	svd_model = svdopt(train, k = 2)

	model = list()
	model$lmodel = lmodel
	model$svd_model = svd_model

	return(model)
}

predict <- function(model, user, item, features) {
	return((predict_linear(model$lmodel, user, item, features) + predict_svd(model$svd_model, user, item)) / 2)
}

