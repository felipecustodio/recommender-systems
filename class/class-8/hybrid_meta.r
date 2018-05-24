source("svdopt.r")
predict_svd = predict

hybrid_meta <- function(ratings, features, k = 2) {
	cat("Training SVD Model...", "\n")
	svd = svdopt(ratings, k = 2)

	colnames(ratings) = c("userId", "itemId", "ratings")
	library(reshape2)
	ratings = dcast(ratings, userId~itemId, value.var = "ratings", na.rm=FALSE)
	ratings[is.na(ratings)==T] = 0
	ratings = ratings[,-1]

	library(proxy)
	sim = as.matrix(simil(svd$Q))
	sim[is.na(sim)==T] = 0

	model = list()
	model$sim = sim
	model$k = k
	model$ratings = ratings

	return(model)
}

predict <- function(model, user, item) {
	similar.items = order(-model$sim[item, ])
	rated.items = which(model$ratings[user, ] > 0)
	most.similar.rated = intersect(similar.items, rated.items)[1:model$k]

	sum = 0
	sumWeight = 0
	for(j in most.similar.rated) {
		sum = sum + model$sim[item, j]
		sumWeight = sumWeight + model$sim[item, j] * model$ratings[user, j]
	}

	return(sumWeight / sum)
}







