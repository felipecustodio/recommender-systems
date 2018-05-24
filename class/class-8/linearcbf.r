linearcbf <- function(train, test, features, lr = 0.05, reg = 0.02, miter = 10) {
	nusers = max(c(train[,1], test[,1]))
	nitems = max(c(train[,2], test[,2]))
	nfeatures = ncol(features)

	features = as.matrix(cbind(features, 1))	
	profiles = matrix(rnorm(nusers * (nfeatures + 1), mean=0, sd=0.1), nusers, nfeatures+1)

	for(l in 1:miter) {
		for(j in 1:nrow(train)) {
			u = train[j, 1]
			i = train[j, 2]
			r_ui = train[j, 3]
			pred = profiles[u,] %*% features[i,]
			e_ui = pred - r_ui

			for(k in 1:nfeatures) {
				profiles[u, k] = profiles[u, k] - lr * (e_ui * features[i, k] + reg * profiles[u, k])
			}
			k = nfeatures + 1
			profiles[u, k] = profiles[u, k] - lr * (e_ui * features[i, k]) 
		}
	}

	model = list()
	model$profiles = profiles

	return(model)
}

predict <- function(model, user, item, features) {
	features = as.matrix(cbind(features, 1))
	return(model$profiles[user,] %*% features[item,])
}

