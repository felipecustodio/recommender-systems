svdopt <- function(R, k, lr = 0.05, reg = 0.002, miter = 10) {
	global_mean <- mean(as.matrix(R[,3]), na.rm = TRUE)
	bu <- rep(0, max(R[,1]))
	bi <- rep(0, max(R[,2]))
	nusers <- max(R[,1])
	nitems <- max(R[,2])
	P <- matrix(rnorm(nusers*k, mean=0, sd=0.1), nusers, k)
	Q <- matrix(rnorm(nitems*k, mean=0, sd=0.1), nitems, k)
	error <- list()

	for(l in 1:miter) {
		cat("Iter: ", l, "\n")
		sq_error <- 0
		for(j in 1:nrow(R)) {
			u <- R[j,1]
			i <- R[j,2]
			r_ui <- R[j,3]
			pred <- global_mean + bu[u] + bi[i] + P[u, ] %*% Q[i,]
			e_ui <- r_ui - pred
			sq_error <- sq_error + e_ui^2
			bu[u] <- bu[u] + lr * e_ui
			bi[i] <- bi[i] + lr * e_ui
			for(f in 1:k) {
				temp_uf <- P[u,f]
         		P[u,f] <- P[u,f] + lr * (e_ui * Q[i,f] - reg * P[u,f])
			    Q[i,f] <- Q[i,f] + lr * (e_ui * temp_uf - reg * Q[i,f])
			}
		}
		error <- c(error, sqrt(sq_error/nrow(R)))
	}

	return(list(global_mean = global_mean, bu = bu, bi = bi, P = P, Q = Q, error = error))
}

predict <- function(model, user, item) {
	pred = model$global_mean + model$bu[user] + model$bi[item] + model$P[user,] %*% model$Q[item,]
	return (pred)
}
