svdpp <- function(R, N, k, lr = 0.05, reg = 0.002, miter = 10) {
	global_mean <- mean(as.matrix(R[,3]), na.rm = TRUE)
	bu <- rep(0, max(R[,1]))
	bi <- rep(0, max(R[,2]))
	nusers <- max(R[,1])
	nitems <- max(R[,2])
	P <- matrix(rnorm(nusers*k, mean=0, sd=0.1), nusers, k)
	Q <- matrix(rnorm(nitems*k, mean=0, sd=0.1), nitems, k)
	Y <- matrix(rnorm(nitems*k, mean=0, sd=0.1), nitems, k)
 
	error <- list()

	for(l in 1:miter) {
		sq_error <- 0
		cat("Iter: ", l)
		for(r in 1:nrow(R)) {
			u <- R[r,1]
			i <- R[r,2]
			r_ui <- R[r,3]
			N_u <- N[N[,1] == u,2]
			y_sum <- rep(0, k)
			for(j in N_u) {
				y_sum <- y_sum + Y[j,]
			}
			p_plus_y <- P[u, ] + (1/sqrt(length(N_u))) * y_sum
			pred <- global_mean + bu[u] + bi[i] + p_plus_y %*% Q[i,]
			e_ui <- r_ui - pred
			sq_error <- sq_error + e_ui^2
			bu[u] <- bu[u] + lr * e_ui
			bi[i] <- bi[i] + lr * e_ui
			for(f in 1:k) {
         		P[u,f] <- P[u,f] + lr * (e_ui * Q[i,f] - reg * P[u,f])

				temp_if <- Q[i,f]
			    Q[i,f] <- Q[i,f] + lr * (e_ui * p_plus_y[f] - reg * Q[i,f])

				for(j in N_u) {
					Y[j,f] <- Y[j,f] + lr * (e_ui * (1/sqrt(length(N_u))) * temp_if - reg * Y[j,f])
				}
			}
		}
		rmse <- sqrt(sq_error/nrow(R))
		cat(" RMSE: ", rmse, "\n")
		error <- c(error, rmse)
	}

	return(list(bu = bu, bi = bi, P = P, Q = Q, Y = Y, error = error))
}
