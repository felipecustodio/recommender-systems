funkSVD <- function(R, k, lr = 0.01, reg = 0.5, miter = 2) {
	global_mean <- mean(as.matrix(R), na.rm = TRUE)
	bu <- rep(0, nrow(R))
	bi <- rep(0, ncol(R))
	P <- matrix(0.1, nrow = nrow(R), ncol = k)
	Q <- matrix(0.1, nrow = ncol(R), ncol = k)
	K <- which(!is.na(R), arr.ind = TRUE)
	for(f in sample(k)) {
		for(l in 1:miter) {
			for(j in 1:nrow(K)){
      			u <- K[j,1]
      			i <- K[j,2]
      			r_ui <- R[u,i]
      			if(!is.na(r_ui)){
         			pred <- global_mean + bu[u] + bi[i] + P[u, ] %*% Q[i, ]
         			e_ui <- r_ui - pred
					bu[u] <- bu[u] + lr * e_ui
					bi[i] <- bi[i] + lr * e_ui
         			temp_uf <- P[u,f]
         			P[u,f] <- P[u,f] + lr * (e_ui * Q[i,f] - reg * P[u,f])
			       	Q[i,f] <- Q[i,f] + lr * (e_ui * temp_uf - reg * Q[i,f])
      			}
   			}
		}
	}
	return(list(bu = bu, bi = bi, P = P, Q = Q))
}

