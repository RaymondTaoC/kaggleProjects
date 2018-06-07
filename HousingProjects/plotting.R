
#!/usr/bin/env Rscript
library(tibble)
library(magrittr)

# Generic Plot yFr against xFr with least squares reg. line.
plotDatLine <- function(linM,  xFr, yFr, xStr, yStr, title=""){
	plot(xFr,yFr,xlab=xStr, ylab=yStr, main=title)
	abline(linM)
}

# Plot standerdized residuals against xFr
plotSRvX = function(linM, xFr, xStr){
	res =  rstandard(linM)
	ls.lm = lm(res~xFr)
	plot(xFr,res, ylab='Standardized Residuslas', xlab=xStr)
	abline(ls.lm)
}

# Plot (|standardized residuals|)^{1/2} against xFr
plotSASRvX <- function(linM, xFr, xStr){
	res = sqrt(abs(rstandard(linM)))
	ls.lm = lm(res~xFr)
	plot(xFr, res, ylab="(|Standardized Residuslas|)^{1/2}", xlab=xStr)
	abline(ls.lm)
}

# Plot Gaussian Density of frame 
plotDensity = function(frame, xStr){
	d = density(frame)
	plot(d, xlab=xStr, ylab='Density (Change me)', main='Gaussian Kernel Density')
}


# Generic plot seperating data by a threshold
threshold_plot <- function(dat, mod, cut, yLab, name, point_size, opacity){
dat$dfb = dffits(mod)
dat$Grouping = cut(dat$dfb,
               		breaks = c(-Inf, -cut, cut, Inf))
	ggplot(dat, aes(x = seq_len(nrow(dat)), y = dfb, color = Grouping)) +
  				geom_hline(yintercept = c(cut, -cut), colour = 'red') +
		geom_point(alpha = opacity, size = point_size) +
		xlab('Observation') + ylab(yLab) +
		ggtitle(name) +
		annotate("text", x = Inf, y = Inf, hjust = 1.5, vjust = 2, 
                  label = paste('Threshold: ', round(cut, 4)))

#dat$color <- ifelse(((dat$dfb >= cut) | (dat$dfb <= -cut)), TRUE, FALSE)
#length(which(dat$color))
}


# Plot dffits
plot_dffits <- function(df, model, point_size = 1, opacity = 1) {
	threshold_plot(df, model, 2 * sqrt((p + 1)/n), 'DFFITS', paste("Influence Diagnostics"), point_size, opacity)
}


# Plot dfbetas
plot_dfbetas <- function(df, model, point_size = 1, opacity = 1) {
	threshold_plot(df, model, 2 / sqrt(nrow(dat)), 'DFBETAS', paste("Influence Diagnostics"), point_size, opacity)
}


# Plot with Local Linear Smoothing
plot_lowess <- function(x, y) {
	plot(x, y, xlab = deparse(substitute(x)), ylab = deparse(substitute(y)))
	lines(lowess(x, y), col='firebrick1',lwd=1)
}

