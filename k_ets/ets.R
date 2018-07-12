myArgs <- commandArgs(trailingOnly = TRUE)

# Convert to numerics
nums = as.numeric(myArgs)

# cat will write the result to the stdout stream
cat(max(nums))


library('forecast')

# nums <- c(27, 27, 7, 24, 39, 40, 24, 45, 36, 37, 31, 47, 16, 24, 6, 21, 35, 36, 21, 40, 32, 33, 27, 42, 14, 21, 5, 19, 31, 32, 19, 36, 29, 29, 24, 42, 15)

nums<- ts(nums, frequency=12)

fit<-ets(nums)

# summary(fit)

future<- forecast(fit, h=3)

# plot(future)

cat(as.numeric(future$mean))

