myArgs <- commandArgs(trailingOnly = TRUE)


# Convert to numerics
nums = as.numeric(myArgs)

# nums <- c(27, 27, 7, 24, 39, 40, 24, 45, 36, 37, 31, 47, 16, 24, 6, 21, 35, 36, 21, 40, 32, 33, 27, 42, 14, 21, 5, 19, 31, 32, 19, 36, 29, 29, 24, 42, 15)
# cat(nums)


library('forecast')

ahead <- nums[1]
ts_frequency <- nums[2]
ts_v <- nums[-(1:2)]

ts <- ts(ts_v, frequency=ts_frequency)

fit <- ets(ts)

# summary(fit)

future <- forecast(fit, h=ahead)

# plot(future)

cat(as.numeric(future$mean))




if(FALSE) {
    library('forecast')
    nums <- c(27, 27, 7, 24, 39, 40, 24, 45, 36, 37, 31, 47, 16, 24, 6, 21, 35, 36, 21, 40, 32, 33, 27, 42, 14, 21, 5, 19, 31, 32, 19, 36, 29, 29, 24, 42, 15)
    tsx <- ts(nums, frequency=12)
    ts_fit <- ets(tsx)
    output <- forecast(ts_fit, h=24)
    output
}


